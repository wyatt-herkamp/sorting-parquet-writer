use arrow::array::{Float64Array, RecordBatch, StringArray, TimestampNanosecondArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::datasource::MemTable;
use datafusion::prelude::{SessionConfig, SessionContext, col};
use parquet::file::metadata::SortingColumn;
use parquet::file::properties::WriterProperties;
use rand::prelude::*;
use sorting_parquet_writer::sorting::sort_record_batch;
use sorting_parquet_writer::writers::SortingParquetWriter;
use std::hint::black_box;
use std::sync::Arc;

// Schema, sort columns, and data generation mirror benches/sorting_writer.rs
// exactly so the two bench files can be compared line-for-line.

fn create_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("ticker", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Nanosecond, Some(Arc::from("UTC"))),
            false,
        ),
    ]))
}

fn sorting_columns() -> Vec<SortingColumn> {
    vec![
        SortingColumn {
            column_idx: 0,
            descending: false,
            nulls_first: false,
        },
        SortingColumn {
            column_idx: 3,
            descending: false,
            nulls_first: false,
        },
        SortingColumn {
            column_idx: 2,
            descending: false,
            nulls_first: false,
        },
    ]
}

fn generate_batch(size: usize) -> RecordBatch {
    let mut rng = rand::rng();
    let tickers = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "META", "NVDA"];

    let ticker_data: Vec<String> = (0..size)
        .map(|_| tickers.choose(&mut rng).unwrap().to_string())
        .collect();
    let price_data: Vec<f64> = (0..size).map(|_| rng.random_range(50.0..2000.0)).collect();
    let sequence_data: Vec<u64> = (0..size).map(|_| rng.random_range(1..1_000_000)).collect();
    let timestamp_data: Vec<i64> = (0..size)
        .map(|_| rng.random_range(1_640_995_200_000_000_000i64..1_672_531_200_000_000_000i64))
        .collect();

    RecordBatch::try_new(
        create_schema(),
        vec![
            Arc::new(StringArray::from(ticker_data)),
            Arc::new(Float64Array::from(price_data)),
            Arc::new(UInt64Array::from(sequence_data)),
            Arc::new(
                TimestampNanosecondArray::from(timestamp_data).with_timezone(Arc::from("UTC")),
            ),
        ],
    )
    .unwrap()
}

fn generate_batches(total_rows: usize, batch_size: usize) -> Vec<RecordBatch> {
    let mut batches = Vec::new();
    let mut remaining = total_rows;
    while remaining > 0 {
        let size = remaining.min(batch_size);
        batches.push(generate_batch(size));
        remaining -= size;
    }
    batches
}

// SPW's max_memory_bytes bounds only its sort buffer; DataFusion's
// memory_limit covers sort + ParquetSink column writers together, so matching
// the SPW budget deterministically OOMs the sink on small budgets. Compare
// wall-clock on identical workloads with DF at its default (unlimited memory,
// default 10 MiB sort_spill_reservation) instead. target_partitions(1) keeps
// DF single-threaded to match SPW's write path; batch_size matches SPW.
fn build_df_ctx() -> SessionContext {
    let config = SessionConfig::new()
        .with_batch_size(1024)
        .with_target_partitions(1);
    SessionContext::new_with_config(config)
}

async fn df_sort_and_write(
    ctx: &SessionContext,
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    out_dir: &std::path::Path,
) {
    let table = MemTable::try_new(schema, vec![batches]).unwrap();
    let df = ctx
        .read_table(Arc::new(table))
        .unwrap()
        .sort(vec![
            col("ticker").sort(true, false),
            col("timestamp").sort(true, false),
            col("sequence").sort(true, false),
        ])
        .unwrap();
    df.write_parquet(
        out_dir.to_str().unwrap(),
        DataFrameWriteOptions::new(),
        None,
    )
    .await
    .unwrap();
}

// ── End-to-end sort + write, SPW vs DataFusion side-by-side ─────────────────
// Criterion pattern: https://bheisler.github.io/criterion.rs/book/user_guide/comparing_functions.html

fn bench_sort_and_write(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("sort_and_write");

    for &total_rows in &[100_000usize, 500_000] {
        let batches = generate_batches(total_rows, 1024);
        group.throughput(Throughput::Elements(total_rows as u64));

        group.bench_with_input(
            BenchmarkId::new("SPW", total_rows),
            &batches,
            |b, batches| {
                b.iter(|| {
                    let temp = tempfile::NamedTempFile::new().unwrap();
                    let file = temp.reopen().unwrap();
                    let props = WriterProperties::builder()
                        .set_sorting_columns(Some(sorting_columns()))
                        .build();
                    let mut writer =
                        SortingParquetWriter::try_new(file, create_schema(), props).unwrap();
                    for batch in batches {
                        writer.write(black_box(batch)).unwrap();
                    }
                    writer.finish().unwrap();
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("DataFusion", total_rows),
            &batches,
            |b, batches| {
                b.iter(|| {
                    let temp = tempfile::TempDir::new().unwrap();
                    let ctx = build_df_ctx();
                    rt.block_on(df_sort_and_write(
                        &ctx,
                        create_schema(),
                        black_box(batches.clone()),
                        temp.path(),
                    ));
                })
            },
        );
    }
    group.finish();
}

// ── Isolated in-memory sort, SPW vs DataFusion side-by-side ─────────────────

fn bench_sort_record_batch(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("sort_record_batch");
    let ctx = build_df_ctx();

    for &size in &[1_000usize, 10_000, 100_000, 500_000] {
        let batch = generate_batch(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("SPW", size), &batch, |b, batch| {
            let cols = sorting_columns();
            b.iter(|| {
                black_box(sort_record_batch(black_box(batch), &cols).unwrap());
            })
        });

        group.bench_with_input(BenchmarkId::new("DataFusion", size), &batch, |b, batch| {
            b.iter(|| {
                rt.block_on(async {
                    let df = ctx
                        .read_batch(black_box(batch.clone()))
                        .unwrap()
                        .sort(vec![
                            col("ticker").sort(true, false),
                            col("timestamp").sort(true, false),
                            col("sequence").sort(true, false),
                        ])
                        .unwrap();
                    black_box(df.collect().await.unwrap());
                })
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sort_and_write, bench_sort_record_batch);
criterion_main!(benches);
