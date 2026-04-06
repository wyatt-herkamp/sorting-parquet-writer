use arrow::array::{Float64Array, RecordBatch, StringArray, TimestampNanosecondArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::SortingColumn;
use parquet::file::properties::WriterProperties;
use rand::prelude::*;
use sorting_parquet_writer::record_batch::streaming_merge::{RunInfo, SortedRunMerger};
use sorting_parquet_writer::sorting::{create_row_converter, sort_record_batch};
use sorting_parquet_writer::writers::{SortedGroupsParquetWriter, SortingParquetWriter};
use std::hint::black_box;
use std::sync::Arc;

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

/// Pre-generate batches for consistent benchmarking
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

/// Create sorted run files on disk, return RunInfo and a temp dir (to keep them alive)
fn create_sorted_run_files(
    num_runs: usize,
    rows_per_run: usize,
) -> (Vec<RunInfo>, tempfile::TempDir) {
    let schema = create_schema();
    let cols = sorting_columns();
    let row_converter = create_row_converter(&cols, schema.as_ref()).unwrap();
    let temp_dir = tempfile::TempDir::with_prefix("bench_runs").unwrap();

    let mut runs = Vec::with_capacity(num_runs);
    for i in 0..num_runs {
        let batch = generate_batch(rows_per_run);
        let sorted = sort_record_batch(&batch, &cols).unwrap();

        // Capture min/max sort keys
        let sort_cols: Vec<_> = cols
            .iter()
            .map(|col| sorted.column(col.column_idx as usize).clone())
            .collect();
        let rows = row_converter.convert_columns(&sort_cols).unwrap();
        let min_sort_key = rows.row(0).as_ref().to_vec();
        let max_sort_key = rows.row(sorted.num_rows() - 1).as_ref().to_vec();

        let path = temp_dir.path().join(format!("run_{i}.parquet"));
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();
        writer.write(&sorted).unwrap();
        writer.close().unwrap();
        runs.push(RunInfo {
            path,
            min_sort_key: Arc::new(min_sort_key),
            max_sort_key: Arc::new(max_sort_key),
        });
    }

    (runs, temp_dir)
}

// ── End-to-end SortingParquetWriter ─────────────────────────────────────────

fn bench_sorting_writer(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorting_writer");

    for &(total_rows, max_mem) in &[
        (100_000usize, 10_000usize),
        (100_000, 50_000),
        (500_000, 50_000),
        (500_000, 100_000),
    ] {
        let batches = generate_batches(total_rows, 1024);
        let num_runs = total_rows.div_ceil(max_mem);

        group.throughput(Throughput::Elements(total_rows as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("{total_rows}_rows/{num_runs}_runs"), max_mem),
            &batches,
            |b, batches| {
                b.iter(|| {
                    let temp = tempfile::NamedTempFile::new().unwrap();
                    let file = temp.reopen().unwrap();
                    let props = WriterProperties::builder()
                        .set_sorting_columns(Some(sorting_columns()))
                        .build();
                    let options = sorting_parquet_writer::writers::SortingWriterOptions {
                        flush_threshold: sorting_parquet_writer::writers::FlushThreshold::Rows(
                            max_mem,
                        ),
                        ..Default::default()
                    };
                    let mut writer = SortingParquetWriter::try_new_with_options(
                        file,
                        create_schema(),
                        props,
                        options,
                    )
                    .unwrap();
                    for batch in batches {
                        writer.write(black_box(batch)).unwrap();
                    }
                    writer.finish().unwrap();
                })
            },
        );
    }
    group.finish();
}

fn bench_sorting_writer_mem_limit(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorting_writer_mem_limit");

    for &(total_rows, max_mem) in &[
        (100_000usize, 10 * 1024 * 1024), // 10 MB
        (100_000, 50 * 1024 * 1024),      // 50 MB
        (500_000, 50 * 1024 * 1024),      // 50 MB
        (500_000, 100 * 1024 * 1024),     // 100 MB
    ] {
        let batches = generate_batches(total_rows, 1024);
        let num_runs = total_rows.div_ceil(max_mem);

        group.throughput(Throughput::Elements(total_rows as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("{total_rows}_rows/{num_runs}_runs"), max_mem),
            &batches,
            |b, batches| {
                b.iter(|| {
                    let temp = tempfile::NamedTempFile::new().unwrap();
                    let file = temp.reopen().unwrap();
                    let props = WriterProperties::builder()
                        .set_sorting_columns(Some(sorting_columns()))
                        .build();
                    let options = sorting_parquet_writer::writers::SortingWriterOptions {
                        flush_threshold: sorting_parquet_writer::writers::FlushThreshold::Bytes(
                            max_mem,
                        ),
                        ..Default::default()
                    };
                    let mut writer = SortingParquetWriter::try_new_with_options(
                        file,
                        create_schema(),
                        props,
                        options,
                    )
                    .unwrap();
                    for batch in batches {
                        writer.write(black_box(batch)).unwrap();
                    }
                    writer.finish().unwrap();
                })
            },
        );
    }
    group.finish();
}

// ── Isolated merge phase ────────────────────────────────────────────────────

fn bench_merge_phase(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_phase");

    for &(num_runs, rows_per_run) in &[
        (2, 50_000),
        (5, 50_000),
        (10, 50_000),
        (20, 10_000),
        (20, 50_000),
    ] {
        let total_rows = num_runs * rows_per_run;
        let (runs, _temp_dir) = create_sorted_run_files(num_runs, rows_per_run);
        let schema = create_schema();
        let cols = sorting_columns();
        group.throughput(Throughput::Elements(total_rows as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("{num_runs}_runs"), rows_per_run),
            &runs,
            |b, runs| {
                b.iter(|| {
                    let row_converter = create_row_converter(&cols, schema.as_ref()).unwrap();
                    let merger = SortedRunMerger::try_new(
                        black_box(runs.clone()),
                        cols.clone(),
                        row_converter,
                        100_000,
                    )
                    .unwrap();
                    for batch in merger {
                        black_box(batch.unwrap());
                    }
                })
            },
        );
    }
    group.finish();
}

// ── SortedGroupsParquetWriter for comparison ────────────────────────────────

fn bench_sorted_groups_writer(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorted_groups_writer");

    for &total_rows in &[100_000, 500_000] {
        let batches = generate_batches(total_rows, 1024);

        group.throughput(Throughput::Elements(total_rows as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(total_rows),
            &batches,
            |b, batches| {
                b.iter(|| {
                    let temp = tempfile::NamedTempFile::new().unwrap();
                    let file = temp.reopen().unwrap();
                    let props = WriterProperties::builder()
                        .set_sorting_columns(Some(sorting_columns()))
                        .build();
                    let mut writer =
                        SortedGroupsParquetWriter::try_new(file, create_schema(), props).unwrap();
                    for batch in batches {
                        writer.write(black_box(batch)).unwrap();
                    }
                    writer.close().unwrap();
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sorting_writer_mem_limit,
    bench_sorting_writer,
    bench_merge_phase,
    bench_sorted_groups_writer
);
criterion_main!(benches);
