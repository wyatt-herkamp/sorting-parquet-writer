use arrow::array::{
    Array, Float64Array, RecordBatch, StringArray, TimestampNanosecondArray, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parquet::file::metadata::SortingColumn;
use rand::{prelude::*, rng};
use sorting_parquet_writer::sorting::sort_record_batch;
use std::{hint::black_box, sync::Arc};

/// Generate test data with varying distributions
#[derive(Clone)]
struct TestData {
    batch: RecordBatch,
    single_column_sort: Vec<SortingColumn>,
    multi_column_sort: Vec<SortingColumn>,
}

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

fn generate_random_data(size: usize) -> TestData {
    let mut rng = rng();
    let tickers = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "META", "NVDA"];

    // Generate random ticker data
    let ticker_data: Vec<String> = (0..size)
        .map(|_| tickers.choose(&mut rng).unwrap().to_string())
        .collect();

    let price_data: Vec<f64> = (0..size).map(|_| rng.random_range(50.0..2000.0)).collect();

    let sequence_data: Vec<u64> = (0..size).map(|_| rng.random_range(1..1000000)).collect();

    let timestamp_data: Vec<i64> = (0..size)
        .map(|_| rng.random_range(1640995200000000000i64..1672531200000000000i64)) // 2022-2023 range
        .collect();

    let batch = create_batch(ticker_data, price_data, sequence_data, timestamp_data);

    TestData {
        batch,
        single_column_sort: vec![SortingColumn {
            column_idx: 0, // ticker
            descending: false,
            nulls_first: false,
        }],
        multi_column_sort: vec![
            SortingColumn {
                column_idx: 0, // ticker
                descending: false,
                nulls_first: false,
            },
            SortingColumn {
                column_idx: 3, // timestamp
                descending: false,
                nulls_first: false,
            },
            SortingColumn {
                column_idx: 2, // sequence
                descending: false,
                nulls_first: false,
            },
        ],
    }
}

fn generate_sorted_data(size: usize) -> TestData {
    let tickers = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA"];

    let mut ticker_data: Vec<String> = Vec::with_capacity(size);
    let mut price_data: Vec<f64> = Vec::with_capacity(size);
    let mut sequence_data: Vec<u64> = Vec::with_capacity(size);
    let mut timestamp_data: Vec<i64> = Vec::with_capacity(size);

    // Generate already sorted data
    for i in 0..size {
        ticker_data.push(tickers[i % tickers.len()].to_string());
        price_data.push(100.0 + (i as f64) * 0.1);
        sequence_data.push(i as u64);
        timestamp_data.push(1640995200000000000i64 + (i as i64) * 1000000000); // 1 second intervals
    }

    ticker_data.sort();

    let batch = create_batch(ticker_data, price_data, sequence_data, timestamp_data);

    TestData {
        batch,
        single_column_sort: vec![SortingColumn {
            column_idx: 0,
            descending: false,
            nulls_first: false,
        }],
        multi_column_sort: vec![
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
        ],
    }
}

fn generate_reverse_sorted_data(size: usize) -> TestData {
    let mut data = generate_sorted_data(size);

    // Reverse the ticker column to make it reverse sorted
    let ticker_array = data
        .batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let mut ticker_vec: Vec<String> = (0..ticker_array.len())
        .map(|i| ticker_array.value(i).to_string())
        .collect();
    ticker_vec.reverse();

    let price_data: Vec<f64> = (0..data.batch.num_rows())
        .map(|i| {
            data.batch
                .column(1)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(i)
        })
        .collect();

    let sequence_data: Vec<u64> = (0..data.batch.num_rows())
        .map(|i| {
            data.batch
                .column(2)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(i)
        })
        .collect();

    let timestamp_data: Vec<i64> = (0..data.batch.num_rows())
        .map(|i| {
            data.batch
                .column(3)
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .unwrap()
                .value(i)
        })
        .collect();

    data.batch = create_batch(ticker_vec, price_data, sequence_data, timestamp_data);
    data
}

fn create_batch(
    ticker_data: Vec<String>,
    price_data: Vec<f64>,
    sequence_data: Vec<u64>,
    timestamp_data: Vec<i64>,
) -> RecordBatch {
    let ticker_array = StringArray::from(ticker_data);
    let price_array = Float64Array::from(price_data);
    let sequence_array = UInt64Array::from(sequence_data);
    let timestamp_array =
        TimestampNanosecondArray::from(timestamp_data).with_timezone(Arc::from("UTC"));

    RecordBatch::try_new(
        create_schema(),
        vec![
            Arc::new(ticker_array),
            Arc::new(price_array),
            Arc::new(sequence_array),
            Arc::new(timestamp_array),
        ],
    )
    .unwrap()
}

fn bench_sort_record_batch_by_size(c: &mut Criterion) {
    let sizes = vec![1_000, 10_000, 100_000, 500_000];

    for size in sizes {
        let mut group = c.benchmark_group("sort_by_size");
        group.throughput(Throughput::Elements(size as u64));

        // Random data single column
        let random_data = generate_random_data(size);
        group.bench_with_input(
            BenchmarkId::new("random_single_col", size),
            &random_data,
            |b, data| {
                b.iter(|| {
                    sort_record_batch(black_box(&data.batch), black_box(&data.single_column_sort))
                        .unwrap()
                })
            },
        );

        // Random data multi column
        group.bench_with_input(
            BenchmarkId::new("random_multi_col", size),
            &random_data,
            |b, data| {
                b.iter(|| {
                    sort_record_batch(black_box(&data.batch), black_box(&data.multi_column_sort))
                        .unwrap()
                })
            },
        );

        group.finish();
    }
}

fn bench_sort_record_batch_by_distribution(c: &mut Criterion) {
    let size = 100_000;
    let mut group = c.benchmark_group("sort_by_distribution");
    group.throughput(Throughput::Elements(size as u64));

    // Random data
    let random_data = generate_random_data(size);
    group.bench_function("random", |b| {
        b.iter(|| {
            sort_record_batch(
                black_box(&random_data.batch),
                black_box(&random_data.single_column_sort),
            )
            .unwrap()
        })
    });

    // Already sorted data (best case)
    let sorted_data = generate_sorted_data(size);
    group.bench_function("already_sorted", |b| {
        b.iter(|| {
            sort_record_batch(
                black_box(&sorted_data.batch),
                black_box(&sorted_data.single_column_sort),
            )
            .unwrap()
        })
    });

    // Reverse sorted data (worst case for some algorithms)
    let reverse_data = generate_reverse_sorted_data(size);
    group.bench_function("reverse_sorted", |b| {
        b.iter(|| {
            sort_record_batch(
                black_box(&reverse_data.batch),
                black_box(&reverse_data.single_column_sort),
            )
            .unwrap()
        })
    });

    group.finish();
}

fn bench_sort_record_batch_by_columns(c: &mut Criterion) {
    let size = 50_000;
    let mut group = c.benchmark_group("sort_by_num_columns");
    group.throughput(Throughput::Elements(size as u64));

    let data = generate_random_data(size);

    // Single column sort
    group.bench_function("single_column", |b| {
        b.iter(|| {
            sort_record_batch(black_box(&data.batch), black_box(&data.single_column_sort)).unwrap()
        })
    });

    // Multi-column sort (3 columns)
    group.bench_function("multi_column", |b| {
        b.iter(|| {
            sort_record_batch(black_box(&data.batch), black_box(&data.multi_column_sort)).unwrap()
        })
    });

    group.finish();
}

fn bench_sort_record_batch_data_types(c: &mut Criterion) {
    let size = 50_000;
    let mut group = c.benchmark_group("sort_by_data_type");
    group.throughput(Throughput::Elements(size as u64));

    let data = generate_random_data(size);

    // String column sort
    group.bench_function("string_sort", |b| {
        let string_sort = vec![SortingColumn {
            column_idx: 0, // ticker (string)
            descending: false,
            nulls_first: false,
        }];
        b.iter(|| sort_record_batch(black_box(&data.batch), black_box(&string_sort)).unwrap())
    });

    // Float column sort
    group.bench_function("float_sort", |b| {
        let float_sort = vec![SortingColumn {
            column_idx: 1, // price (f64)
            descending: false,
            nulls_first: false,
        }];
        b.iter(|| sort_record_batch(black_box(&data.batch), black_box(&float_sort)).unwrap())
    });

    // Integer column sort
    group.bench_function("integer_sort", |b| {
        let int_sort = vec![SortingColumn {
            column_idx: 2, // sequence (u64)
            descending: false,
            nulls_first: false,
        }];
        b.iter(|| sort_record_batch(black_box(&data.batch), black_box(&int_sort)).unwrap())
    });

    // Timestamp column sort
    group.bench_function("timestamp_sort", |b| {
        let timestamp_sort = vec![SortingColumn {
            column_idx: 3, // timestamp
            descending: false,
            nulls_first: false,
        }];
        b.iter(|| sort_record_batch(black_box(&data.batch), black_box(&timestamp_sort)).unwrap())
    });

    group.finish();
}

fn bench_sort_record_batch_sort_orders(c: &mut Criterion) {
    let size = 50_000;
    let mut group = c.benchmark_group("sort_by_order");
    group.throughput(Throughput::Elements(size as u64));

    let data = generate_random_data(size);

    // Ascending sort
    group.bench_function("ascending", |b| {
        let asc_sort = vec![SortingColumn {
            column_idx: 0,
            descending: false,
            nulls_first: false,
        }];
        b.iter(|| sort_record_batch(black_box(&data.batch), black_box(&asc_sort)).unwrap())
    });

    // Descending sort
    group.bench_function("descending", |b| {
        let desc_sort = vec![SortingColumn {
            column_idx: 0,
            descending: true,
            nulls_first: false,
        }];
        b.iter(|| sort_record_batch(black_box(&data.batch), black_box(&desc_sort)).unwrap())
    });

    // Nulls first vs nulls last would require data with nulls, skipping for now

    group.finish();
}

criterion_group!(
    benches,
    bench_sort_record_batch_by_size,
    bench_sort_record_batch_by_distribution,
    bench_sort_record_batch_by_columns,
    bench_sort_record_batch_data_types,
    bench_sort_record_batch_sort_orders
);
criterion_main!(benches);
