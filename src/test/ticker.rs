use std::sync::{Arc, LazyLock};

use arrow::{
    array::{RecordBatch, StringArray, StringBuilder, TimestampNanosecondBuilder},
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
};
use chrono::{DateTime, Utc};
use rand::{Rng, seq::IndexedRandom};

use crate::test::{
    TestArrowType, TestError,
    random_time::{random_date, random_time_between},
};
const TICKERS: &[&str] = &[
    "AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX", "ADBE", "INTC", "STLA", "FIG",
    "KLAR", "WOOF", "GPRO",
];

#[derive(Debug, Clone, PartialEq)]
pub struct TickerItem {
    pub ticker: String,
    pub price: f64,
    pub sequence: u64,
    pub conditions: Vec<i32>,
    pub timestamp: DateTime<Utc>,
}

impl TestArrowType for TickerItem {
    fn random_instances(n: usize) -> Vec<Self>
    where
        Self: Sized,
    {
        let mut results = Vec::with_capacity(n);
        let date = random_date();
        let ninet = date.and_hms_opt(9, 30, 0).unwrap().and_utc();
        let four = date.and_hms_opt(16, 0, 0).unwrap().and_utc();
        let mut rng = rand::rng();
        let starting_sequence: u64 = rng.random_range(1..=1000);
        let mut random_time = random_time_between(ninet, four);

        for i in 0..n {
            random_time += chrono::Duration::seconds(rng.random_range(0..=60));
            results.push(Self {
                ticker: TICKERS.choose(&mut rng).unwrap().to_string(),
                price: rng.random_range(100.0..1500.0),
                sequence: starting_sequence + i as u64,
                conditions: vec![1, 2, 3],
                timestamp: random_time,
            });
        }
        results
    }
    fn sorting_columns() -> Vec<parquet::format::SortingColumn>
    where
        Self: Sized,
    {
        vec![
            parquet::format::SortingColumn {
                column_idx: 0,
                descending: false,
                nulls_first: false,
            },
            parquet::format::SortingColumn {
                column_idx: 3,
                descending: false,
                nulls_first: false,
            },
            parquet::format::SortingColumn {
                column_idx: 2,
                descending: false,
                nulls_first: false,
            },
        ]
    }
    fn is_sorted(records: &[Self]) -> Option<&[Self]>
    where
        Self: Sized,
    {
        records.windows(2).find(|w| {
            let a = &w[0];
            let b = &w[1];
            if a.ticker != b.ticker {
                a.ticker > b.ticker
            } else if a.timestamp != b.timestamp {
                a.timestamp > b.timestamp
            } else {
                a.sequence > b.sequence
            }
        })
    }

    fn schema() -> arrow::datatypes::SchemaRef {
        static SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
            Arc::new(Schema::new(vec![
                Field::new("ticker", DataType::Utf8, false),
                Field::new("price", DataType::Float64, false),
                Field::new("sequence", DataType::UInt64, false),
                Field::new(
                    "timestamp",
                    DataType::Timestamp(TimeUnit::Nanosecond, Some(Arc::from("UTC"))),
                    false,
                ),
                Field::new(
                    "conditions",
                    DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                    false,
                ),
            ]))
        });
        SCHEMA.clone()
    }

    fn into_record_batch(records: Vec<Self>) -> Result<RecordBatch, TestError>
    where
        Self: Sized,
    {
        let len = records.len();
        let mut tickers = StringBuilder::with_capacity(len, len * 5);

        let mut prices = arrow::array::Float64Builder::with_capacity(len);
        let mut timestamps =
            TimestampNanosecondBuilder::with_capacity(len).with_timezone(Arc::from("UTC"));
        let mut sequences = arrow::array::UInt64Builder::with_capacity(len);
        let mut conditions =
            arrow::array::ListBuilder::new(arrow::array::Int32Builder::with_capacity(len));
        for record in records {
            let timestamp_nanos = record
                .timestamp
                .timestamp_nanos_opt()
                .ok_or_else(|| TestError::ChronoError("Timestamp out of range for nanoseconds"))?;
            tickers.append_value(&record.ticker);
            timestamps.append_value(timestamp_nanos);
            prices.append_value(record.price);
            sequences.append_value(record.sequence);
            conditions.append_value(record.conditions.iter().map(|v| Some(*v)));
        }
        let batch = RecordBatch::try_new(
            Self::schema(),
            vec![
                Arc::new(tickers.finish()),
                Arc::new(prices.finish()),
                Arc::new(sequences.finish()),
                Arc::new(timestamps.finish()),
                Arc::new(conditions.finish()),
            ],
        )?;
        Ok(batch)
    }

    fn from_record_batch(batch: &RecordBatch) -> Result<Vec<Self>, TestError> {
        let ticker_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| TestError::CastError {
                from: batch.column(0).data_type().clone(),
                to: "StringArray",
            })?;
        let price_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .ok_or_else(|| TestError::CastError {
                from: batch.column(1).data_type().clone(),
                to: "Float64Array",
            })?;
        let sequence_array: &arrow::array::PrimitiveArray<arrow::datatypes::UInt64Type> = batch
            .column(2)
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .ok_or_else(|| TestError::CastError {
                from: batch.column(2).data_type().clone(),
                to: "UInt64Array",
            })?;
        let timestamp_array = batch
            .column(3)
            .as_any()
            .downcast_ref::<arrow::array::TimestampNanosecondArray>()
            .ok_or_else(|| TestError::CastError {
                from: batch.column(3).data_type().clone(),
                to: "TimestampNanosecondArray",
            })?;

        let conditions_array = batch
            .column(4)
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .ok_or_else(|| TestError::CastError {
                from: batch.column(4).data_type().clone(),
                to: "ListArray",
            })?;

        let mut results = Vec::with_capacity(batch.num_rows());
        for i in 0..batch.num_rows() {
            let timestamp = timestamp_array.value(i);
            let datetime = DateTime::<Utc>::from_timestamp_nanos(timestamp);
            let conditions_values = {
                let value_array = conditions_array.value(i);
                let int_array = value_array
                    .as_any()
                    .downcast_ref::<arrow::array::Int32Array>()
                    .ok_or_else(|| TestError::CastError {
                        from: value_array.data_type().clone(),
                        to: "Int32Array",
                    })?;
                (0..int_array.len()).map(|j| int_array.value(j)).collect()
            };
            results.push(Self {
                ticker: ticker_array.value(i).to_string(),
                price: price_array.value(i),
                sequence: sequence_array.value(i),
                timestamp: datetime,
                conditions: conditions_values,
            });
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, time::Duration};

    use parquet::arrow::{ArrowWriter, arrow_reader::{ArrowReaderBuilder, ArrowReaderOptions}};

    use crate::writers::{SortedGroupsParquetWriter, SortingParquetWriter};

    use super::*;
    #[test]
    fn test_random_ticker_item() {
        let item = TickerItem::random_instances(100);
        let batch = TickerItem::into_record_batch(item).unwrap();
        let sorting_columns = TickerItem::sorting_columns();

        let sorted = crate::sorting::sort_record_batch(&batch, &sorting_columns).unwrap();
        arrow::util::pretty::print_batches(std::slice::from_ref(&sorted)).unwrap();

        let items_back = TickerItem::from_record_batch(&sorted).unwrap();
        assert_eq!(items_back.len(), 100);
        // Assert that items are sorted
        assert_eq!(TickerItem::is_sorted(&items_back), None);
    }

    #[test]
    fn large_sort() {
        let item = TickerItem::random_instances(1024 * 1024);
        println!("Generated {} items", item.len());
        let instant = std::time::Instant::now();
        let mut batches = Vec::new();
        for batch in item.chunks(128) {
            let batch = TickerItem::into_record_batch(batch.to_vec()).unwrap();
            let sorted =
                crate::sorting::sort_record_batch(&batch, &TickerItem::sorting_columns()).unwrap();
            batches.push(sorted);
        }
        println!("Sorted chunks in {:?}", instant.elapsed());

        let instant = std::time::Instant::now();
        let merged =
            crate::record_batch::merge_sorted_batches(&batches, &TickerItem::sorting_columns())
                .unwrap();
        println!("Merged in {:?}", instant.elapsed());
        let items_back = TickerItem::from_record_batch(&merged).unwrap();
        assert_eq!(items_back.len(), 1024 * 1024);
        assert_eq!(TickerItem::is_sorted(&items_back), None);
    }

    #[test]
    fn create_test_files() -> anyhow::Result<()> {
        let file = std::fs::File::create("test_output.sorted.parquet")?;
        let unsorted_file = std::fs::File::create("test_output.unsorted.parquet")?;
        let props = parquet::file::properties::WriterProperties::builder()
            .set_sorting_columns(Some(TickerItem::sorting_columns()))
            .build();
        let schema = TickerItem::schema();
        let mut sorted_writer = SortedGroupsParquetWriter::try_new(file, schema, props)?;
        let mut unsorted_writer = ArrowWriter::try_new(unsorted_file, TickerItem::schema(), None)?;
        let mut duration_sum_sorted = Duration::ZERO;
        let mut duration_sum_unsorted = Duration::ZERO;

        for i in 0..200 {
            eprintln!("Writing batch {}/200", i + 1);
            let items = TickerItem::random_instances(1024 * 1024);
            for chunk in items.chunks(128) {
                let batch = TickerItem::into_record_batch(chunk.to_vec())?;
                let start = std::time::Instant::now();
                sorted_writer.write(&batch)?;
                duration_sum_sorted += start.elapsed();
                let start = std::time::Instant::now();
                unsorted_writer.write(&batch)?;
                duration_sum_unsorted += start.elapsed();
            }
        }
        sorted_writer.close()?;
        unsorted_writer.close()?;
        println!(
            "Total sorted write time: {}",
            humantime::format_duration(duration_sum_sorted)
        );
        println!(
            "Total unsorted write time: {}",
            humantime::format_duration(duration_sum_unsorted)
        );
        Ok(())
    }
    #[test]
    fn create_test_sorted() -> anyhow::Result<()> {
        let path = PathBuf::from("test_output.sorted.parquet");
        let file = std::fs::File::create(&path)?;
        let props = parquet::file::properties::WriterProperties::builder()
            .set_sorting_columns(Some(TickerItem::sorting_columns()))
            .build();
        let schema = TickerItem::schema();
        let mut sorted_writer: SortingParquetWriter =
            SortingParquetWriter::try_new(file, schema, props)?;
        let mut duration_sum_sorted = Duration::ZERO;

        for i in 0..50 {
            eprintln!("Writing batch {}/50", i + 1);
            let items = TickerItem::random_instances(1024 * 1024);
            for chunk in items.chunks(128) {
                let batch = TickerItem::into_record_batch(chunk.to_vec())?;
                let start = std::time::Instant::now();
                sorted_writer.write(&batch)?;
                duration_sum_sorted += start.elapsed();
            }
        }
        sorted_writer.finish()?;
        println!(
            "Total sorted write time: {}",
            humantime::format_duration(duration_sum_sorted)
        );

        let mut reader = ArrowReaderBuilder::try_new_with_options(
            std::fs::File::open(&path)?,
            ArrowReaderOptions::new().with_schema(TickerItem::schema())
        ).unwrap().with_batch_size(200).build()?;
        while let Some(batch) = reader.next(){
            let batch = batch?;
            let items_back = TickerItem::from_record_batch(&batch)?;
            assert_eq!(items_back.len(), batch.num_rows());
            assert_eq!(TickerItem::is_sorted(&items_back), None);
        }
        Ok(())
    }
}
