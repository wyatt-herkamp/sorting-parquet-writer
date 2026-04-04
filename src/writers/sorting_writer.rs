use std::fs::File;
use std::io::Write;
use std::mem;
use std::rc::Rc;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use crate::SortingParquetError;
use crate::record_batch::streaming_merge::{RunInfo, SortedRunMerger};

/// Default maximum number of rows to buffer in memory before flushing to a sorted run file.
const DEFAULT_MAX_MEMORY_ROWS: usize = 1_000_000;

/// A Parquet writer that produces a globally sorted output file.
///
/// Uses an external merge sort strategy:
/// 1. **Write phase:** Buffers incoming `RecordBatch`es in memory up to `max_memory_rows`.
///    When the limit is reached, the buffer is sorted and written to a temporary "run" file.
/// 2. **Merge phase (at `finish()`):** All sorted run files are merged via a streaming
///    k-way merge, producing the final globally sorted output.
///
/// Memory usage is bounded by `max_memory_rows` during the write phase, and by
/// approximately one batch per run file during the merge phase.
pub struct SortingParquetWriter<W: Write + Send> {
    schema: SchemaRef,
    properties: WriterProperties,
    target: ArrowWriter<W>,
    row_converter: Option<arrow_row::RowConverter>,
    // In-memory buffer
    buffer: Vec<RecordBatch>,
    buffered_rows: usize,
    max_memory_rows: usize,

    // Sorted run files on disk
    temp_dir: TempDir,
    run_files: Vec<RunInfo>,
    run_count: usize,
}

impl<W: Write + Send> SortingParquetWriter<W> {
    /// Create a new `SortingParquetWriter` with the default memory limit (1M rows).
    pub fn try_new(
        writer: W,
        schema: SchemaRef,
        properties: WriterProperties,
    ) -> Result<Self, SortingParquetError> {
        Self::try_new_with_memory_limit(writer, schema, properties, DEFAULT_MAX_MEMORY_ROWS)
    }

    /// Create a new `SortingParquetWriter` with a custom memory limit.
    ///
    /// `max_memory_rows` controls how many rows are buffered in memory before
    /// being sorted and flushed to a temporary run file on disk.
    pub fn try_new_with_memory_limit(
        writer: W,
        schema: SchemaRef,
        properties: WriterProperties,
        max_memory_rows: usize,
    ) -> Result<Self, SortingParquetError> {
        if properties.sorting_columns().is_none() {
            return Err(SortingParquetError::NoSortingColumnsConfigured);
        }
        let target = ArrowWriter::try_new(writer, schema.clone(), Some(properties.clone()))?;
        let temp_dir = TempDir::with_prefix("sorting_parquet_writer")?;
        let row_converter = crate::sorting::create_row_converter(
            properties
                .sorting_columns()
                .ok_or(SortingParquetError::NoSortingColumnsConfigured)?,
            schema.as_ref(),
        )?;
        Ok(Self {
            schema,
            properties,
            target,
            row_converter: Some(row_converter),
            buffer: Vec::new(),
            buffered_rows: 0,
            max_memory_rows,
            temp_dir,
            run_files: Vec::new(),
            run_count: 0,
        })
    }

    /// Writes a `RecordBatch` to the writer. Data is buffered in memory and
    /// periodically flushed to sorted run files on disk.
    pub fn write(&mut self, batch: &RecordBatch) -> Result<(), SortingParquetError> {
        self.buffer.push(batch.clone());
        self.buffered_rows += batch.num_rows();

        if self.buffered_rows >= self.max_memory_rows {
            self.flush_to_run()?;
        }
        Ok(())
    }

    /// Sort the in-memory buffer and write it to a new run file.
    fn flush_to_run(&mut self) -> Result<(), SortingParquetError> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let sorting_columns = self
            .properties
            .sorting_columns()
            .ok_or(SortingParquetError::NoSortingColumnsConfigured)?
            .clone();

        // Concatenate all buffered batches, then drop the originals to free memory
        // before the sort creates another copy.
        let combined = arrow::compute::concat_batches(&self.schema, &self.buffer)?;
        self.buffer.clear();
        self.buffered_rows = 0;

        // Sort the combined batch and extract min/max sort keys in one pass
        let (sorted, (min_sort_key, max_sort_key)) =
            crate::sorting::sort_record_batch_with_row_converter_returning_extremes(
                &combined,
                &sorting_columns,
                self.row_converter
                    .as_ref()
                    .ok_or(SortingParquetError::WriterClosed)?,
            )?;
        drop(combined);

        // Write to a new run file
        let run_path = self
            .temp_dir
            .path()
            .join(format!("run_{}.parquet", self.run_count));
        self.run_count += 1;

        let run_file = File::create(&run_path)?;
        let mut run_writer = ArrowWriter::try_new(
            run_file,
            self.schema.clone(),
            Some(
                WriterProperties::builder()
                    .set_write_page_header_statistics(false)
                    .set_statistics_enabled(parquet::file::properties::EnabledStatistics::None)
                    .build(),
            ),
        )?;

        // Write as a single batch — ArrowWriter handles page sizing internally
        run_writer.write(&sorted)?;
        run_writer.close()?;

        self.run_files.push(RunInfo {
            path: run_path,
            min_sort_key: Rc::new(min_sort_key),
            max_sort_key: Rc::new(max_sort_key),
        });
        Ok(())
    }

    /// Finishes writing, performing the final merge of all sorted runs
    /// and producing the globally sorted output file.
    pub fn finish(&mut self) -> Result<(), SortingParquetError> {
        let sorting_columns = self
            .properties
            .sorting_columns()
            .ok_or(SortingParquetError::NoSortingColumnsConfigured)?
            .clone();

        // Flush any remaining buffered data to a run
        self.flush_to_run()?;

        let target = &mut self.target;
        let output_batch_size = self
            .properties
            .max_row_group_row_count()
            .unwrap_or(DEFAULT_MAX_MEMORY_ROWS);

        match self.run_files.len() {
            0 => {
                // No data written at all
                target.finish()?;
            }
            1 => {
                // Single run — already fully sorted, just copy through
                let file = File::open(&self.run_files[0].path)?;
                let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
                    .with_batch_size(output_batch_size)
                    .build()?;
                for batch in reader {
                    target.write(&batch?)?;
                    target.flush()?;
                }
                target.finish()?;
            }
            _ => {
                // Multiple runs — streaming k-way merge

                let merger = SortedRunMerger::try_new(
                    mem::take(&mut self.run_files),
                    sorting_columns,
                    self.row_converter
                        .take()
                        .expect("RowConverter should be set if we have sorting columns"),
                    output_batch_size,
                )?;

                for batch_result in merger {
                    target.write(&batch_result?)?;
                    target.flush()?;
                }
                target.finish()?;
            }
        }

        // temp_dir drops here, cleaning up all run files automatically
        Ok(())
    }
    pub fn inner_mut(&mut self) -> &mut W {
        self.target.inner_mut()
    }
}

#[cfg(test)]
mod tests {
    use crate::test::get_test_dir;

    use super::*;
    use arrow::array::{Int32Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::arrow_reader::{ArrowReaderBuilder, ArrowReaderOptions};
    use parquet::file::metadata::SortingColumn;
    use std::sync::Arc;

    fn create_test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]))
    }

    fn create_test_batch(ids: Vec<i32>, names: Vec<&str>) -> RecordBatch {
        let schema = create_test_schema();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_sorting_parquet_writer() {
        let schema = create_test_schema();
        let sorting_columns = vec![SortingColumn {
            column_idx: 0,
            descending: false,
            nulls_first: false,
        }];
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(2))
            .set_sorting_columns(Some(sorting_columns))
            .build();

        let test_file = File::create(get_test_dir().join("output.parquet")).unwrap();
        let mut writer =
            SortingParquetWriter::try_new(test_file, schema.clone(), properties).unwrap();

        let test_input = vec![
            (vec![3, 1], vec!["c", "a"]),
            (vec![4], vec!["d"]),
            (vec![2], vec!["b"]),
            (vec![5, 0], vec!["e", "z"]),
            (vec![6], vec!["f"]),
            (vec![8], vec!["h"]),
            (vec![7], vec!["g"]),
            (vec![9], vec!["i"]),
            (vec![10], vec!["j"]),
        ];
        for (ids, names) in test_input {
            let batch = create_test_batch(ids, names);
            writer.write(&batch).unwrap();
        }
        writer.finish().unwrap();

        let test_file = File::open(get_test_dir().join("output.parquet")).unwrap();
        let mut parquet_reader = ArrowReaderBuilder::try_new_with_options(
            test_file,
            ArrowReaderOptions::new().with_schema(schema.clone()),
        )
        .unwrap()
        .build()
        .unwrap();

        let batch = parquet_reader.next().unwrap().unwrap();
        let expected_ids: Vec<i32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let expected_names: Vec<&str> = vec!["z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j"];
        let actual_ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .iter()
            .flatten()
            .collect::<Vec<i32>>();
        let actual_names = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .flatten()
            .collect::<Vec<&str>>();
        assert_eq!(actual_ids, expected_ids, "IDs should be sorted");
        assert_eq!(
            actual_names, expected_names,
            "Names should be sorted according to ID order"
        );
    }

    #[test]
    fn test_sorting_writer_forced_spill() {
        // Use a very small memory limit to force multiple run files
        let schema = create_test_schema();
        let sorting_columns = vec![SortingColumn {
            column_idx: 0,
            descending: false,
            nulls_first: false,
        }];
        let properties = WriterProperties::builder()
            .set_sorting_columns(Some(sorting_columns))
            .build();

        let temp = tempfile::NamedTempFile::new().unwrap();
        let file = temp.reopen().unwrap();
        let mut writer =
            SortingParquetWriter::try_new_with_memory_limit(file, schema.clone(), properties, 3)
                .unwrap();

        // Write 10 rows across multiple batches, out of order
        writer
            .write(&create_test_batch(vec![9, 7, 5], vec!["i", "g", "e"]))
            .unwrap();
        writer
            .write(&create_test_batch(vec![3, 1], vec!["c", "a"]))
            .unwrap();
        writer
            .write(&create_test_batch(vec![8, 6, 4], vec!["h", "f", "d"]))
            .unwrap();
        writer
            .write(&create_test_batch(vec![2, 0], vec!["b", "z"]))
            .unwrap();
        writer.finish().unwrap();

        // Read back and verify global sort
        let file = temp.reopen().unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();

        let mut all_ids = Vec::new();
        let mut all_names = Vec::new();
        for batch in reader {
            let batch = batch.unwrap();
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let names = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for i in 0..batch.num_rows() {
                all_ids.push(ids.value(i));
                all_names.push(names.value(i).to_string());
            }
        }

        assert_eq!(all_ids, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            all_names,
            vec!["z", "a", "b", "c", "d", "e", "f", "g", "h", "i"]
        );
    }

    #[test]
    fn test_sorting_writer_single_run() {
        // All data fits in memory — single run, no merge needed
        let schema = create_test_schema();
        let sorting_columns = vec![SortingColumn {
            column_idx: 0,
            descending: false,
            nulls_first: false,
        }];
        let properties = WriterProperties::builder()
            .set_sorting_columns(Some(sorting_columns))
            .build();

        let temp = tempfile::NamedTempFile::new().unwrap();
        let file = temp.reopen().unwrap();
        let mut writer =
            SortingParquetWriter::try_new_with_memory_limit(file, schema.clone(), properties, 100)
                .unwrap();

        writer
            .write(&create_test_batch(vec![3, 1, 2], vec!["c", "a", "b"]))
            .unwrap();
        writer.finish().unwrap();

        let file = temp.reopen().unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();

        let mut all_ids = Vec::new();
        for batch in reader {
            let batch = batch.unwrap();
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                all_ids.push(ids.value(i));
            }
        }
        assert_eq!(all_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_multi_run_with_complex_types() {
        use crate::test::{TestArrowType, TickerItem};
        use parquet::arrow::arrow_reader::{ArrowReaderBuilder, ArrowReaderOptions};

        let temp = tempfile::NamedTempFile::new().unwrap();
        let file = temp.reopen().unwrap();
        let props = WriterProperties::builder()
            .set_sorting_columns(Some(TickerItem::sorting_columns()))
            .build();
        let schema = TickerItem::schema();
        let mut writer =
            SortingParquetWriter::try_new_with_memory_limit(file, schema.clone(), props, 100_000)
                .unwrap();

        for _ in 0..3 {
            let items = TickerItem::random_instances(100_000);
            for chunk in items.chunks(128) {
                let batch = TickerItem::into_record_batch(chunk).unwrap();
                writer.write(&batch).unwrap();
            }
        }
        writer.finish().unwrap();

        let file = temp.reopen().unwrap();
        let reader = ArrowReaderBuilder::try_new_with_options(
            file,
            ArrowReaderOptions::new().with_schema(TickerItem::schema()),
        )
        .unwrap()
        .with_batch_size(200)
        .build()
        .unwrap();
        let mut total_rows = 0;
        for batch in reader {
            let batch = batch.unwrap();
            let items = TickerItem::from_record_batch(&batch).unwrap();
            assert_eq!(TickerItem::is_sorted(&items), None);
            total_rows += batch.num_rows();
        }
        assert_eq!(total_rows, 300_000);
    }
}
