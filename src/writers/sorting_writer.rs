use std::fs::File;
use std::io::Write;
use std::mem;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use crate::SortingParquetError;
use crate::record_batch::streaming_merge::{RunInfo, SortedRunMerger};
use crate::sorting::SortExtremes;
use crate::writers::progress::{
    FinishPhase, FinishProgress, FinishProgressHandler, NoopProgressHandler,
};

/// Default maximum number of rows to buffer in memory before flushing to a sorted run file.
const DEFAULT_MAX_MEMORY_ROWS: usize = 1_000_000;

/// Controls when the in-memory buffer is flushed to a sorted run file on disk.
///
/// # Example
///
/// ```rust
/// use sorting_parquet_writer::writers::FlushThreshold;
///
/// // Flush after 500k rows
/// let by_rows = FlushThreshold::Rows(500_000);
///
/// // Flush after ~256 MB of buffered data
/// let by_bytes = FlushThreshold::Bytes(256 * 1024 * 1024);
///
/// // Flush when either limit is reached (whichever comes first)
/// let either = FlushThreshold::Either {
///     max_rows: 500_000,
///     max_bytes: 256 * 1024 * 1024,
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushThreshold {
    /// Flush when the buffered row count reaches this limit.
    Rows(usize),
    /// Flush when the estimated in-memory size of buffered data reaches this
    /// many bytes. The size is estimated using Arrow's `get_array_memory_size()`.
    Bytes(usize),
    /// Flush when *either* the row count or byte size limit is reached,
    /// whichever comes first. This is useful for bounding memory usage when
    /// row sizes vary.
    Either { max_rows: usize, max_bytes: usize },
}

impl FlushThreshold {
    /// Returns `true` if the current buffer state exceeds this threshold.
    fn should_flush(&self, buffered_rows: usize, buffered_bytes: usize) -> bool {
        match self {
            FlushThreshold::Rows(max) => buffered_rows >= *max,
            FlushThreshold::Bytes(max) => buffered_bytes >= *max,
            FlushThreshold::Either {
                max_rows,
                max_bytes,
            } => buffered_rows >= *max_rows || buffered_bytes >= *max_bytes,
        }
    }
}

/// Configuration options for the sorting writer's external merge sort behavior.
///
/// These options control how data is buffered, sorted, and spilled to disk
/// during the write phase. They are separate from [`WriterProperties`], which
/// controls Parquet encoding and compression for the final output file.
///
/// # Example
///
/// ```rust,no_run
/// use sorting_parquet_writer::writers::{SortingWriterOptions, FlushThreshold};
/// use std::path::PathBuf;
///
/// let options = SortingWriterOptions {
///     flush_threshold: FlushThreshold::Either {
///         max_rows: 500_000,
///         max_bytes: 256 * 1024 * 1024,
///     },
///     temp_dir: Some(PathBuf::from("/fast-ssd/tmp")),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct SortingWriterOptions {
    /// Controls when buffered data is flushed to a sorted run file.
    ///
    /// Default: `FlushThreshold::Rows(1_000_000)`
    pub flush_threshold: FlushThreshold,

    /// Directory for temporary sorted run files. If `None`, the system's
    /// default temp directory is used.
    ///
    /// Run files are automatically cleaned up when the writer is finished or dropped.
    ///
    /// Default: `None` (system temp directory)
    pub temp_dir: Option<PathBuf>,

    /// Writer properties for temporary run files. Controls compression and
    /// encoding of intermediate sorted data on disk.
    ///
    /// By default, run files disable statistics for faster writes since they
    /// are only read during the merge phase and immediately deleted.
    ///
    /// Tip: Use fast compression (e.g., LZ4) for run files even if the final
    /// output uses stronger compression like ZSTD.
    ///
    /// Default: statistics disabled, default compression
    pub run_file_properties: Option<WriterProperties>,

    /// When `true`, each incoming [`RecordBatch`] is sorted individually on
    /// [`write()`](SortingParquetWriter::write) and the flush phase merges the
    /// pre-sorted batches with a streaming k-way merge instead of concatenating
    /// and re-sorting them from scratch.
    ///
    /// This trades a small amount of extra work per `write()` call for a
    /// significantly cheaper flush, because merging already-sorted runs is
    /// *O(n)* rather than *O(n log n)*.
    ///
    /// Enable this when individual batches arrive unsorted and the flush
    /// buffer typically contains many batches. If each batch is already sorted
    /// (e.g. coming from a sorted source), only the merge benefit applies.
    ///
    /// Default: `false`
    pub merge_sort_batches: bool,
}

impl Default for SortingWriterOptions {
    fn default() -> Self {
        Self {
            flush_threshold: FlushThreshold::Rows(DEFAULT_MAX_MEMORY_ROWS),
            temp_dir: None,
            run_file_properties: None,
            merge_sort_batches: false,
        }
    }
}

/// A Parquet writer that produces a globally sorted output file.
///
/// Uses an external merge sort strategy:
/// 1. **Write phase:** Buffers incoming [`RecordBatch`]es in memory until the
///    configured [`FlushThreshold`] is reached. When the limit is reached,
///    the buffer is sorted and written to a temporary "run" file on disk.
/// 2. **Merge phase (at [`finish()`](Self::finish)):** All sorted run files are merged
///    via a streaming k-way merge, producing the final globally sorted output.
///
/// Memory usage is bounded by `max_memory_rows` during the write phase, and by
/// approximately one batch per run file during the merge phase.
///
/// # Example
///
/// ```rust,no_run
/// use sorting_parquet_writer::writers::SortingParquetWriter;
/// use parquet::file::properties::WriterProperties;
/// use parquet::file::metadata::SortingColumn;
/// use arrow::datatypes::{Schema, Field, DataType, SchemaRef};
/// use std::sync::Arc;
///
/// let schema: SchemaRef = Arc::new(Schema::new(vec![
///     Field::new("id", DataType::Int32, false),
/// ]));
/// let props = WriterProperties::builder()
///     .set_sorting_columns(Some(vec![SortingColumn {
///         column_idx: 0, descending: false, nulls_first: false,
///     }]))
///     .build();
///
/// let file = std::fs::File::create("output.parquet").unwrap();
/// let mut writer = SortingParquetWriter::try_new(file, schema, props).unwrap();
/// // writer.write(&batch).unwrap();
/// // let file = writer.finish().unwrap();
/// ```
pub struct SortingParquetWriter<W: Write + Send> {
    schema: SchemaRef,
    properties: WriterProperties,
    target: ArrowWriter<W>,
    row_converter: Option<arrow_row::RowConverter>,
    buffer: Vec<RecordBatch>,
    buffered_rows: usize,
    buffered_bytes: usize,
    options: SortingWriterOptions,
    temp_dir: TempDir,
    run_files: Vec<RunInfo>,
    run_count: usize,
}

impl<W: Write + Send> SortingParquetWriter<W> {
    /// Creates a new `SortingParquetWriter` with default sorting options.
    ///
    /// Uses a 1M row memory buffer and the system's default temp directory.
    /// The `properties` must have sorting columns configured via
    /// [`WriterPropertiesBuilder::set_sorting_columns`](parquet::file::properties::WriterPropertiesBuilder::set_sorting_columns).
    ///
    /// # Errors
    ///
    /// Returns [`SortingParquetError::NoSortingColumnsConfigured`] if
    /// `properties` does not have sorting columns set.
    pub fn try_new(
        writer: W,
        schema: SchemaRef,
        properties: WriterProperties,
    ) -> Result<Self, SortingParquetError> {
        Self::try_new_with_options(writer, schema, properties, SortingWriterOptions::default())
    }

    /// Creates a new `SortingParquetWriter` with custom sorting options.
    ///
    /// See [`SortingWriterOptions`] for configurable parameters including
    /// memory limits, temp directory, and run file compression.
    ///
    /// # Errors
    ///
    /// Returns [`SortingParquetError::NoSortingColumnsConfigured`] if
    /// `properties` does not have sorting columns set.
    pub fn try_new_with_options(
        writer: W,
        schema: SchemaRef,
        properties: WriterProperties,
        options: SortingWriterOptions,
    ) -> Result<Self, SortingParquetError> {
        if properties.sorting_columns().is_none() {
            return Err(SortingParquetError::NoSortingColumnsConfigured);
        }
        let target = ArrowWriter::try_new(writer, schema.clone(), Some(properties.clone()))?;
        let temp_dir = match &options.temp_dir {
            Some(dir) => TempDir::with_prefix_in("sorting_parquet_writer", dir)?,
            None => TempDir::with_prefix("sorting_parquet_writer")?,
        };
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
            buffered_bytes: 0,
            options,
            temp_dir,
            run_files: Vec::new(),
            run_count: 0,
        })
    }

    // ── Writing ─────────────────────────────────────────────────────────

    /// Writes a [`RecordBatch`] to the writer.
    ///
    /// Data is buffered in memory and periodically sorted and flushed to
    /// temporary run files on disk when the configured [`FlushThreshold`] is reached.
    /// The batch schema must match the schema provided at construction.
    pub fn write(&mut self, batch: &RecordBatch) -> Result<(), SortingParquetError> {
        self.buffered_rows += batch.num_rows();
        self.buffered_bytes += batch.get_array_memory_size();
        if self.options.merge_sort_batches {
            let sorting_columns = self
                .properties
                .sorting_columns()
                .ok_or(SortingParquetError::NoSortingColumnsConfigured)?
                .clone();
            let sorted_batch = crate::sorting::sort_record_batch_with_row_converter(
                batch,
                &sorting_columns,
                self.row_converter
                    .as_ref()
                    .ok_or(SortingParquetError::WriterClosed)?,
            )?;
            self.buffer.push(sorted_batch);
        } else {
            self.buffer.push(batch.clone());
        }

        if self
            .options
            .flush_threshold
            .should_flush(self.buffered_rows, self.buffered_bytes)
        {
            self.flush_to_run()?;
        }
        Ok(())
    }

    /// Manually flushes the in-memory buffer to a new sorted run file on disk.
    ///
    /// This can be used to control memory usage externally (e.g., based on
    /// system memory pressure) regardless of the configured [`FlushThreshold`].
    ///
    /// This is a no-op if the buffer is empty.
    pub fn flush_buffer(&mut self) -> Result<(), SortingParquetError> {
        self.flush_to_run()
    }

    /// Appends a key-value metadata entry to the Parquet file footer.
    ///
    /// This metadata is written when [`finish()`](Self::finish) is called.
    pub fn append_key_value_metadata(&mut self, kv_metadata: parquet::file::metadata::KeyValue) {
        self.target.append_key_value_metadata(kv_metadata);
    }

    // ── Finalization ────────────────────────────────────────────────────

    /// Finishes writing, performing the final merge of all sorted runs
    /// and producing the globally sorted output file.
    ///
    /// This consumes the writer and returns the underlying `W`. All temporary
    /// run files are cleaned up automatically.
    ///
    /// This is the only way to produce a valid Parquet file — dropping the
    /// writer without calling `finish()` will not write the Parquet footer.
    pub fn finish(self) -> Result<W, SortingParquetError> {
        self.finish_with_progress(NoopProgressHandler)
    }

    /// Like [`finish()`](Self::finish), but calls `handler` after each batch
    /// is written to the final output during the merge phase.
    ///
    /// The handler receives a [`FinishProgress`] with rows written, total rows,
    /// batch count, and the current phase. Use [`FinishProgress::fraction_complete()`]
    /// for a `[0.0, 1.0]` progress fraction.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use sorting_parquet_writer::writers::{SortingParquetWriter, FinishProgress};
    /// # fn example(writer: SortingParquetWriter<std::fs::File>) {
    /// writer.finish_with_progress(|p: &FinishProgress| {
    ///     println!("Merge progress: {:.1}%", p.fraction_complete() * 100.0);
    /// }).unwrap();
    /// # }
    /// ```
    pub fn finish_with_progress(
        mut self,
        mut handler: impl FinishProgressHandler,
    ) -> Result<W, SortingParquetError> {
        let sorting_columns = self
            .properties
            .sorting_columns()
            .ok_or(SortingParquetError::NoSortingColumnsConfigured)?
            .clone();

        // Flush any remaining buffered data to a run
        self.flush_to_run()?;

        let output_batch_size = self
            .properties
            .max_row_group_row_count()
            .unwrap_or(DEFAULT_MAX_MEMORY_ROWS);

        let num_runs = self.run_files.len();

        match num_runs {
            0 => {
                // No data written at all
            }
            1 => {
                // Single run — already fully sorted, just copy through
                let file = File::open(&self.run_files[0].path)?;
                let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                let total_rows = builder.metadata().file_metadata().num_rows() as u64;
                let reader = builder.with_batch_size(output_batch_size).build()?;

                let mut progress = FinishProgress {
                    phase: FinishPhase::CopyThrough,
                    rows_written: 0,
                    batches_written: 0,
                    total_rows,
                    num_runs: 1,
                };

                for batch in reader {
                    let batch = batch?;
                    self.target.write(&batch)?;
                    self.target.flush()?;
                    progress.rows_written += batch.num_rows() as u64;
                    progress.batches_written += 1;
                    handler.on_batch_written(&progress);
                }
            }
            _ => {
                // Read total row count from all run file metadata
                let total_rows = self.read_total_rows()?;

                let mut progress = FinishProgress {
                    phase: FinishPhase::Merging,
                    rows_written: 0,
                    batches_written: 0,
                    total_rows,
                    num_runs,
                };

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
                    let batch = batch_result?;
                    self.target.write(&batch)?;
                    self.target.flush()?;
                    progress.rows_written += batch.num_rows() as u64;
                    progress.batches_written += 1;
                    handler.on_batch_written(&progress);
                }
            }
        }

        // into_inner calls finish() internally, writing the Parquet footer
        let writer = self.target.into_inner()?;
        // temp_dir drops here, cleaning up all run files automatically
        Ok(writer)
    }

    // ── Introspection ───────────────────────────────────────────────────

    /// Returns the number of rows currently buffered in memory, waiting to
    /// be sorted and flushed to a run file.
    pub fn in_progress_rows(&self) -> usize {
        self.buffered_rows
    }

    /// Returns the estimated byte size of data currently buffered in memory,
    /// waiting to be sorted and flushed to a run file.
    pub fn in_progress_bytes(&self) -> usize {
        self.buffered_bytes
    }

    /// Returns the number of sorted run files that have been flushed to disk.
    ///
    /// Each run file contains up to `max_memory_rows` sorted rows. During
    /// [`finish()`](Self::finish), all run files are merged into the final output.
    pub fn num_run_files(&self) -> usize {
        self.run_files.len()
    }

    /// Returns the total number of bytes written to the target writer so far.
    ///
    /// Note: during the write phase this is always 0 because data is buffered
    /// to temporary run files. Bytes are written to the target only during
    /// [`finish()`](Self::finish).
    pub fn bytes_written(&self) -> usize {
        self.target.bytes_written()
    }

    // ── Access ──────────────────────────────────────────────────────────

    /// Returns a reference to the Arrow schema used by this writer.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Returns a reference to the Parquet writer properties.
    pub fn writer_properties(&self) -> &WriterProperties {
        &self.properties
    }

    /// Returns a reference to the sorting writer options.
    pub fn sorting_options(&self) -> &SortingWriterOptions {
        &self.options
    }

    /// Returns an immutable reference to the underlying writer.
    pub fn inner(&self) -> &W {
        self.target.inner()
    }

    /// Returns a mutable reference to the underlying writer.
    pub fn inner_mut(&mut self) -> &mut W {
        self.target.inner_mut()
    }

    // ── Internal ────────────────────────────────────────────────────────

    /// Sum the row counts from all run file Parquet metadata.
    fn read_total_rows(&self) -> Result<u64, SortingParquetError> {
        let mut total = 0u64;
        for run in &self.run_files {
            let file = File::open(&run.path)?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
            total += builder.metadata().file_metadata().num_rows() as u64;
        }
        Ok(total)
    }
    fn flush_to_run_merge_sort(
        &mut self,
    ) -> Result<(RecordBatch, SortExtremes), SortingParquetError> {
        let sorting_columns = self
            .properties
            .sorting_columns()
            .ok_or(SortingParquetError::NoSortingColumnsConfigured)?
            .clone();
        let (record, (min_sort_key, max_sort_key)) =
            crate::record_batch::merge_sorted_batches_with_row_converter_returning_extremes(
                &self.buffer,
                &sorting_columns,
                self.row_converter
                    .as_ref()
                    .ok_or(SortingParquetError::WriterClosed)?,
            )?;
        self.buffer.clear();
        self.buffered_rows = 0;
        self.buffered_bytes = 0;

        Ok((record, (min_sort_key, max_sort_key)))
    }

    fn flush_to_run_concat_and_sort(
        &mut self,
    ) -> Result<(RecordBatch, SortExtremes), SortingParquetError> {
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
        self.buffered_bytes = 0;
        let (sorted, (min_sort_key, max_sort_key)) =
            crate::sorting::sort_record_batch_with_row_converter_returning_extremes(
                &combined,
                &sorting_columns,
                self.row_converter
                    .as_ref()
                    .ok_or(SortingParquetError::WriterClosed)?,
            )?;
        Ok((sorted, (min_sort_key, max_sort_key)))
    }
    /// Sort the in-memory buffer and write it to a new run file.
    fn flush_to_run(&mut self) -> Result<(), SortingParquetError> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        // Sort the combined batch and extract min/max sort keys in one pass
        let (sorted, (min_sort_key, max_sort_key)) = if self.options.merge_sort_batches {
            self.flush_to_run_merge_sort()?
        } else {
            self.flush_to_run_concat_and_sort()?
        };

        // Write to a new run file
        let run_path = self
            .temp_dir
            .path()
            .join(format!("run_{}.parquet", self.run_count));
        self.run_count += 1;

        let run_file_props = self.options.run_file_properties.clone().unwrap_or_else(|| {
            WriterProperties::builder()
                .set_write_page_header_statistics(false)
                .set_statistics_enabled(parquet::file::properties::EnabledStatistics::None)
                .build()
        });

        let run_file = File::create(&run_path)?;
        let mut run_writer =
            ArrowWriter::try_new(run_file, self.schema.clone(), Some(run_file_props))?;

        run_writer.write(&sorted)?;
        run_writer.close()?;

        self.run_files.push(RunInfo {
            path: run_path,
            min_sort_key: Arc::new(min_sort_key),
            max_sort_key: Arc::new(max_sort_key),
        });
        Ok(())
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
        let options = SortingWriterOptions {
            flush_threshold: FlushThreshold::Rows(3),
            ..Default::default()
        };
        let mut writer =
            SortingParquetWriter::try_new_with_options(file, schema.clone(), properties, options)
                .unwrap();

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
        let options = SortingWriterOptions {
            flush_threshold: FlushThreshold::Rows(100),
            ..Default::default()
        };
        let mut writer =
            SortingParquetWriter::try_new_with_options(file, schema.clone(), properties, options)
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
        let options = SortingWriterOptions {
            flush_threshold: FlushThreshold::Rows(100_000),
            ..Default::default()
        };
        let mut writer =
            SortingParquetWriter::try_new_with_options(file, schema.clone(), props, options)
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

    #[test]
    fn test_flush_threshold_bytes() {
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
        // Use a very small byte threshold to force spills
        let options = SortingWriterOptions {
            flush_threshold: FlushThreshold::Bytes(1),
            ..Default::default()
        };
        let mut writer =
            SortingParquetWriter::try_new_with_options(file, schema.clone(), properties, options)
                .unwrap();

        writer
            .write(&create_test_batch(vec![3, 1], vec!["c", "a"]))
            .unwrap();
        assert!(
            writer.num_run_files() > 0,
            "Should have spilled to run file"
        );
        assert_eq!(writer.in_progress_rows(), 0);
        assert_eq!(writer.in_progress_bytes(), 0);

        writer
            .write(&create_test_batch(vec![2, 0], vec!["b", "z"]))
            .unwrap();
        writer.finish().unwrap();

        // Verify output is sorted
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
        assert_eq!(all_ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_flush_threshold_either() {
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
        // Row limit is very high, but byte limit is tiny — bytes should trigger
        let options = SortingWriterOptions {
            flush_threshold: FlushThreshold::Either {
                max_rows: usize::MAX,
                max_bytes: 1,
            },
            ..Default::default()
        };
        let mut writer =
            SortingParquetWriter::try_new_with_options(file, schema.clone(), properties, options)
                .unwrap();

        writer
            .write(&create_test_batch(vec![3, 1, 2], vec!["c", "a", "b"]))
            .unwrap();
        assert!(
            writer.num_run_files() > 0,
            "Bytes threshold should have triggered"
        );

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
    fn test_flush_buffer_manual() {
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
        let mut writer = SortingParquetWriter::try_new(file, schema.clone(), properties).unwrap();

        writer
            .write(&create_test_batch(vec![3, 1], vec!["c", "a"]))
            .unwrap();
        assert_eq!(writer.num_run_files(), 0);
        assert!(writer.in_progress_rows() > 0);
        assert!(writer.in_progress_bytes() > 0);

        writer.flush_buffer().unwrap();
        assert_eq!(writer.num_run_files(), 1);
        assert_eq!(writer.in_progress_rows(), 0);
        assert_eq!(writer.in_progress_bytes(), 0);

        // Flush on empty buffer is a no-op
        writer.flush_buffer().unwrap();
        assert_eq!(writer.num_run_files(), 1);

        writer
            .write(&create_test_batch(vec![2, 0], vec!["b", "z"]))
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
        assert_eq!(all_ids, vec![0, 1, 2, 3]);
    }
}
