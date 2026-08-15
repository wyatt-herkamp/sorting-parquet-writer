use std::fs::File;
use std::io::Write;
use std::mem;
use std::num::NonZeroUsize;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use arrow_row::RowConverter;
use parking_lot::Mutex;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::metadata::SortingColumn;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use tempfile::TempDir;
use tokio::task::JoinSet;

use crate::SortingParquetError;
use crate::record_batch::streaming_merge::{RunInfo, SortedRunMerger};
use crate::sorting::SortExtremes;
use crate::writers::progress::{
    FinishPhase, FinishProgress, FinishProgressHandler, NoopProgressHandler,
};
use crate::writers::{DEFAULT_MAX_MEMORY_ROWS, SortingWriterOptions};

/// Default number of spill tasks allowed to run concurrently in the background
/// (see [`BackgroundWorker::max_concurrent_workers`]).
const DEFAULT_MAX_CONCURRENT_WORKERS: usize = 4;

/// Controls the background spill task pool used by
/// [`ConcurrentSortingParquetWriter`].
///
/// Each time the in-memory buffer is flushed, the sort + run-file write is
/// offloaded to a [`tokio::task::spawn_blocking`] task so the caller can keep
/// buffering the next batch. This struct bounds how many of those tasks may be
/// in flight at once.
#[derive(Debug, Clone)]
pub struct BackgroundWorker {
    /// Maximum number of spill tasks allowed to run concurrently.
    ///
    /// When this many tasks are already in flight, [`write`] / [`flush_buffer`]
    /// will await the completion of an earlier spill before scheduling a new
    /// one. This bounds peak memory: each in-flight task owns a buffer's worth
    /// of rows, so unbounded concurrency would defeat the bounded-memory
    /// guarantee of the write phase.
    ///
    /// `None` means no bound (every flush is scheduled immediately). The
    /// [`Default`] is `Some(4)`; raise it to trade memory for throughput, or
    /// set `None` only if you bound memory by other means.
    ///
    /// [`write`]: ConcurrentSortingParquetWriter::write
    /// [`flush_buffer`]: ConcurrentSortingParquetWriter::flush_buffer
    pub max_concurrent_workers: Option<NonZeroUsize>,
}

impl Default for BackgroundWorker {
    fn default() -> Self {
        Self {
            max_concurrent_workers: NonZeroUsize::new(DEFAULT_MAX_CONCURRENT_WORKERS),
        }
    }
}

/// An async Parquet writer that produces a globally sorted output file, spilling
/// sorted runs to disk concurrently in the background.
///
/// This is the asynchronous counterpart to
/// [`SortingParquetWriter`](crate::writers::SortingParquetWriter): it uses the
/// same external merge sort strategy, but each spill (sort + run-file write) is
/// offloaded to a background [`tokio::task::spawn_blocking`] task. This lets the
/// caller keep buffering the next batch while previous runs are still being
/// sorted and written. It therefore requires a Tokio runtime.
///
/// 1. **Write phase:** [`write`](Self::write) buffers incoming [`RecordBatch`]es
///    in memory until the configured
///    [`FlushThreshold`](crate::writers::FlushThreshold) is reached. The buffer
///    is then handed to a background task that sorts it and writes it to a
///    temporary "run" file on disk. `write` returns as soon as the task is
///    *scheduled* — the run file may still be in flight. The number of
///    concurrent spill tasks is bounded by [`BackgroundWorker`].
/// 2. **Merge phase (at [`finish()`](Self::finish)):** all outstanding spill
///    tasks are awaited, then the sorted run files are merged via a streaming
///    k-way merge, producing the final globally sorted output.
///
/// Peak memory is bounded by the flush threshold times the number of in-flight
/// spill tasks during the write phase, and by approximately one batch per run
/// file during the merge phase.
///
/// # Errors from background tasks
///
/// A spill that fails (e.g. a sort error or a full disk) surfaces the first time
/// its task is awaited — during back-pressure in a later [`write`](Self::write),
/// during [`wait_for_background_tasks`](Self::wait_for_background_tasks), or
/// during [`finish`](Self::finish). It is never silently dropped.
///
/// # Example
///
/// ```rust,no_run
/// use sorting_parquet_writer::writers::ConcurrentSortingParquetWriter;
/// use parquet::file::properties::WriterProperties;
/// use parquet::file::metadata::SortingColumn;
/// use arrow::datatypes::{Schema, Field, DataType, SchemaRef};
/// use std::sync::Arc;
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let schema: SchemaRef = Arc::new(Schema::new(vec![
///     Field::new("id", DataType::Int32, false),
/// ]));
/// let props = WriterProperties::builder()
///     .set_sorting_columns(Some(vec![SortingColumn {
///         column_idx: 0, descending: false, nulls_first: false,
///     }]))
///     .build();
///
/// let file = std::fs::File::create("output.parquet")?;
/// let mut writer = ConcurrentSortingParquetWriter::try_new(file, schema, props)?;
/// // writer.write(&batch).await?;
/// let file = writer.finish().await?;
/// # Ok(())
/// # }
/// ```
pub struct ConcurrentSortingParquetWriter<W: Write + Send> {
    schema: SchemaRef,
    properties: WriterProperties,
    target: ArrowWriter<W>,
    row_converter: Arc<arrow_row::RowConverter>,
    buffer: Vec<RecordBatch>,
    buffered_rows: usize,
    buffered_bytes: usize,
    options: SortingWriterOptions,
    temp_dir: TempDir,
    run_files: Arc<Mutex<Vec<RunInfo>>>,
    run_count: usize,
    background_options: BackgroundWorker,
    join_set: JoinSet<Result<(), SortingParquetError>>,
}

impl<W: Write + Send> ConcurrentSortingParquetWriter<W> {
    /// Creates a new `ConcurrentSortingParquetWriter` with default sorting
    /// options and a default [`BackgroundWorker`].
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
        Self::try_new_with_options(
            writer,
            schema,
            properties,
            SortingWriterOptions::default(),
            BackgroundWorker::default(),
        )
    }

    /// Creates a new `ConcurrentSortingParquetWriter` with custom sorting
    /// options and background spill configuration.
    ///
    /// See [`SortingWriterOptions`] for configurable parameters including
    /// memory limits, temp directory, and run file compression, and
    /// [`BackgroundWorker`] for bounding background spill concurrency.
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
        background_options: BackgroundWorker,
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
            row_converter: Arc::new(row_converter),
            buffer: Vec::new(),
            buffered_rows: 0,
            buffered_bytes: 0,
            options,
            temp_dir,
            run_files: Arc::new(Mutex::new(Vec::new())),
            run_count: 0,
            background_options,
            join_set: JoinSet::new(),
        })
    }

    // ── Writing ─────────────────────────────────────────────────────────

    /// Writes a [`RecordBatch`] to the writer.
    ///
    /// Data is buffered in memory. When the configured
    /// [`FlushThreshold`](crate::writers::FlushThreshold) is reached, the buffer
    /// is handed to a background task that sorts it and writes a temporary run
    /// file; this call returns once that task is *scheduled*, not once the file
    /// is written. If the background spill pool is already at
    /// [`BackgroundWorker::max_concurrent_workers`], this awaits an earlier
    /// spill before scheduling the new one (and surfaces any error it produced).
    ///
    /// The batch schema must match the schema provided at construction.
    pub async fn write(&mut self, batch: &RecordBatch) -> Result<(), SortingParquetError> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        if batch.schema_ref() != &self.schema {
            return Err(SortingParquetError::ArrowError(
                arrow::error::ArrowError::SchemaError(
                    "Batch schema does not match writer schema".to_string(),
                ),
            ));
        }
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
                self.row_converter.as_ref(),
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
            self.flush_to_run().await?;
        }
        Ok(())
    }

    /// Awaits all outstanding background spill tasks, returning the first error
    /// any of them produced (or a task panic, surfaced as an
    /// [`SortingParquetError::IoError`]).
    ///
    /// This is normally not needed — [`finish`](Self::finish) drains the pool
    /// itself — but it lets callers force completion early, e.g. to surface a
    /// spill failure before continuing, or to release memory held by in-flight
    /// buffers.
    pub async fn wait_for_background_tasks(&mut self) -> Result<(), SortingParquetError> {
        while let Some(res) = self.join_set.join_next().await {
            Self::handle_join_result(res)?;
        }
        Ok(())
    }
    /// Manually flushes the in-memory buffer to a new sorted run file on disk.
    ///
    /// This can be used to control memory usage externally (e.g., based on
    /// system memory pressure) regardless of the configured
    /// [`FlushThreshold`](crate::writers::FlushThreshold). The spill runs in the
    /// background like any other; this returns once it is scheduled. A no-op if
    /// the buffer is empty (in particular, calling it twice in a row produces
    /// only one run file).
    pub async fn flush_buffer(&mut self) -> Result<(), SortingParquetError> {
        self.flush_to_run().await
    }

    /// Appends a key-value metadata entry to the Parquet file footer.
    ///
    /// This metadata is written when [`finish()`](Self::finish) is called.
    pub fn append_key_value_metadata(&mut self, kv_metadata: parquet::file::metadata::KeyValue) {
        self.target.append_key_value_metadata(kv_metadata);
    }

    // ── Finalization ────────────────────────────────────────────────────

    /// Finishes writing: awaits all outstanding background spill tasks, performs
    /// the final merge of all sorted runs, and produces the globally sorted
    /// output file.
    ///
    /// This consumes the writer and returns the underlying `W`. All temporary
    /// run files are cleaned up automatically. Any error from a background spill
    /// task is surfaced here.
    ///
    /// This is the only way to produce a valid Parquet file — dropping the
    /// writer without calling `finish()` will not write the Parquet footer.
    pub async fn finish(self) -> Result<W, SortingParquetError> {
        self.finish_with_progress(NoopProgressHandler).await
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
    /// # use sorting_parquet_writer::writers::{ConcurrentSortingParquetWriter, FinishProgress};
    /// # async fn example(writer: ConcurrentSortingParquetWriter<std::fs::File>) -> Result<(), Box<dyn std::error::Error>> {
    /// writer.finish_with_progress(|p: &FinishProgress| {
    ///     println!("Merge progress: {:.1}%", p.fraction_complete() * 100.0);
    /// }).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn finish_with_progress(
        mut self,
        mut handler: impl FinishProgressHandler,
    ) -> Result<W, SortingParquetError> {
        let sorting_columns = self
            .properties
            .sorting_columns()
            .ok_or(SortingParquetError::NoSortingColumnsConfigured)?
            .clone();

        // Flush any remaining buffered data to a run
        self.flush_to_run().await?;

        // Await every outstanding spill, surfacing the first error (or panic).
        self.wait_for_background_tasks().await?;

        let output_batch_size = self
            .properties
            .max_row_group_row_count()
            .unwrap_or(DEFAULT_MAX_MEMORY_ROWS);
        let num_runs = { self.run_files.lock().len() };

        match num_runs {
            0 => {
                // No data written at all
            }
            1 => {
                // Single run — already fully sorted, just copy through
                let run_files = self.run_files.lock();
                let file = File::open(&run_files[0].path)?;
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
                let run_files = { mem::take(&mut *self.run_files.lock()) };
                // Multiple runs — streaming k-way merge
                let merger = SortedRunMerger::try_new(
                    run_files,
                    sorting_columns,
                    self.row_converter.clone().into(),
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

    /// Returns the number of sorted runs that have been *scheduled* for spilling.
    ///
    /// Because spills run in the background, a counted run may still be sorting
    /// or writing its file when this returns; it reflects runs handed to the
    /// spill pool, not runs guaranteed to be on disk. Each run holds the rows
    /// buffered at the moment of its flush (bounded by the configured
    /// [`FlushThreshold`](crate::writers::FlushThreshold)). During
    /// [`finish()`](Self::finish), all runs are awaited and merged into the
    /// final output.
    pub fn num_run_files(&self) -> usize {
        self.run_count
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
    /// Takes ownership of the buffered batches, resetting the in-memory
    /// accounting so [`in_progress_rows`](Self::in_progress_rows) /
    /// [`in_progress_bytes`](Self::in_progress_bytes) reflect that the data has
    /// left the buffer (it is now owned by a background spill task).
    fn take_buffer(&mut self) -> Vec<RecordBatch> {
        self.buffered_rows = 0;
        self.buffered_bytes = 0;
        mem::take(&mut self.buffer)
    }

    /// Translates the result of awaiting a single spill task into a writer
    /// error: a task failure propagates its [`SortingParquetError`], and a task
    /// panic is surfaced as an [`SortingParquetError::IoError`].
    fn handle_join_result(
        res: Result<Result<(), SortingParquetError>, tokio::task::JoinError>,
    ) -> Result<(), SortingParquetError> {
        match res {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(join_err) => Err(SortingParquetError::IoError(std::io::Error::other(
                format!("Background spill task panicked: {join_err}"),
            ))),
        }
    }
    // ── Internal ────────────────────────────────────────────────────────

    /// Sum the row counts from all run-file Parquet metadata.
    ///
    /// Called once before the multi-run merge so the [`FinishProgress`]
    /// reported via [`finish_with_progress`](Self::finish_with_progress) has
    /// a fixed denominator (`total_rows`). Reads only the footers, not the
    /// row data.
    fn read_total_rows(&self) -> Result<u64, SortingParquetError> {
        let mut total = 0u64;
        let run_files = self.run_files.lock();
        for run in &*run_files {
            let file = File::open(&run.path)?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
            total += builder.metadata().file_metadata().num_rows() as u64;
        }
        Ok(total)
    }
    fn flush_to_run_merge_sort(
        buffer: Vec<RecordBatch>,
        sorting_columns: Vec<SortingColumn>,
        row_converter: Arc<RowConverter>,
    ) -> Result<(RecordBatch, SortExtremes), SortingParquetError> {
        let (record, (min_sort_key, max_sort_key)) =
            crate::record_batch::merge_sorted_batches_with_row_converter_returning_extremes(
                &buffer,
                &sorting_columns,
                row_converter.as_ref(),
            )?;

        Ok((record, (min_sort_key, max_sort_key)))
    }

    fn flush_to_run_concat_and_sort(
        buffer: Vec<RecordBatch>,
        sorting_columns: Vec<SortingColumn>,
        row_converter: Arc<RowConverter>,
        schema: SchemaRef,
    ) -> Result<(RecordBatch, SortExtremes), SortingParquetError> {
        // Concatenate all buffered batches, then drop the originals to free memory
        // before the sort creates another copy.
        let combined = arrow::compute::concat_batches(&schema, &buffer)?;

        let (sorted, (min_sort_key, max_sort_key)) =
            crate::sorting::sort_record_batch_with_row_converter_returning_extremes(
                &combined,
                &sorting_columns,
                row_converter.as_ref(),
            )?;
        Ok((sorted, (min_sort_key, max_sort_key)))
    }
    /// Sort the in-memory buffer and write it to a new run file.
    ///
    /// Picks between the two flush strategies based on
    /// [`SortingWriterOptions::merge_sort_batches`] and tags the resulting
    /// run with its min/max sort keys so
    /// [`SortedRunMerger`](crate::record_batch::streaming_merge::SortedRunMerger)
    /// can lazily activate runs by range during the final merge.
    async fn flush_to_run(&mut self) -> Result<(), SortingParquetError> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        if let Some(limit) = self.background_options.max_concurrent_workers {
            while self.join_set.len() >= limit.get() {
                if let Some(res) = self.join_set.join_next().await {
                    Self::handle_join_result(res)?;
                }
            }
        }
        let run_files = Arc::clone(&self.run_files);
        let run_path = self
            .temp_dir
            .path()
            .join(format!("run_{}.parquet", self.run_count));
        self.run_count += 1;
        let buffer = self.take_buffer();
        let row_converter = self.row_converter.clone();
        let merge_sort_batches = self.options.merge_sort_batches;
        let sorting_columns = self
            .properties
            .sorting_columns()
            .ok_or(SortingParquetError::NoSortingColumnsConfigured)?
            .clone();
        let run_file_props = self.options.run_file_properties.clone().unwrap_or_else(|| {
            WriterProperties::builder()
                .set_write_page_header_statistics(false)
                .set_statistics_enabled(EnabledStatistics::None)
                .build()
        });
        let schema = self.schema.clone();
        self.join_set.spawn_blocking(move || {
            // Sort the combined batch and extract min/max sort keys in one pass
            let (sorted, (min_sort_key, max_sort_key)) = if merge_sort_batches {
                Self::flush_to_run_merge_sort(buffer, sorting_columns, row_converter)?
            } else {
                Self::flush_to_run_concat_and_sort(
                    buffer,
                    sorting_columns,
                    row_converter,
                    schema.clone(),
                )?
            };
            let run_file = File::create(&run_path)?;
            let mut run_writer =
                ArrowWriter::try_new(run_file, schema.clone(), Some(run_file_props))?;

            run_writer.write(&sorted)?;
            run_writer.close()?;

            run_files.lock().push(RunInfo {
                path: run_path,
                min_sort_key: Arc::new(min_sort_key),
                max_sort_key: Arc::new(max_sort_key),
            });
            Ok(())
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::test::get_test_dir;
    use crate::writers::FlushThreshold;

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

    #[tokio::test]
    async fn test_sorting_parquet_writer() {
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
            ConcurrentSortingParquetWriter::try_new(test_file, schema.clone(), properties).unwrap();

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
            writer.write(&batch).await.unwrap();
        }
        writer.finish().await.unwrap();

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

    #[tokio::test]
    async fn test_sorting_writer_forced_spill() {
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
        let mut writer = ConcurrentSortingParquetWriter::try_new_with_options(
            file,
            schema.clone(),
            properties,
            options,
            BackgroundWorker::default(),
        )
        .unwrap();

        writer
            .write(&create_test_batch(vec![9, 7, 5], vec!["i", "g", "e"]))
            .await
            .unwrap();
        writer
            .write(&create_test_batch(vec![3, 1], vec!["c", "a"]))
            .await
            .unwrap();
        writer
            .write(&create_test_batch(vec![8, 6, 4], vec!["h", "f", "d"]))
            .await
            .unwrap();
        writer
            .write(&create_test_batch(vec![2, 0], vec!["b", "z"]))
            .await
            .unwrap();
        writer.finish().await.unwrap();

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

    #[tokio::test]
    async fn test_sorting_writer_single_run() {
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
        let mut writer = ConcurrentSortingParquetWriter::try_new_with_options(
            file,
            schema.clone(),
            properties,
            options,
            BackgroundWorker::default(),
        )
        .unwrap();

        writer
            .write(&create_test_batch(vec![3, 1, 2], vec!["c", "a", "b"]))
            .await
            .unwrap();
        writer.finish().await.unwrap();

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

    #[tokio::test]
    async fn test_multi_run_with_complex_types() {
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
        let mut writer = ConcurrentSortingParquetWriter::try_new_with_options(
            file,
            schema.clone(),
            props,
            options,
            BackgroundWorker::default(),
        )
        .unwrap();

        for _ in 0..3 {
            let items = TickerItem::random_instances(100_000);
            for chunk in items.chunks(128) {
                let batch = TickerItem::into_record_batch(chunk).unwrap();
                writer.write(&batch).await.unwrap();
            }
        }
        writer.finish().await.unwrap();

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

    #[tokio::test]
    async fn test_flush_threshold_bytes() {
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
        let mut writer = ConcurrentSortingParquetWriter::try_new_with_options(
            file,
            schema.clone(),
            properties,
            options,
            BackgroundWorker::default(),
        )
        .unwrap();

        writer
            .write(&create_test_batch(vec![3, 1], vec!["c", "a"]))
            .await
            .unwrap();
        writer.wait_for_background_tasks().await.unwrap();
        assert!(
            writer.num_run_files() > 0,
            "Should have spilled to run file"
        );
        assert_eq!(writer.in_progress_rows(), 0);
        assert_eq!(writer.in_progress_bytes(), 0);

        writer
            .write(&create_test_batch(vec![2, 0], vec!["b", "z"]))
            .await
            .unwrap();
        writer.finish().await.unwrap();

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

    #[tokio::test]
    async fn test_flush_threshold_either() {
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
        let mut writer = ConcurrentSortingParquetWriter::try_new_with_options(
            file,
            schema.clone(),
            properties,
            options,
            BackgroundWorker::default(),
        )
        .unwrap();

        writer
            .write(&create_test_batch(vec![3, 1, 2], vec!["c", "a", "b"]))
            .await
            .unwrap();
        assert!(
            writer.num_run_files() > 0,
            "Bytes threshold should have triggered"
        );

        writer.finish().await.unwrap();

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

    #[tokio::test]
    async fn test_flush_buffer_manual() {
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
            ConcurrentSortingParquetWriter::try_new(file, schema.clone(), properties).unwrap();

        writer
            .write(&create_test_batch(vec![3, 1], vec!["c", "a"]))
            .await
            .unwrap();
        assert_eq!(writer.num_run_files(), 0);
        assert!(writer.in_progress_rows() > 0);
        assert!(writer.in_progress_bytes() > 0);

        writer.flush_buffer().await.unwrap();
        assert_eq!(writer.num_run_files(), 1);
        assert_eq!(writer.in_progress_rows(), 0);
        assert_eq!(writer.in_progress_bytes(), 0);

        // Flush on empty buffer is a no-op
        writer.flush_buffer().await.unwrap();
        assert_eq!(writer.num_run_files(), 1);

        writer
            .write(&create_test_batch(vec![2, 0], vec!["b", "z"]))
            .await
            .unwrap();
        writer.finish().await.unwrap();

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

    #[tokio::test]
    async fn test_max_concurrent_workers_bound() {
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
        // One run per write, bounded to 2 concurrent spill tasks — exercises the
        // back-pressure path in `flush_to_run`.
        let options = SortingWriterOptions {
            flush_threshold: FlushThreshold::Rows(1),
            ..Default::default()
        };
        let background_options = BackgroundWorker {
            max_concurrent_workers: NonZeroUsize::new(2),
        };
        let mut writer = ConcurrentSortingParquetWriter::try_new_with_options(
            file,
            schema.clone(),
            properties,
            options,
            background_options,
        )
        .unwrap();

        // Write 20 single-row batches in descending order; output must come out
        // ascending and complete.
        for id in (0..20).rev() {
            writer
                .write(&create_test_batch(vec![id], vec!["x"]))
                .await
                .unwrap();
        }
        assert_eq!(writer.num_run_files(), 20);
        writer.finish().await.unwrap();

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
        assert_eq!(all_ids, (0..20).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn test_merge_sort_batches_path() {
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
        // merge_sort_batches sorts each incoming batch up front, then k-way
        // merges the buffer on flush. Force multiple runs to exercise it.
        let options = SortingWriterOptions {
            flush_threshold: FlushThreshold::Rows(4),
            merge_sort_batches: true,
            ..Default::default()
        };
        let mut writer = ConcurrentSortingParquetWriter::try_new_with_options(
            file,
            schema.clone(),
            properties,
            options,
            BackgroundWorker::default(),
        )
        .unwrap();

        writer
            .write(&create_test_batch(vec![5, 1], vec!["e", "a"]))
            .await
            .unwrap();
        writer
            .write(&create_test_batch(vec![9, 3], vec!["i", "c"]))
            .await
            .unwrap();
        writer
            .write(&create_test_batch(vec![7, 2], vec!["g", "b"]))
            .await
            .unwrap();
        writer
            .write(&create_test_batch(
                vec![8, 0, 6, 4],
                vec!["h", "z", "f", "d"],
            ))
            .await
            .unwrap();
        writer.finish().await.unwrap();

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
        assert_eq!(all_ids, (0..10).collect::<Vec<_>>());
    }
}
