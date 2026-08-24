use std::fs::File;
use std::io::Write;
use std::mem;
use std::sync::Arc;
mod options;
use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
pub use options::*;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use tempfile::TempDir;

use crate::SortingParquetError;
use crate::record_batch::streaming_merge::{RunInfo, SortedRunMerger};
use crate::sorting::SortExtremes;
use crate::writers::compaction::{
    CompactionFailure, CompactionJob, CompactionOutput, CompactionPolicy, CompactionStats,
    peak_fan_in, select_overlap_cluster,
};
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
    /// `Arc` so an in-flight [`CompactionJob`] can hold the directory alive:
    /// dropping the writer mid-job must not unlink the job's output path.
    temp_dir: Arc<TempDir>,
    run_files: Vec<RunInfo>,
    run_count: usize,
    /// Compaction jobs handed out but not yet applied or abandoned. Their runs
    /// are absent from `run_files`, so finishing with any outstanding is an
    /// error rather than silent data loss.
    runs_in_flight: usize,
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
            temp_dir: Arc::new(temp_dir),
            run_files: Vec::new(),
            run_count: 0,
            runs_in_flight: 0,
        })
    }

    // ── Writing ─────────────────────────────────────────────────────────

    /// Writes a [`RecordBatch`] to the writer.
    ///
    /// Data is buffered in memory and periodically sorted and flushed to
    /// temporary run files on disk when the configured [`FlushThreshold`] is reached.
    /// The batch schema must match the schema provided at construction.
    pub fn write(&mut self, batch: &RecordBatch) -> Result<(), SortingParquetError> {
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
    /// A no-op if the buffer is empty (in particular, calling it twice in a
    /// row produces only one run file).
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
        // Runs held by an outstanding compaction job aren't in `run_files`;
        // merging without them would silently drop rows.
        if self.runs_in_flight > 0 {
            return Err(SortingParquetError::CompactionInFlight(self.runs_in_flight));
        }

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
                let total_rows = self.run_files[0].num_rows;
                let file = File::open(&self.run_files[0].path)?;
                let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
                    .with_batch_size(output_batch_size)
                    .build()?;

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
                let total_rows = self.total_run_rows();

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

    /// Returns the sorted run files flushed so far, with their key ranges and
    /// sizes.
    ///
    /// Runs currently held by an outstanding [`CompactionJob`] are not
    /// included.
    pub fn run_files(&self) -> &[RunInfo] {
        &self.run_files
    }

    /// Returns the largest number of run files the final merge would need to
    /// hold open simultaneously.
    ///
    /// This — not [`num_run_files()`](Self::num_run_files) — is the number
    /// that determines whether [`finish()`](Self::finish) is expensive.
    /// [`SortedRunMerger`] opens a run only once the merge position reaches
    /// its `min_sort_key`, so runs with disjoint key ranges give `1` no matter
    /// how many there are, while fully overlapping runs give the full count.
    ///
    /// Use it to decide when to compact:
    ///
    /// ```rust,no_run
    /// # use sorting_parquet_writer::writers::{SortingParquetWriter, CompactionPolicy};
    /// # fn example(writer: &mut SortingParquetWriter<std::fs::File>) -> Result<(), Box<dyn std::error::Error>> {
    /// if writer.peak_merge_fan_in() > 128 {
    ///     writer.compact(&CompactionPolicy::default())?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn peak_merge_fan_in(&self) -> usize {
        peak_fan_in(&self.run_files)
    }

    /// Returns the number of compaction jobs handed out by
    /// [`take_compaction_job`](Self::take_compaction_job) that have not yet
    /// been applied or abandoned.
    ///
    /// [`finish()`](Self::finish) fails while this is non-zero.
    pub fn compactions_in_flight(&self) -> usize {
        self.runs_in_flight
    }

    // ── Compaction ──────────────────────────────────────────────────────

    /// Compacts overlapping run files in place, merging them into fewer,
    /// larger runs to bound the cost of the final merge.
    ///
    /// Returns `Ok(None)` when `policy` selects nothing — most importantly
    /// when peak fan-in is already within
    /// [`CompactionPolicy::target_fan_in`], which is always the case for runs
    /// with disjoint key ranges.
    ///
    /// This blocks the calling thread for the duration of the merge. To
    /// overlap compaction with writing, use
    /// [`take_compaction_job`](Self::take_compaction_job) instead — it is the
    /// same work, handed to you as a `Send + 'static` value.
    ///
    /// On failure the selected runs are returned to the writer, so a failed
    /// compaction never loses data.
    pub fn compact(
        &mut self,
        policy: &CompactionPolicy,
    ) -> Result<Option<CompactionStats>, SortingParquetError> {
        let Some(job) = self.take_compaction_job(policy) else {
            return Ok(None);
        };
        match job.run() {
            Ok(output) => {
                let stats = output.stats;
                self.apply_compaction(output)?;
                Ok(Some(stats))
            }
            Err(failure) => Err(self.abandon_compaction(failure)),
        }
    }

    /// Selects runs to compact and detaches them from the writer, returning a
    /// self-contained job.
    ///
    /// Returns `None` when the policy selects nothing to do.
    ///
    /// The returned [`CompactionJob`] is `Send + 'static` — run it on another
    /// thread, a pool, or `tokio::task::spawn_blocking`, then give it back
    /// with [`apply_compaction`](Self::apply_compaction) or
    /// [`abandon_compaction`](Self::abandon_compaction). The writer keeps
    /// accepting [`write()`](Self::write) calls in the meantime, but
    /// [`finish()`](Self::finish) fails until every job is returned.
    ///
    /// See the [`compaction`](crate::writers::compaction) module docs for a
    /// full example.
    pub fn take_compaction_job(&mut self, policy: &CompactionPolicy) -> Option<CompactionJob> {
        let sorting_columns = self.properties.sorting_columns()?.clone();
        let selected = select_overlap_cluster(&self.run_files, policy)?;

        // Remove back-to-front so the earlier indices stay valid.
        let mut inputs = Vec::with_capacity(selected.len());
        for &idx in selected.iter().rev() {
            inputs.push(self.run_files.remove(idx));
        }
        inputs.reverse();

        // `run_count` is monotonic, so this never collides with a removed run.
        let output_path = self
            .temp_dir
            .path()
            .join(format!("run_{}.parquet", self.run_count));
        self.run_count += 1;
        self.runs_in_flight += 1;

        Some(CompactionJob::new(
            inputs,
            output_path,
            self.schema.clone(),
            sorting_columns,
            self.run_file_properties(),
            policy.output_batch_size,
            self.temp_dir.clone(),
        ))
    }

    /// Adopts a completed compaction: the merged run joins the writer's run
    /// list and the runs it replaced are deleted from disk.
    ///
    /// Failures to unlink a replaced file are ignored — the run is already
    /// unreferenced, and the writer's temp directory removes it on drop
    /// regardless. Failing here would strand the freshly merged run instead.
    pub fn apply_compaction(
        &mut self,
        output: CompactionOutput,
    ) -> Result<(), SortingParquetError> {
        self.run_files.push(output.run);
        self.runs_in_flight = self.runs_in_flight.saturating_sub(1);
        for replaced in output.replaced {
            let _ = std::fs::remove_file(&replaced.path);
        }
        Ok(())
    }

    /// Returns the runs from a failed compaction to the writer, and hands back
    /// the error that caused it.
    ///
    /// The input files are untouched, so the writer is left exactly as it was
    /// before [`take_compaction_job`](Self::take_compaction_job) — only the
    /// wasted merge work is lost. The job has already cleaned up its partial
    /// output.
    ///
    /// The returned error is yours to propagate or log; discarding it is fine
    /// and simply retries the work later.
    pub fn abandon_compaction(&mut self, failure: CompactionFailure) -> SortingParquetError {
        self.run_files.extend(failure.inputs);
        self.runs_in_flight = self.runs_in_flight.saturating_sub(1);
        failure.error
    }

    /// The properties used when writing run files, matching `flush_to_run`.
    fn run_file_properties(&self) -> WriterProperties {
        self.options.run_file_properties.clone().unwrap_or_else(|| {
            WriterProperties::builder()
                .set_write_page_header_statistics(false)
                .set_statistics_enabled(EnabledStatistics::None)
                .build()
        })
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
    ///
    /// Individual settings are read through the getters on
    /// [`SortingWriterOptions`], e.g.
    /// [`flush_threshold()`](SortingWriterOptions::flush_threshold).
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

    /// Sum the row counts of all run files.
    ///
    /// Used as the fixed denominator (`total_rows`) for the [`FinishProgress`]
    /// reported via [`finish_with_progress`](Self::finish_with_progress).
    /// Reads the counts recorded on each [`RunInfo`] at flush time, so no
    /// footers are reopened.
    fn total_run_rows(&self) -> u64 {
        self.run_files.iter().map(|run| run.num_rows).sum()
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
    ///
    /// Picks between the two flush strategies based on
    /// [`SortingWriterOptions::merge_sort_batches`] and tags the resulting
    /// run with its min/max sort keys so
    /// [`SortedRunMerger`](crate::record_batch::streaming_merge::SortedRunMerger)
    /// can lazily activate runs by range during the final merge.
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
                .set_statistics_enabled(EnabledStatistics::None)
                .build()
        });

        let run_file = File::create(&run_path)?;
        let mut run_writer =
            ArrowWriter::try_new(run_file, self.schema.clone(), Some(run_file_props))?;

        let num_rows = sorted.num_rows() as u64;
        run_writer.write(&sorted)?;
        run_writer.close()?;
        let file_size = std::fs::metadata(&run_path)?.len();

        self.run_files.push(RunInfo {
            path: run_path,
            min_sort_key: Arc::new(min_sort_key),
            max_sort_key: Arc::new(max_sort_key),
            num_rows,
            file_size,
        });

        // Opt-in safety valve: keep the final merge's fan-in bounded so a
        // long-running writer can't accumulate enough overlapping runs to
        // exhaust file descriptors or memory at `finish()` time.
        if let Some(target_fan_in) = self.options.auto_compact_at {
            self.compact(&CompactionPolicy {
                target_fan_in,
                ..CompactionPolicy::default()
            })?;
        }
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
        let options = SortingWriterOptions::builder()
            .with_flush_after_rows(3)
            .build();
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
        let options = SortingWriterOptions::builder()
            .with_flush_after_rows(100)
            .build();
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
        let options = SortingWriterOptions::builder()
            .with_flush_after_rows(100_000)
            .build();
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
        let options = SortingWriterOptions::builder()
            .with_flush_after_bytes(1)
            .build();
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
        let options = SortingWriterOptions::builder()
            .with_flush_after_rows_or_bytes(usize::MAX, 1)
            .build();
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

    // ── Compaction ──────────────────────────────────────────────────────

    fn sorting_props() -> WriterProperties {
        WriterProperties::builder()
            .set_sorting_columns(Some(vec![SortingColumn {
                column_idx: 0,
                descending: false,
                nulls_first: false,
            }]))
            .build()
    }

    /// A writer that spills every `rows_per_run` rows into a run covering the
    /// whole key range, so the runs maximally overlap — the case compaction
    /// exists for.
    fn writer_with_overlapping_runs(
        file: std::fs::File,
        num_runs: usize,
        rows_per_run: usize,
        options: SortingWriterOptions,
    ) -> (SortingParquetWriter<std::fs::File>, Vec<i32>) {
        let schema = create_test_schema();
        let mut writer =
            SortingParquetWriter::try_new_with_options(file, schema, sorting_props(), options)
                .unwrap();

        let mut expected = Vec::new();
        for run in 0..num_runs {
            // Stride through the key space so every run spans it end to end.
            let ids: Vec<i32> = (0..rows_per_run)
                .map(|i| (i * num_runs + run) as i32)
                .collect();
            let names: Vec<String> = ids.iter().map(|id| format!("n{id}")).collect();
            let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            expected.extend_from_slice(&ids);
            writer
                .write(&create_test_batch(ids.clone(), name_refs))
                .unwrap();
        }
        expected.sort_unstable();
        (writer, expected)
    }

    fn read_ids(temp: &tempfile::NamedTempFile) -> Vec<i32> {
        let file = temp.reopen().unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let mut ids = Vec::new();
        for batch in reader {
            let batch = batch.unwrap();
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                ids.push(col.value(i));
            }
        }
        ids
    }

    fn overlapping_options(rows_per_run: usize) -> SortingWriterOptions {
        SortingWriterOptions::builder()
            .with_flush_after_rows(rows_per_run)
            .build()
    }

    #[test]
    fn test_compaction_reduces_fan_in_and_preserves_order() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let (mut writer, expected) =
            writer_with_overlapping_runs(temp.reopen().unwrap(), 12, 50, overlapping_options(50));

        assert_eq!(writer.num_run_files(), 12);
        assert_eq!(writer.peak_merge_fan_in(), 12);

        let policy = CompactionPolicy {
            target_fan_in: 4,
            max_merge_inputs: 5,
            ..Default::default()
        };
        let stats = writer.compact(&policy).unwrap().expect("should compact");
        assert_eq!(stats.input_runs, 5);
        assert_eq!(stats.rows, 5 * 50);
        assert!(stats.bytes_written > 0);

        // Five runs became one.
        assert_eq!(writer.num_run_files(), 8);
        assert_eq!(writer.peak_merge_fan_in(), 8);

        writer.finish().unwrap();
        assert_eq!(read_ids(&temp), expected);
    }

    #[test]
    fn test_compaction_loop_reaches_target_fan_in() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let (mut writer, expected) =
            writer_with_overlapping_runs(temp.reopen().unwrap(), 16, 40, overlapping_options(40));

        let policy = CompactionPolicy {
            target_fan_in: 4,
            max_merge_inputs: 4,
            ..Default::default()
        };
        while writer.compact(&policy).unwrap().is_some() {}

        assert!(
            writer.peak_merge_fan_in() <= 4,
            "fan-in {} should be within target",
            writer.peak_merge_fan_in()
        );

        writer.finish().unwrap();
        assert_eq!(read_ids(&temp), expected);
    }

    #[test]
    fn test_compaction_output_matches_uncompacted_output() {
        // The whole feature must be observationally invisible in the output.
        let plain = tempfile::NamedTempFile::new().unwrap();
        let (writer, expected) =
            writer_with_overlapping_runs(plain.reopen().unwrap(), 10, 60, overlapping_options(60));
        writer.finish().unwrap();

        let compacted = tempfile::NamedTempFile::new().unwrap();
        let (mut writer, _) = writer_with_overlapping_runs(
            compacted.reopen().unwrap(),
            10,
            60,
            overlapping_options(60),
        );
        let policy = CompactionPolicy {
            target_fan_in: 2,
            max_merge_inputs: 4,
            ..Default::default()
        };
        while writer.compact(&policy).unwrap().is_some() {}
        writer.finish().unwrap();

        assert_eq!(read_ids(&plain), expected);
        assert_eq!(read_ids(&compacted), expected);
    }

    #[test]
    fn test_compaction_down_to_single_run_uses_copy_through() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let (mut writer, expected) =
            writer_with_overlapping_runs(temp.reopen().unwrap(), 6, 30, overlapping_options(30));

        let policy = CompactionPolicy {
            target_fan_in: 1,
            max_merge_inputs: 16,
            ..Default::default()
        };
        writer.compact(&policy).unwrap().expect("should compact");
        assert_eq!(writer.num_run_files(), 1);

        // finish() now takes the CopyThrough branch.
        writer.finish().unwrap();
        assert_eq!(read_ids(&temp), expected);
    }

    #[test]
    fn test_compaction_deletes_replaced_run_files() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let (mut writer, _) =
            writer_with_overlapping_runs(temp.reopen().unwrap(), 6, 30, overlapping_options(30));

        let policy = CompactionPolicy {
            target_fan_in: 2,
            max_merge_inputs: 4,
            ..Default::default()
        };
        let job = writer.take_compaction_job(&policy).expect("should select");
        let replaced: Vec<_> = job.input_runs().iter().map(|r| r.path.clone()).collect();
        assert_eq!(replaced.len(), 4);
        assert!(replaced.iter().all(|p| p.exists()));

        let output = job.run().unwrap();
        let new_path = output.run.path.clone();
        writer.apply_compaction(output).unwrap();

        assert!(
            replaced.iter().all(|p| !p.exists()),
            "replaced run files should be unlinked"
        );
        assert!(new_path.exists());
    }

    #[test]
    fn test_compaction_noop_cases() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let schema = create_test_schema();
        let mut writer =
            SortingParquetWriter::try_new(temp.reopen().unwrap(), schema, sorting_props()).unwrap();
        let policy = CompactionPolicy::default();

        // No runs at all.
        assert!(writer.compact(&policy).unwrap().is_none());
        assert_eq!(writer.peak_merge_fan_in(), 0);

        // A single run is never worth compacting.
        writer
            .write(&create_test_batch(vec![3, 1], vec!["c", "a"]))
            .unwrap();
        writer.flush_buffer().unwrap();
        assert_eq!(writer.num_run_files(), 1);
        assert!(writer.compact(&policy).unwrap().is_none());
        assert_eq!(writer.num_run_files(), 1);
    }

    #[test]
    fn test_compaction_skips_disjoint_runs() {
        // Runs written in ascending order don't overlap, so the merger already
        // handles them one file at a time and compaction must decline.
        let temp = tempfile::NamedTempFile::new().unwrap();
        let schema = create_test_schema();
        let mut writer = SortingParquetWriter::try_new_with_options(
            temp.reopen().unwrap(),
            schema,
            sorting_props(),
            overlapping_options(20),
        )
        .unwrap();

        for run in 0..10i32 {
            let ids: Vec<i32> = (0..20).map(|i| run * 20 + i).collect();
            let names: Vec<String> = ids.iter().map(|id| format!("n{id}")).collect();
            let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            writer.write(&create_test_batch(ids, name_refs)).unwrap();
        }

        assert_eq!(writer.num_run_files(), 10);
        assert_eq!(writer.peak_merge_fan_in(), 1);

        let policy = CompactionPolicy {
            target_fan_in: 1,
            ..Default::default()
        };
        assert!(writer.compact(&policy).unwrap().is_none());
        assert_eq!(writer.num_run_files(), 10);
    }

    #[test]
    fn test_compaction_job_runs_on_another_thread() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let (mut writer, expected) =
            writer_with_overlapping_runs(temp.reopen().unwrap(), 10, 50, overlapping_options(50));

        let policy = CompactionPolicy {
            target_fan_in: 2,
            max_merge_inputs: 6,
            ..Default::default()
        };
        let job = writer.take_compaction_job(&policy).expect("should select");
        assert_eq!(writer.compactions_in_flight(), 1);
        assert_eq!(writer.num_run_files(), 4);
        assert_eq!(job.estimated_rows(), 6 * 50);

        // The job is Send + 'static, so this compiles and runs off-thread.
        let output = std::thread::spawn(move || job.run())
            .join()
            .expect("compaction thread panicked")
            .expect("compaction should succeed");

        writer.apply_compaction(output).unwrap();
        assert_eq!(writer.compactions_in_flight(), 0);
        assert_eq!(writer.num_run_files(), 5);

        writer.finish().unwrap();
        assert_eq!(read_ids(&temp), expected);
    }

    #[test]
    fn test_writes_continue_while_a_compaction_job_is_in_flight() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let (mut writer, mut expected) =
            writer_with_overlapping_runs(temp.reopen().unwrap(), 8, 40, overlapping_options(40));

        let policy = CompactionPolicy {
            target_fan_in: 2,
            max_merge_inputs: 4,
            ..Default::default()
        };
        let job = writer.take_compaction_job(&policy).expect("should select");
        let handle = std::thread::spawn(move || job.run());

        // Keep writing against detached runs.
        let extra: Vec<i32> = vec![-5, -3, -1];
        let names: Vec<String> = extra.iter().map(|id| format!("n{id}")).collect();
        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        writer
            .write(&create_test_batch(extra.clone(), name_refs))
            .unwrap();
        expected.extend_from_slice(&extra);
        expected.sort_unstable();

        let output = handle.join().unwrap().unwrap();
        writer.apply_compaction(output).unwrap();
        writer.finish().unwrap();

        assert_eq!(read_ids(&temp), expected);
    }

    #[test]
    fn test_finish_rejects_an_in_flight_compaction() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let (mut writer, _) =
            writer_with_overlapping_runs(temp.reopen().unwrap(), 6, 30, overlapping_options(30));

        let policy = CompactionPolicy {
            target_fan_in: 2,
            max_merge_inputs: 4,
            ..Default::default()
        };
        let job = writer.take_compaction_job(&policy).expect("should select");

        // Finishing now would silently drop the detached runs.
        let err = writer.finish().unwrap_err();
        assert!(
            matches!(err, SortingParquetError::CompactionInFlight(1)),
            "expected CompactionInFlight, got {err:?}"
        );
        drop(job);
    }

    #[test]
    fn test_abandon_compaction_restores_runs() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let (mut writer, expected) =
            writer_with_overlapping_runs(temp.reopen().unwrap(), 6, 30, overlapping_options(30));

        let policy = CompactionPolicy {
            target_fan_in: 2,
            max_merge_inputs: 4,
            ..Default::default()
        };
        let job = writer.take_compaction_job(&policy).expect("should select");
        assert_eq!(writer.num_run_files(), 2);

        let failure = CompactionFailure {
            error: SortingParquetError::WriterClosed,
            inputs: job.input_runs().to_vec(),
        };
        drop(job);
        let err = writer.abandon_compaction(failure);
        assert!(matches!(err, SortingParquetError::WriterClosed));

        assert_eq!(writer.num_run_files(), 6);
        assert_eq!(writer.compactions_in_flight(), 0);

        // No rows lost.
        writer.finish().unwrap();
        assert_eq!(read_ids(&temp), expected);
    }

    #[test]
    fn test_auto_compact_at_bounds_fan_in() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let options = SortingWriterOptions::builder()
            .with_flush_after_rows(40)
            .with_auto_compact_at(4)
            .build();
        let (writer, expected) =
            writer_with_overlapping_runs(temp.reopen().unwrap(), 20, 40, options);

        assert!(
            writer.peak_merge_fan_in() <= 4,
            "auto-compaction should have bounded fan-in, got {}",
            writer.peak_merge_fan_in()
        );

        writer.finish().unwrap();
        assert_eq!(read_ids(&temp), expected);
    }
}
