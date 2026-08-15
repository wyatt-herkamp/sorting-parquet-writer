//! Compaction of sorted run files during the write phase.
//!
//! [`SortingParquetWriter`](crate::writers::SortingParquetWriter) spills a new
//! run file every time its buffer fills. A long-running writer can accumulate
//! thousands of them, and every run that is *active* at the same moment during
//! the final merge costs an open file descriptor plus a decoded batch and its
//! `Rows` encoding. Compaction merges selected runs back into fewer, larger
//! runs so the final merge stays cheap.
//!
//! # What is actually worth compacting
//!
//! [`SortedRunMerger`](crate::record_batch::streaming_merge::SortedRunMerger)
//! activates runs lazily: a run's file is opened only once the merge position
//! reaches its [`min_sort_key`](RunInfo::min_sort_key). So when run key-ranges
//! are **disjoint**, the merger already holds about one file open at a time
//! and compacting them buys nothing — it just rewrites data.
//!
//! The cost comes from **overlapping** runs, which must all be open at once.
//! The number that matters is therefore not the run count but the *peak
//! interval-stabbing depth* over the `[min_sort_key, max_sort_key]` ranges —
//! the most runs the merger will ever have open simultaneously. See
//! [`peak_fan_in`].
//!
//! ```text
//! runs:  A [==========]
//!        B      [==========]
//!        C          [=========]
//!        D                        [======]
//!               ^ peak depth 3
//!
//! -> compact {A, B, C}; leave D alone. Peak fan-in 3 -> 1.
//! ```
//!
//! Merging k mutually-overlapping runs into one drops peak fan-in by exactly
//! k-1. [`select_overlap_cluster`] finds that cluster with a sweep line and
//! returns nothing at all when the runs are already spread out enough.
//!
//! # Running compaction off the write thread
//!
//! A [`CompactionJob`] owns everything it needs — the input runs, the schema,
//! the sorting columns, the writer properties, and its output path — and
//! shares no state with the writer that produced it. It is `Send + 'static`,
//! so it can be handed to a thread, a thread pool, or
//! `tokio::task::spawn_blocking` without this crate depending on any async
//! runtime.
//!
//! ```rust,no_run
//! # use sorting_parquet_writer::writers::{SortingParquetWriter, CompactionPolicy};
//! # fn example(writer: &mut SortingParquetWriter<std::fs::File>) -> Result<(), Box<dyn std::error::Error>> {
//! let policy = CompactionPolicy::default();
//!
//! if let Some(job) = writer.take_compaction_job(&policy) {
//!     // `job` is Send + 'static — this could equally be spawn_blocking.
//!     let handle = std::thread::spawn(move || job.run());
//!
//!     match handle.join().expect("compaction thread panicked") {
//!         Ok(output) => writer.apply_compaction(output)?,
//!         Err(failure) => {
//!             // Hand the runs back so no data is lost; retry later.
//!             let err = writer.abandon_compaction(failure);
//!             eprintln!("compaction failed, runs restored: {err}");
//!         }
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! While a job is outstanding its runs are not tracked by the writer, so
//! [`finish`](crate::writers::SortingParquetWriter::finish) refuses to run
//! with [`SortingParquetError::CompactionInFlight`] until every job has been
//! applied or abandoned.

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::SortingColumn;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use crate::SortingParquetError;
use crate::record_batch::streaming_merge::{RunInfo, SortedRunMerger};

/// Default number of simultaneously-active runs tolerated before compaction
/// kicks in.
const DEFAULT_TARGET_FAN_IN: usize = 64;
/// Default cap on how many runs a single compaction job will merge.
const DEFAULT_MAX_MERGE_INPUTS: usize = 16;
/// Default row count of batches produced while compacting.
const DEFAULT_COMPACTION_BATCH_SIZE: usize = 1_000_000;

/// Controls which runs a compaction pass selects, and how much work it does.
///
/// The defaults are tuned to leave small writers completely alone: with a
/// [`target_fan_in`](Self::target_fan_in) of 64, compaction does nothing until
/// the final merge would need to hold 65+ run files open at once.
///
/// # Example
///
/// ```rust
/// use sorting_parquet_writer::writers::CompactionPolicy;
///
/// // Aggressive: keep the merge fan-in very low.
/// let policy = CompactionPolicy {
///     target_fan_in: 8,
///     max_merge_inputs: 32,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct CompactionPolicy {
    /// Leave the runs alone while peak fan-in is at or below this value.
    ///
    /// This is the knob that matters. Because disjoint runs have a peak fan-in
    /// of 1 regardless of how many there are, a writer whose input arrives in
    /// roughly sorted order will never compact no matter how long it runs.
    ///
    /// Default: `64`
    pub target_fan_in: usize,

    /// Maximum number of runs merged by a single compaction job.
    ///
    /// Bounds the duration of one job so it can be interleaved with writing.
    /// When the overlapping cluster is larger than this, the smallest runs by
    /// [`RunInfo::file_size`] are chosen — the same fan-in reduction for the
    /// fewest bytes rewritten.
    ///
    /// Default: `16`
    pub max_merge_inputs: usize,

    /// Minimum number of runs worth merging. Values below `2` are treated as
    /// `2`, since compacting a single run is a pure copy.
    ///
    /// Default: `2`
    pub min_merge_inputs: usize,

    /// Skip runs that would push the compacted output past this many rows.
    ///
    /// Prevents compaction from building one enormous run that later passes
    /// would have to rewrite in full. `None` means no limit.
    ///
    /// Default: `None`
    pub max_output_rows: Option<u64>,

    /// Row count of the batches streamed from the merger into the compacted
    /// run file.
    ///
    /// Default: `1_000_000`
    pub output_batch_size: usize,
}

impl Default for CompactionPolicy {
    fn default() -> Self {
        Self {
            target_fan_in: DEFAULT_TARGET_FAN_IN,
            max_merge_inputs: DEFAULT_MAX_MERGE_INPUTS,
            min_merge_inputs: 2,
            max_output_rows: None,
            output_batch_size: DEFAULT_COMPACTION_BATCH_SIZE,
        }
    }
}

/// Sweep the run key-ranges and return `(peak depth, runs live at the peak)`.
///
/// Events are ordered by key with starts before ends at equal keys, matching
/// the merger's inclusive `min_sort_key <= current position` activation test,
/// so runs that merely touch at an endpoint count as overlapping.
fn sweep_overlap(runs: &[RunInfo]) -> (usize, Vec<usize>) {
    if runs.is_empty() {
        return (0, Vec::new());
    }

    // (key, 0 = start | 1 = end, run index)
    let mut events: Vec<(&[u8], u8, usize)> = Vec::with_capacity(runs.len() * 2);
    for (idx, run) in runs.iter().enumerate() {
        events.push((run.min_sort_key.as_slice(), 0, idx));
        events.push((run.max_sort_key.as_slice(), 1, idx));
    }
    events.sort_unstable_by(|a, b| a.0.cmp(b.0).then_with(|| a.1.cmp(&b.1)));

    let mut live: Vec<usize> = Vec::new();
    let mut peak = 0usize;
    let mut peak_set: Vec<usize> = Vec::new();

    for (_, kind, idx) in events {
        if kind == 0 {
            live.push(idx);
            // Depth only ever increases on a start event.
            if live.len() > peak {
                peak = live.len();
                peak_set.clear();
                peak_set.extend_from_slice(&live);
            }
        } else if let Some(pos) = live.iter().position(|&live_idx| live_idx == idx) {
            live.swap_remove(pos);
        }
    }

    (peak, peak_set)
}

/// Returns the largest number of run files the final merge would need to hold
/// open at the same time.
///
/// This is the peak stabbing depth of the runs' `[min_sort_key, max_sort_key]`
/// ranges. Disjoint runs give `1` no matter how many there are; fully
/// overlapping runs give `runs.len()`. An empty slice gives `0`.
///
/// This is the quantity [`CompactionPolicy::target_fan_in`] bounds.
pub fn peak_fan_in(runs: &[RunInfo]) -> usize {
    sweep_overlap(runs).0
}

/// Choose the set of runs to compact, as indices into `runs`.
///
/// Returns `None` when compaction would not pay for itself: when peak fan-in
/// is already within [`CompactionPolicy::target_fan_in`], or when the policy's
/// caps leave fewer than [`CompactionPolicy::min_merge_inputs`] candidates.
///
/// The returned indices are sorted ascending.
///
/// This function performs no I/O — it looks only at the sort-key ranges and
/// sizes recorded on each [`RunInfo`].
pub fn select_overlap_cluster(runs: &[RunInfo], policy: &CompactionPolicy) -> Option<Vec<usize>> {
    let (peak, mut cluster) = sweep_overlap(runs);
    if peak <= policy.target_fan_in {
        return None;
    }

    // Same fan-in reduction whichever members we drop, so keep the cheapest
    // ones to rewrite. Tie-break on index to stay deterministic.
    if cluster.len() > policy.max_merge_inputs {
        cluster.sort_unstable_by_key(|&idx| (runs[idx].file_size, idx));
        cluster.truncate(policy.max_merge_inputs);
    }

    if let Some(max_output_rows) = policy.max_output_rows {
        cluster.sort_unstable_by_key(|&idx| (runs[idx].num_rows, idx));
        let mut total = 0u64;
        let mut kept = Vec::with_capacity(cluster.len());
        for &idx in &cluster {
            let next = total.saturating_add(runs[idx].num_rows);
            if next > max_output_rows {
                break;
            }
            total = next;
            kept.push(idx);
        }
        cluster = kept;
    }

    // Merging a single run is a pure copy, so never go below 2.
    if cluster.len() < policy.min_merge_inputs.max(2) {
        return None;
    }

    cluster.sort_unstable();
    Some(cluster)
}

/// Progress of a single compaction job.
#[derive(Debug, Clone)]
pub struct CompactionProgress {
    /// Rows written to the compacted run so far.
    pub rows_written: u64,
    /// Batches written to the compacted run so far.
    pub batches_written: u64,
    /// Total rows across the job's input runs.
    pub total_rows: u64,
    /// Number of input runs being merged.
    pub input_runs: usize,
}

impl CompactionProgress {
    /// Returns progress as a fraction in `[0.0, 1.0]`.
    pub fn fraction_complete(&self) -> f64 {
        if self.total_rows == 0 {
            1.0
        } else {
            self.rows_written as f64 / self.total_rows as f64
        }
    }
}

/// Trait for receiving progress updates while a [`CompactionJob`] runs.
///
/// Mirrors [`FinishProgressHandler`](crate::writers::FinishProgressHandler):
/// implement it on a struct, or pass a closure via the blanket impl for
/// `FnMut(&CompactionProgress)`.
pub trait CompactionProgressHandler {
    /// Called after each batch is written to the compacted run file.
    fn on_batch_written(&mut self, progress: &CompactionProgress);
}

impl<F: FnMut(&CompactionProgress)> CompactionProgressHandler for F {
    fn on_batch_written(&mut self, progress: &CompactionProgress) {
        self(progress);
    }
}

/// Sentinel handler used by [`CompactionJob::run`] when the caller doesn't
/// supply a progress callback. Every call is a no-op and gets inlined away.
pub(crate) struct NoopCompactionProgressHandler;

impl CompactionProgressHandler for NoopCompactionProgressHandler {
    fn on_batch_written(&mut self, _: &CompactionProgress) {}
}

/// What a compaction job cost and accomplished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionStats {
    /// Number of run files merged.
    pub input_runs: usize,
    /// Rows written to the compacted run.
    pub rows: u64,
    /// Total on-disk size of the input runs, in bytes.
    pub bytes_read: u64,
    /// On-disk size of the compacted run, in bytes.
    pub bytes_written: u64,
}

/// A successful compaction, ready to be handed back to the writer via
/// [`SortingParquetWriter::apply_compaction`](crate::writers::SortingParquetWriter::apply_compaction).
#[derive(Debug)]
pub struct CompactionOutput {
    /// The newly written, compacted run.
    pub run: RunInfo,
    /// The input runs it replaces. Their files are deleted by
    /// `apply_compaction`.
    pub replaced: Vec<RunInfo>,
    /// What the job cost.
    pub stats: CompactionStats,
}

/// A failed compaction. Carries the untouched input runs back to the caller so
/// they can be returned to the writer with
/// [`SortingParquetWriter::abandon_compaction`](crate::writers::SortingParquetWriter::abandon_compaction)
/// rather than lost.
///
/// The job removes its own partially written output file before returning
/// this, so no orphan is left behind.
#[derive(Debug)]
pub struct CompactionFailure {
    /// Why the compaction failed.
    pub error: SortingParquetError,
    /// The input runs, still intact on disk.
    pub inputs: Vec<RunInfo>,
}

impl std::fmt::Display for CompactionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "compaction of {} run(s) failed: {}",
            self.inputs.len(),
            self.error
        )
    }
}

impl std::error::Error for CompactionFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl From<CompactionFailure> for SortingParquetError {
    fn from(failure: CompactionFailure) -> Self {
        failure.error
    }
}

/// A self-contained unit of compaction work.
///
/// Produced by
/// [`SortingParquetWriter::take_compaction_job`](crate::writers::SortingParquetWriter::take_compaction_job),
/// which removes the selected runs from the writer. The job borrows nothing
/// from the writer and is `Send + 'static`, so [`run`](Self::run) can execute
/// on any thread. See the [module docs](self) for a worked example.
///
/// The job must be given back to the writer — via `apply_compaction` on
/// success or `abandon_compaction` on failure — before the writer can finish.
pub struct CompactionJob {
    inputs: Vec<RunInfo>,
    output_path: PathBuf,
    schema: SchemaRef,
    sorting_columns: Vec<SortingColumn>,
    run_file_properties: WriterProperties,
    output_batch_size: usize,
    /// Keeps the writer's temp directory alive for the duration of the job, so
    /// dropping the writer mid-compaction can't unlink the output's parent.
    _temp_dir: Arc<TempDir>,
}

impl CompactionJob {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        inputs: Vec<RunInfo>,
        output_path: PathBuf,
        schema: SchemaRef,
        sorting_columns: Vec<SortingColumn>,
        run_file_properties: WriterProperties,
        output_batch_size: usize,
        temp_dir: Arc<TempDir>,
    ) -> Self {
        Self {
            inputs,
            output_path,
            schema,
            sorting_columns,
            run_file_properties,
            output_batch_size,
            _temp_dir: temp_dir,
        }
    }

    /// The runs this job will merge.
    pub fn input_runs(&self) -> &[RunInfo] {
        &self.inputs
    }

    /// Total rows across the input runs — the exact size of the output, and a
    /// good proxy for how long the job will take.
    pub fn estimated_rows(&self) -> u64 {
        self.inputs.iter().map(|run| run.num_rows).sum()
    }

    /// Merges the input runs into a single compacted run file.
    ///
    /// This is the expensive part, and the part that is safe to run on another
    /// thread.
    pub fn run(self) -> Result<CompactionOutput, CompactionFailure> {
        self.run_with_progress(NoopCompactionProgressHandler)
    }

    /// Like [`run`](Self::run), but reports progress after each batch written
    /// to the compacted run.
    pub fn run_with_progress(
        self,
        mut handler: impl CompactionProgressHandler,
    ) -> Result<CompactionOutput, CompactionFailure> {
        let total_rows = self.estimated_rows();
        let bytes_read: u64 = self.inputs.iter().map(|run| run.file_size).sum();
        let input_runs = self.inputs.len();

        match self.execute(&mut handler, total_rows) {
            Ok((run, rows)) => {
                let stats = CompactionStats {
                    input_runs,
                    rows,
                    bytes_read,
                    bytes_written: run.file_size,
                };
                Ok(CompactionOutput {
                    run,
                    replaced: self.inputs,
                    stats,
                })
            }
            Err(error) => {
                // Never leave a half-written run behind.
                let _ = std::fs::remove_file(&self.output_path);
                Err(CompactionFailure {
                    error,
                    inputs: self.inputs,
                })
            }
        }
    }

    /// Does the merge. Returns the new run plus its row count, or the first
    /// error encountered. Takes `&self` so the caller still owns `inputs` and
    /// can hand them back on failure.
    fn execute(
        &self,
        handler: &mut impl CompactionProgressHandler,
        total_rows: u64,
    ) -> Result<(RunInfo, u64), SortingParquetError> {
        // `RowConverter` is not `Clone`, so the job builds its own rather than
        // taking the writer's. A second one is used for the extremes below;
        // both are cheap next to the merge itself.
        let row_converter =
            crate::sorting::create_row_converter(&self.sorting_columns, self.schema.as_ref())?;
        let extremes_converter =
            crate::sorting::create_row_converter(&self.sorting_columns, self.schema.as_ref())?;

        let merger = SortedRunMerger::try_new(
            self.inputs.clone(),
            self.sorting_columns.clone(),
            row_converter,
            self.output_batch_size,
        )?;

        let file = File::create(&self.output_path)?;
        let mut writer = ArrowWriter::try_new(
            file,
            self.schema.clone(),
            Some(self.run_file_properties.clone()),
        )?;

        let mut progress = CompactionProgress {
            rows_written: 0,
            batches_written: 0,
            total_rows,
            input_runs: self.inputs.len(),
        };
        let mut min_sort_key: Option<Vec<u8>> = None;
        let mut max_sort_key: Option<Vec<u8>> = None;

        for batch in merger {
            let batch = batch?;
            let num_rows = batch.num_rows();
            if num_rows == 0 {
                continue;
            }

            // The merger emits globally sorted batches, so the first row of
            // the first batch and the last row of the last batch are the
            // extremes. Encode just those single rows.
            if min_sort_key.is_none() {
                min_sort_key = Some(encode_row(
                    &extremes_converter,
                    &batch,
                    &self.sorting_columns,
                    0,
                )?);
            }
            max_sort_key = Some(encode_row(
                &extremes_converter,
                &batch,
                &self.sorting_columns,
                num_rows - 1,
            )?);

            // Deliberately no per-batch `flush()` here, unlike the final
            // output path: a compacted run is an intermediate, so let the
            // Parquet writer pick its own row group boundaries.
            writer.write(&batch)?;

            progress.rows_written += num_rows as u64;
            progress.batches_written += 1;
            handler.on_batch_written(&progress);
        }

        writer.close()?;

        // Inputs are never empty runs, so the merge must have produced rows.
        let (min_sort_key, max_sort_key) = match (min_sort_key, max_sort_key) {
            (Some(min), Some(max)) => (min, max),
            _ => return Err(SortingParquetError::UnexpectedIndexOutOfBounds),
        };

        let file_size = std::fs::metadata(&self.output_path)?.len();

        Ok((
            RunInfo {
                path: self.output_path.clone(),
                min_sort_key: Arc::new(min_sort_key),
                max_sort_key: Arc::new(max_sort_key),
                num_rows: progress.rows_written,
                file_size,
            },
            progress.rows_written,
        ))
    }
}

/// Encode a single row's sort key by slicing the sort columns down to one row.
fn encode_row(
    row_converter: &arrow_row::RowConverter,
    batch: &arrow::array::RecordBatch,
    sorting_columns: &[SortingColumn],
    row_idx: usize,
) -> Result<Vec<u8>, SortingParquetError> {
    let cols: Vec<_> = sorting_columns
        .iter()
        .map(|col| batch.column(col.column_idx as usize).slice(row_idx, 1))
        .collect();
    let rows = row_converter.convert_columns(&cols)?;
    Ok(rows.row(0).as_ref().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `RunInfo` with no file behind it. Selection is pure, so these tests
    /// never touch the disk.
    fn run(min: &[u8], max: &[u8], num_rows: u64, file_size: u64) -> RunInfo {
        RunInfo {
            path: PathBuf::from("/nonexistent"),
            min_sort_key: Arc::new(min.to_vec()),
            max_sort_key: Arc::new(max.to_vec()),
            num_rows,
            file_size,
        }
    }

    /// `n` runs all spanning the same range, so every one overlaps every other.
    fn overlapping(n: usize) -> Vec<RunInfo> {
        (0..n).map(|_| run(b"a", b"z", 100, 1000)).collect()
    }

    /// `n` runs with strictly disjoint, ascending ranges.
    fn disjoint(n: usize) -> Vec<RunInfo> {
        (0..n)
            .map(|i| {
                let lo = [b'a' + (i as u8) * 2];
                let hi = [b'a' + (i as u8) * 2 + 1];
                run(&lo, &hi, 100, 1000)
            })
            .collect()
    }

    // ── peak_fan_in ─────────────────────────────────────────────────────

    #[test]
    fn peak_fan_in_of_no_runs_is_zero() {
        assert_eq!(peak_fan_in(&[]), 0);
    }

    #[test]
    fn peak_fan_in_of_disjoint_runs_is_one() {
        // The whole point: 50 disjoint runs cost the merger no more than 1.
        assert_eq!(peak_fan_in(&disjoint(12)), 1);
    }

    #[test]
    fn peak_fan_in_of_fully_overlapping_runs_is_the_count() {
        assert_eq!(peak_fan_in(&overlapping(7)), 7);
    }

    #[test]
    fn peak_fan_in_counts_nested_ranges() {
        let runs = vec![run(b"a", b"z", 1, 1), run(b"m", b"n", 1, 1)];
        assert_eq!(peak_fan_in(&runs), 2);
    }

    #[test]
    fn peak_fan_in_treats_touching_ranges_as_overlapping() {
        // The merger activates on `min <= position`, so a run starting exactly
        // where another ends is still open at the same moment.
        let runs = vec![run(b"a", b"c", 1, 1), run(b"c", b"e", 1, 1)];
        assert_eq!(peak_fan_in(&runs), 2);
    }

    #[test]
    fn peak_fan_in_handles_single_row_runs() {
        let runs = vec![run(b"k", b"k", 1, 1), run(b"k", b"k", 1, 1)];
        assert_eq!(peak_fan_in(&runs), 2);
    }

    #[test]
    fn peak_fan_in_uses_the_local_maximum_not_the_total() {
        // Two overlapping pairs, far apart. Peak is 2, not 4.
        let runs = vec![
            run(b"a", b"d", 1, 1),
            run(b"b", b"e", 1, 1),
            run(b"w", b"y", 1, 1),
            run(b"x", b"z", 1, 1),
        ];
        assert_eq!(peak_fan_in(&runs), 2);
    }

    // ── selection ───────────────────────────────────────────────────────

    #[test]
    fn disjoint_runs_are_never_selected() {
        // The core claim of the design: spread-out runs cost the final merge
        // nothing, so compacting them is pure write amplification.
        let runs = disjoint(12);
        assert!(select_overlap_cluster(&runs, &CompactionPolicy::default()).is_none());

        // Even with the threshold driven to zero, a fan-in of 1 yields a
        // one-run cluster, which is below the floor of 2.
        let policy = CompactionPolicy {
            target_fan_in: 0,
            ..Default::default()
        };
        assert!(select_overlap_cluster(&runs, &policy).is_none());
    }

    #[test]
    fn nothing_selected_while_within_target_fan_in() {
        let runs = overlapping(8);
        let policy = CompactionPolicy {
            target_fan_in: 8,
            ..Default::default()
        };
        assert!(select_overlap_cluster(&runs, &policy).is_none());
    }

    #[test]
    fn overlapping_cluster_is_selected_whole_when_under_the_cap() {
        let runs = overlapping(5);
        let policy = CompactionPolicy {
            target_fan_in: 2,
            max_merge_inputs: 16,
            ..Default::default()
        };
        assert_eq!(
            select_overlap_cluster(&runs, &policy).unwrap(),
            vec![0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn oversized_cluster_keeps_the_smallest_runs() {
        // Any k members give the same fan-in reduction, so pick the cheapest
        // bytes to rewrite: sizes 10, 20, 30 at indices 1, 3, 4.
        let runs = vec![
            run(b"a", b"z", 1, 50),
            run(b"a", b"z", 1, 10),
            run(b"a", b"z", 1, 40),
            run(b"a", b"z", 1, 20),
            run(b"a", b"z", 1, 30),
        ];
        let policy = CompactionPolicy {
            target_fan_in: 2,
            max_merge_inputs: 3,
            ..Default::default()
        };
        assert_eq!(
            select_overlap_cluster(&runs, &policy).unwrap(),
            vec![1, 3, 4]
        );
    }

    #[test]
    fn max_output_rows_trims_the_cluster() {
        let runs = vec![
            run(b"a", b"z", 10, 1),
            run(b"a", b"z", 20, 1),
            run(b"a", b"z", 30, 1),
            run(b"a", b"z", 40, 1),
        ];
        let policy = CompactionPolicy {
            target_fan_in: 2,
            max_output_rows: Some(60),
            ..Default::default()
        };
        // 10 + 20 + 30 fits; adding 40 would not.
        assert_eq!(
            select_overlap_cluster(&runs, &policy).unwrap(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn max_output_rows_smaller_than_any_pair_selects_nothing() {
        let runs = overlapping(4);
        let policy = CompactionPolicy {
            target_fan_in: 1,
            max_output_rows: Some(150), // each run is 100 rows
            ..Default::default()
        };
        assert!(select_overlap_cluster(&runs, &policy).is_none());
    }

    #[test]
    fn min_merge_inputs_is_respected() {
        let runs = overlapping(3);
        let policy = CompactionPolicy {
            target_fan_in: 1,
            min_merge_inputs: 4,
            ..Default::default()
        };
        assert!(select_overlap_cluster(&runs, &policy).is_none());
    }

    #[test]
    fn a_single_run_is_never_compacted() {
        // Merging one run is a pure copy, so the floor of 2 holds even when
        // the policy asks for less.
        let runs = vec![run(b"a", b"z", 100, 1000)];
        let policy = CompactionPolicy {
            target_fan_in: 0,
            min_merge_inputs: 1,
            ..Default::default()
        };
        assert!(select_overlap_cluster(&runs, &policy).is_none());
        assert!(select_overlap_cluster(&[], &policy).is_none());
    }

    #[test]
    fn only_the_overlapping_cluster_is_selected() {
        // Three runs pile up at the front; the tail run is disjoint and must
        // be left alone.
        let runs = vec![
            run(b"a", b"f", 1, 1),
            run(b"b", b"g", 1, 1),
            run(b"c", b"h", 1, 1),
            run(b"x", b"z", 1, 1),
        ];
        let policy = CompactionPolicy {
            target_fan_in: 2,
            ..Default::default()
        };
        assert_eq!(
            select_overlap_cluster(&runs, &policy).unwrap(),
            vec![0, 1, 2]
        );
    }

    // ── threading contract ──────────────────────────────────────────────

    #[test]
    fn compaction_job_is_send_and_static() {
        fn assert_send_static<T: Send + 'static>() {}
        assert_send_static::<CompactionJob>();
        assert_send_static::<CompactionOutput>();
        assert_send_static::<CompactionFailure>();
    }
}
