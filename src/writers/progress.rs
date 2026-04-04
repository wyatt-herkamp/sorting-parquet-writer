/// Information about the current state of the finish/merge phase.
#[derive(Debug, Clone)]
pub struct FinishProgress {
    /// Current phase of the finish operation.
    pub phase: FinishPhase,
    /// Number of rows written to the final output so far.
    pub rows_written: u64,
    /// Number of batches written to the final output so far.
    pub batches_written: u64,
    /// Total number of rows across all runs.
    /// Read from Parquet metadata before merging starts.
    pub total_rows: u64,
    /// Number of run files being merged.
    pub num_runs: usize,
}

/// The current phase of the finish operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishPhase {
    /// Copying through a single sorted run (no merge needed).
    CopyThrough,
    /// Merging multiple sorted runs via k-way merge.
    Merging,
}

impl FinishProgress {
    /// Returns progress as a fraction in `[0.0, 1.0]`.
    pub fn fraction_complete(&self) -> f64 {
        if self.total_rows == 0 {
            1.0
        } else {
            self.rows_written as f64 / self.total_rows as f64
        }
    }
}

/// Trait for receiving progress updates during the finish/merge phase.
///
/// Implement this on a struct to receive callbacks as batches are written
/// to the final output. Alternatively, pass a closure — there is a blanket
/// implementation for `FnMut(&FinishProgress)`.
///
/// # Example
///
/// ```rust,no_run
/// use sorting_parquet_writer::writers::FinishProgress;
///
/// // Using a closure:
/// # fn example(writer: sorting_parquet_writer::writers::SortingParquetWriter<std::fs::File>) {
/// writer.finish_with_progress(|p: &FinishProgress| {
///     println!("Merge progress: {:.1}%", p.fraction_complete() * 100.0);
/// }).unwrap();
/// # }
/// ```
pub trait FinishProgressHandler {
    /// Called after each batch is written to the final output.
    fn on_batch_written(&mut self, progress: &FinishProgress);
}

impl<F: FnMut(&FinishProgress)> FinishProgressHandler for F {
    fn on_batch_written(&mut self, progress: &FinishProgress) {
        self(progress);
    }
}

/// No-op handler that gets optimized away entirely.
pub(crate) struct NoopProgressHandler;

impl FinishProgressHandler for NoopProgressHandler {
    fn on_batch_written(&mut self, _: &FinishProgress) {}
}
