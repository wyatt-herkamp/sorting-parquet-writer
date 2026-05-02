//! Streaming k-way merge across sorted Parquet run files on disk.
//!
//! Used by [`SortingParquetWriter`](crate::writers::SortingParquetWriter)
//! during the finalize phase to combine its spilled run files into a single
//! globally sorted output.
//!
//! # Lazy activation
//!
//! Each run file carries its sort-key range as a [`RunInfo`]. Pending runs
//! are kept in a queue ordered by `min_sort_key` ascending, and a run is
//! opened only once the current heap minimum has caught up to its
//! `min_sort_key`. When run ranges are disjoint, this means the merger
//! holds at most one open file at a time; when ranges overlap, it
//! activates as many runs as needed to maintain a correct global order.
//!
//! Each open run reads one [`RecordBatch`] at a time from its file via
//! [`ParquetRecordBatchReader`], so total memory is bounded by approximately
//! one batch per active run plus the in-flight output batch.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::compute::interleave_record_batch;
use arrow_row::{RowConverter, Rows};
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::metadata::SortingColumn;

use crate::SortingParquetError;

/// A handle to one sorted Parquet run file produced by the writer's spill phase.
///
/// `min_sort_key` and `max_sort_key` are the encoded [`arrow_row`] bytes of
/// the first and last sort keys in the run, respectively. They are populated when the run is written and consumed by
/// [`SortedRunMerger`] to drive lazy activation: a run is held back until the
/// current merge position reaches its `min_sort_key`.
///
/// The keys are wrapped in [`Arc`] so the merger can sort by them without
/// cloning the underlying byte vectors.
#[derive(Clone)]
pub struct RunInfo {
    /// Filesystem path of the sorted Parquet run file.
    pub path: PathBuf,
    /// Encoded sort key of the first row in the run.
    pub min_sort_key: Arc<Vec<u8>>,
    /// Encoded sort key of the last row in the run.
    pub max_sort_key: Arc<Vec<u8>>,
}

/// A cursor into one sorted run file during merge.
struct RunCursor {
    run_idx: usize,
    reader: ParquetRecordBatchReader,
    current_batch: Option<RecordBatch>,
    current_rows: Option<Rows>,
    row_idx: usize,
}

impl RunCursor {
    /// Advance to the next batch from the reader. Returns true if a new batch was loaded.
    fn advance_batch(
        &mut self,
        row_converter: &RowConverter,
        sorting_columns: &[SortingColumn],
    ) -> Result<bool, SortingParquetError> {
        match self.reader.next() {
            Some(Ok(batch)) => {
                let rows = convert_rows(row_converter, &batch, sorting_columns)?;
                self.current_batch = Some(batch);
                self.current_rows = Some(rows);
                self.row_idx = 0;
                Ok(true)
            }
            Some(Err(e)) => Err(SortingParquetError::from(e)),
            None => {
                self.current_batch = None;
                self.current_rows = None;
                Ok(false)
            }
        }
    }

    fn current_batch_num_rows(&self) -> usize {
        self.current_batch
            .as_ref()
            .map(|b| b.num_rows())
            .unwrap_or(0)
    }

    /// Fill `buf` with the sort key bytes for the current row. Returns true if successful.
    fn fill_sort_key(&self, buf: &mut Vec<u8>) -> bool {
        if let Some(rows) = &self.current_rows {
            buf.clear();
            // SAFETY: `row_idx` was set to 0 on batch load and only advanced
            // after a `next_row_idx < current_batch_num_rows()` check, so it
            // is always a valid row index for `rows`.
            buf.extend_from_slice(unsafe { rows.row_unchecked(self.row_idx) }.as_ref());
            true
        } else {
            false
        }
    }
}

/// Heap entry for the k-way merge.
/// Stores an owned copy of the sort key bytes to avoid borrow issues.
/// Uses reverse ordering so `BinaryHeap` (max-heap) acts as a min-heap.
struct MergeEntry {
    run_idx: usize,
    sort_key: Vec<u8>,
}

impl Eq for MergeEntry {}

impl PartialEq for MergeEntry {
    fn eq(&self, other: &Self) -> bool {
        self.sort_key == other.sort_key
    }
}

impl Ord for MergeEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap behavior.
        // Row sort keys are comparable as raw bytes.
        other.sort_key.cmp(&self.sort_key)
    }
}

impl PartialOrd for MergeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn convert_rows(
    row_converter: &RowConverter,
    batch: &RecordBatch,
    sorting_columns: &[SortingColumn],
) -> Result<Rows, SortingParquetError> {
    let cols: Vec<_> = sorting_columns
        .iter()
        .map(|col| batch.column(col.column_idx as usize).clone())
        .collect();
    Ok(row_converter.convert_columns(&cols)?)
}

/// Streams merged, globally-sorted output from multiple sorted Parquet run files.
///
/// Implements a k-way merge with a binary min-heap keyed on the encoded
/// sort-key bytes of the head row of each active run. Each call to
/// [`next_batch`](Self::next_batch) produces up to `output_batch_size` rows
/// by repeatedly popping the heap, recording the `(batch, row)` of the
/// smallest sort key, and pushing the next row from the same run. The
/// output is assembled in one shot via
/// [`arrow::compute::interleave_record_batch`].
///
/// Pending runs are activated lazily based on their [`RunInfo::min_sort_key`]
/// (see the module-level docs). Implements [`Iterator`] so the merger can be
/// driven by a `for` loop or the standard combinators.
pub struct SortedRunMerger {
    /// Active runs with open readers participating in the merge.
    cursors: Vec<RunCursor>,
    /// Pending runs not yet opened, sorted by `min_sort_key` ascending.
    pending_runs: VecDeque<RunInfo>,
    sorting_columns: Vec<SortingColumn>,
    output_batch_size: usize,
    read_batch_size: usize,
    row_converter: RowConverter,
    heap: BinaryHeap<MergeEntry>,
    next_cursor_idx: usize,
}

impl SortedRunMerger {
    /// Builds a merger over the given sorted run files.
    ///
    /// `output_batch_size` is the maximum row count of each batch returned
    /// by [`next_batch`](Self::next_batch); the merger picks an internal
    /// per-run read batch size based on it and the run count to bound the
    /// total memory across all active cursors.
    ///
    /// `run_files` is sorted by [`RunInfo::min_sort_key`] internally; callers
    /// do not need to pre-sort. The first run plus any others sharing the
    /// same `min_sort_key` are opened eagerly so the heap has at least one
    /// entry before the first call to `next_batch`. If `run_files` is empty,
    /// the resulting merger yields `Ok(None)` on the first `next_batch`.
    pub fn try_new(
        mut run_files: Vec<RunInfo>,
        sorting_columns: Vec<SortingColumn>,
        row_converter: RowConverter,
        output_batch_size: usize,
    ) -> Result<Self, SortingParquetError> {
        let num_runs = run_files.len();

        // Adaptive read batch size: bound total memory across all runs
        let read_batch_size = std::cmp::max(1024, output_batch_size / std::cmp::max(1, num_runs));

        // Sort runs by min_sort_key so we can activate them in order
        run_files.sort_unstable_by(|a, b| a.min_sort_key.cmp(&b.min_sort_key));

        let mut pending_runs: VecDeque<RunInfo> = run_files.into();
        let mut cursors = Vec::with_capacity(num_runs);
        let mut heap = BinaryHeap::with_capacity(num_runs);
        let mut next_cursor_idx = 0;

        // Activate the first run to establish the initial merge position
        if let Some(first_run) = pending_runs.pop_front() {
            let initial_key = first_run.min_sort_key.clone();
            Self::activate_run(
                first_run,
                &mut cursors,
                &mut heap,
                &row_converter,
                &sorting_columns,
                read_batch_size,
                &mut next_cursor_idx,
            )?;

            // Also activate any other runs whose min_sort_key <= initial_key
            while let Some(front) = pending_runs.front() {
                if front.min_sort_key <= initial_key {
                    let run = pending_runs.pop_front().unwrap();
                    Self::activate_run(
                        run,
                        &mut cursors,
                        &mut heap,
                        &row_converter,
                        &sorting_columns,
                        read_batch_size,
                        &mut next_cursor_idx,
                    )?;
                } else {
                    break;
                }
            }
        }

        Ok(Self {
            cursors,
            pending_runs,
            sorting_columns,
            output_batch_size,
            read_batch_size,
            row_converter,
            heap,
            next_cursor_idx,
        })
    }

    /// Open a run file and add it to the active cursors + heap.
    fn activate_run(
        run: RunInfo,
        cursors: &mut Vec<RunCursor>,
        heap: &mut BinaryHeap<MergeEntry>,
        row_converter: &RowConverter,
        sorting_columns: &[SortingColumn],
        read_batch_size: usize,
        next_cursor_idx: &mut usize,
    ) -> Result<(), SortingParquetError> {
        let file = File::open(&run.path)?;
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?
            .with_batch_size(read_batch_size)
            .build()?;

        let first_batch = match reader.next() {
            Some(Ok(batch)) => batch,
            Some(Err(e)) => return Err(e.into()),
            None => return Ok(()), // Empty run, skip
        };

        let idx = *next_cursor_idx;
        *next_cursor_idx += 1;

        let rows = convert_rows(row_converter, &first_batch, sorting_columns)?;
        let sort_key = rows.row(0).as_ref().to_vec();

        cursors.push(RunCursor {
            run_idx: idx,
            reader,
            current_batch: Some(first_batch),
            current_rows: Some(rows),
            row_idx: 0,
        });

        heap.push(MergeEntry {
            run_idx: idx,
            sort_key,
        });

        Ok(())
    }

    /// Check if the next pending run should be activated based on the current heap minimum.
    fn should_activate_next_pending(&self) -> bool {
        match (self.pending_runs.front(), self.heap.peek()) {
            (Some(front), Some(peek)) => front.min_sort_key.as_slice() <= peek.sort_key.as_slice(),
            _ => false,
        }
    }

    /// Produces the next merged batch of up to `output_batch_size` rows in
    /// global sort order, or `Ok(None)` once every run (active and pending)
    /// is exhausted.
    ///
    /// Activates any pending runs whose `min_sort_key` has been reached by
    /// the current heap minimum before each pop, so overlapping runs are
    /// joined into the merge at the correct point. Empty run files
    /// encountered during activation are skipped silently.
    pub fn next_batch(&mut self) -> Result<Option<RecordBatch>, SortingParquetError> {
        if self.heap.is_empty() && self.pending_runs.is_empty() {
            return Ok(None);
        }

        // If heap is empty but pending runs exist, activate the next one
        // (and any others with the same min_sort_key)
        if self.heap.is_empty() {
            if let Some(run) = self.pending_runs.pop_front() {
                let key = run.min_sort_key.clone();
                Self::activate_run(
                    run,
                    &mut self.cursors,
                    &mut self.heap,
                    &self.row_converter,
                    &self.sorting_columns,
                    self.read_batch_size,
                    &mut self.next_cursor_idx,
                )?;
                // Activate any others with the same min
                while let Some(front) = self.pending_runs.front() {
                    if front.min_sort_key.as_slice() <= key.as_slice() {
                        let r = self.pending_runs.pop_front().unwrap();
                        Self::activate_run(
                            r,
                            &mut self.cursors,
                            &mut self.heap,
                            &self.row_converter,
                            &self.sorting_columns,
                            self.read_batch_size,
                            &mut self.next_cursor_idx,
                        )?;
                    } else {
                        break;
                    }
                }
            }
            if self.heap.is_empty() {
                return Ok(None);
            }
        }

        // Build a batch pool: snapshot each active cursor's current batch.
        // RecordBatch::clone is O(1) (Arc-backed arrays).
        let mut batch_pool: Vec<RecordBatch> = Vec::with_capacity(self.cursors.len());
        // Maps run_idx -> index into batch_pool for its current batch
        let mut pool_index: Vec<usize> = vec![usize::MAX; self.next_cursor_idx];

        for cursor in &self.cursors {
            if let Some(batch) = &cursor.current_batch {
                pool_index[cursor.run_idx] = batch_pool.len();
                batch_pool.push(batch.clone());
            }
        }

        // Collect up to output_batch_size indices as (pool_batch_idx, row_within_batch)
        let mut indices: Vec<(usize, usize)> = Vec::with_capacity(self.output_batch_size);

        while indices.len() < self.output_batch_size {
            // Activate pending runs BEFORE popping, so any run with
            // min_sort_key <= the current heap minimum participates in this pop.
            if !self.pending_runs.is_empty() {
                while self.should_activate_next_pending() {
                    let run = self.pending_runs.pop_front().unwrap();
                    Self::activate_run(
                        run,
                        &mut self.cursors,
                        &mut self.heap,
                        &self.row_converter,
                        &self.sorting_columns,
                        self.read_batch_size,
                        &mut self.next_cursor_idx,
                    )?;
                }
                // Add newly activated cursors' batches to the pool
                while pool_index.len() < self.next_cursor_idx {
                    let idx = pool_index.len();
                    pool_index.push(usize::MAX);
                    if let Some(batch) = &self.cursors[idx].current_batch {
                        pool_index[idx] = batch_pool.len();
                        batch_pool.push(batch.clone());
                    }
                }
            }

            let Some(mut entry) = self.heap.pop() else {
                break;
            };
            let run_idx = entry.run_idx;

            // Direct index: cursors[run_idx] is always valid because run_idx
            // is assigned sequentially as cursors are pushed.
            let row_idx = self.cursors[run_idx].row_idx;
            indices.push((pool_index[run_idx], row_idx));

            // Advance the cursor, reusing the popped entry's sort_key buffer
            let next_row_idx = row_idx + 1;

            if next_row_idx < self.cursors[run_idx].current_batch_num_rows() {
                self.cursors[run_idx].row_idx = next_row_idx;
                if self.cursors[run_idx].fill_sort_key(&mut entry.sort_key) {
                    self.heap.push(entry);
                }
            } else {
                // Current batch exhausted — try to load next batch
                if self.cursors[run_idx]
                    .advance_batch(&self.row_converter, &self.sorting_columns)?
                {
                    let new_batch = self.cursors[run_idx].current_batch.as_ref().unwrap();
                    pool_index[run_idx] = batch_pool.len();
                    batch_pool.push(new_batch.clone());

                    if self.cursors[run_idx].fill_sort_key(&mut entry.sort_key) {
                        self.heap.push(entry);
                    }
                }
                // If run exhausted, entry (and its Vec) is simply dropped
            }
        }

        if indices.is_empty() {
            return Ok(None);
        }

        // Compact: only pass referenced batches to interleave_record_batch.
        let mut used_pool_indices: Vec<usize> = indices.iter().map(|(bi, _)| *bi).collect();
        used_pool_indices.sort_unstable();
        used_pool_indices.dedup();

        let mut remap = vec![0usize; batch_pool.len()];
        let mut compacted: Vec<&RecordBatch> = Vec::with_capacity(used_pool_indices.len());
        for (new_idx, &old_idx) in used_pool_indices.iter().enumerate() {
            remap[old_idx] = new_idx;
            compacted.push(&batch_pool[old_idx]);
        }

        let remapped_indices: Vec<(usize, usize)> =
            indices.iter().map(|(bi, ri)| (remap[*bi], *ri)).collect();

        let output = interleave_record_batch(&compacted, &remapped_indices)?;
        Ok(Some(output))
    }
}

/// Streams batches from the merger via [`Iterator`].
///
/// Yields `Some(Ok(batch))` while data remains, `Some(Err(_))` if a merge
/// step fails, and `None` after all runs are exhausted. After an `Err` is
/// returned the merger should not be polled again.
impl Iterator for SortedRunMerger {
    type Item = Result<RecordBatch, SortingParquetError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_batch() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}
