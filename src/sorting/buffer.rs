//! A row-count-bounded staging buffer used by
//! [`SortedGroupsParquetWriter`](crate::writers::SortedGroupsParquetWriter)
//! to collect pre-sorted batches into row-group-sized chunks before merging.

use std::mem;

use arrow::array::RecordBatch;

use crate::utils::split_batch;

/// A FIFO buffer of [`RecordBatch`]es that emits a batch of batches as soon as
/// the total row count reaches `maximum_rows_per_group`.
///
/// Batches are kept in insertion order and never reordered. The buffer is
/// agnostic to whether batches are sorted — that's the caller's responsibility.
/// Used by [`SortedGroupsParquetWriter`](crate::writers::SortedGroupsParquetWriter)
/// to assemble exactly-sized row groups whose constituent batches are
/// individually sorted and ready to be merged.
#[derive(Debug, Clone, PartialEq)]
pub struct SortingBuffer {
    buffer: Vec<RecordBatch>,
    num_rows: usize,
    maximum_rows_per_group: usize,
}
impl SortingBuffer {
    /// Creates an empty buffer that will emit a group every
    /// `maximum_rows_per_group` rows.
    pub fn new(maximum_rows_per_group: usize) -> Self {
        Self {
            buffer: Vec::new(),
            num_rows: 0,
            maximum_rows_per_group,
        }
    }

    /// Appends `batch` to the buffer.
    ///
    /// If the buffer's row count reaches `maximum_rows_per_group`, returns
    /// `Some(group)` where `group` contains the previously buffered batches
    /// followed by the prefix of `batch` needed to hit the limit exactly. Any
    /// excess rows in `batch` are split off (zero-copy via [`RecordBatch::slice`])
    /// and remain buffered to seed the next group.
    ///
    /// Returns `None` if the limit was not reached, in which case `batch` is
    /// retained in full.
    pub fn add_batch(&mut self, batch: RecordBatch) -> Option<Vec<RecordBatch>> {
        let new_total_rows = self.num_rows + batch.num_rows();
        if new_total_rows >= self.maximum_rows_per_group {
            let excess_rows = new_total_rows - self.maximum_rows_per_group;
            let rows_to_take = batch.num_rows() - excess_rows;
            let (new_batch, remaining_batch) = split_batch(&batch, rows_to_take);
            let remaining_batch_num_rows = remaining_batch.num_rows();
            let mut replaced = mem::replace(&mut self.buffer, vec![remaining_batch]);
            self.num_rows = remaining_batch_num_rows;
            replaced.push(new_batch);

            return Some(replaced);
        }
        self.buffer.push(batch);
        self.num_rows = new_total_rows;
        None
    }

    /// Drains the buffer, returning all currently held batches and resetting
    /// the row count to zero. Returns `None` if the buffer was empty.
    pub fn flush(&mut self) -> Option<Vec<RecordBatch>> {
        if self.buffer.is_empty() {
            None
        } else {
            let replaced = mem::take(&mut self.buffer);
            self.num_rows = 0;
            Some(replaced)
        }
    }
}
