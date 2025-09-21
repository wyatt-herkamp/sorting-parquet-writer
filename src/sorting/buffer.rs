use std::mem;

use arrow::array::RecordBatch;

use crate::utils::split_batch;
#[derive(Debug, Clone, PartialEq)]
pub struct SortingBuffer {
    buffer: Vec<RecordBatch>,
    num_rows: usize,
    maximum_rows_per_group: usize,
}
impl SortingBuffer {
    /// Creates a new SortingBuffer with the specified maximum number of rows per group.
    pub fn new(maximum_rows_per_group: usize) -> Self {
        Self {
            buffer: Vec::new(),
            num_rows: 0,
            maximum_rows_per_group,
        }
    }
    /// Adds a RecordBatch to the buffer and updates the total number of rows.
    ///
    pub fn add_batch(&mut self, batch: RecordBatch) -> Option<Vec<RecordBatch>> {
        let new_total_rows = self.num_rows + batch.num_rows();
        if new_total_rows >= self.maximum_rows_per_group {
            let excess_rows = new_total_rows - self.maximum_rows_per_group;
            let (new_batch, remaining_batch) = split_batch(&batch, excess_rows);
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

    /// Flushes the current buffer, returning all buffered RecordBatches and clearing the buffer.
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
