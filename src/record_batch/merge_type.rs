use std::sync::Arc;

use arrow::array::{ArrayRef, ListBuilder, RecordBatch};

use crate::{CastError, SortingParquetError};
mod num;
mod timestamp;
pub use timestamp::*;
pub trait MergableDataType<T> {
    fn merge(
        col_idx: usize,
        merge_order: &[(usize, u32)],
        batches: &[RecordBatch],
        options: &T,
    ) -> Result<ArrayRef, SortingParquetError>;
}

impl MergableDataType<usize> for String {
    fn merge(
        col_idx: usize,
        merge_order: &[(usize, u32)],
        batches: &[RecordBatch],
        max_len: &usize,
    ) -> Result<ArrayRef, SortingParquetError> {
        let mut builder = arrow::array::StringBuilder::with_capacity(
            merge_order.len(),
            merge_order.len() * max_len,
        );
        for &(batch_idx, row_idx) in merge_order {
            let array = batches[batch_idx]
                .column(col_idx)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| {
                    CastError::new::<String>(batches[batch_idx].column(col_idx).data_type())
                })?;
            builder.append_value(array.value(row_idx as usize));
        }
        Ok(Arc::new(builder.finish()))
    }
}
impl MergableDataType<usize> for Vec<String> {
    fn merge(
        col_idx: usize,
        merge_order: &[(usize, u32)],
        batches: &[RecordBatch],
        max_len: &usize,
    ) -> Result<ArrayRef, SortingParquetError> {
        let mut builder = ListBuilder::new(arrow::array::StringBuilder::with_capacity(
            merge_order.len(),
            merge_order.len() * max_len,
        ));
        for &(batch_idx, row_idx) in merge_order {
            let array = batches[batch_idx]
                .column(col_idx)
                .as_any()
                .downcast_ref::<arrow::array::ListArray>()
                .ok_or_else(|| {
                    CastError::new::<arrow::array::ListArray>(
                        batches[batch_idx].column(col_idx).data_type(),
                    )
                })?;
            let row_array = array.value(row_idx as usize);
            let value_array = row_array
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| {
                    CastError::new::<arrow::array::StringArray>(row_array.data_type())
                })?;
            builder.values().append_array(value_array);
            builder.append(true);
        }
        Ok(Arc::new(builder.finish()))
    }
}
