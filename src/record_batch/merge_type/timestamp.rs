use std::sync::Arc;

use arrow::array::{ArrayRef, ListBuilder, RecordBatch};

use super::MergableDataType;
use crate::{CastError, SortingParquetError};
macro_rules! timestamp {
    (
        $(
            $timestamp:ident => {
                builder: $builder:ty,
                array: $array:ty
            }
        ),*
    ) => {
        $(
            pub struct $timestamp;
            impl MergableDataType<Option<Arc<str>>> for $timestamp {
                fn merge(
                    col_idx: usize,
                    merge_order: &[(usize, u32)],
                    batches: &[RecordBatch],
                    options: &Option<Arc<str>>,
                ) -> Result<ArrayRef, SortingParquetError> {
                    let mut builder = <$builder>::with_capacity(merge_order.len())
                    .with_timezone_opt(options.clone());
                    for &(batch_idx, row_idx) in merge_order {
                        let array = batches[batch_idx]
                            .column(col_idx)
                            .as_any()
                            .downcast_ref::<$array>().ok_or_else(|| {
                                CastError::new::<$array>(&batches[batch_idx].column(col_idx).data_type())
                            })?;
                        builder.append_value(array.value(row_idx as usize));
                    }
                    Ok(Arc::new(builder.finish()))
                }
            }
            impl MergableDataType<Option<Arc<str>>> for Vec<$timestamp>{
                fn merge(
                    col_idx: usize,
                    merge_order: &[(usize, u32)],
                    batches: &[RecordBatch],
                    options: &Option<Arc<str>>,
                ) -> Result<ArrayRef, SortingParquetError> {
                    let mut builder = ListBuilder::new(<$builder>::with_capacity(merge_order.len())
                        .with_timezone_opt(options.clone()));
                    for &(batch_idx, row_idx) in merge_order {
                        let array = batches[batch_idx]
                            .column(col_idx)
                            .as_any()
                            .downcast_ref::<arrow::array::ListArray>()
                            .ok_or_else(|| {
                                CastError::new::<arrow::array::ListArray>(&batches[batch_idx].column(col_idx).data_type())
                            })?;
                        let row_array = array.value(row_idx as usize);
                        let value_array = row_array
                            .as_any()
                            .downcast_ref::<$array>()
                            .ok_or_else(|| {
                                CastError::new::<$array>(&row_array.data_type())
                            })?;
                        builder.values().append_array(value_array);
                        builder.append(true);
                    }
                    Ok(Arc::new(builder.finish()))
                }
            }
        )*
    };
}
timestamp! {
    TimestampSecond => { builder: arrow::array::TimestampSecondBuilder, array: arrow::array::TimestampSecondArray },
    TimestampNanosecond => { builder: arrow::array::TimestampNanosecondBuilder, array: arrow::array::TimestampNanosecondArray },
    TimestampMillisecond => { builder: arrow::array::TimestampMillisecondBuilder, array: arrow::array::TimestampMillisecondArray },
    TimestampMicrosecond => { builder: arrow::array::TimestampMicrosecondBuilder, array: arrow::array::TimestampMicrosecondArray }
}
