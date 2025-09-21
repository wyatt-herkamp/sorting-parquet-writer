use super::MergableDataType;
use std::sync::Arc;

use arrow::array::{ArrayRef, ListBuilder, RecordBatch};

use crate::{CastError, SortingParquetError};
macro_rules! simple_type {
    (
        $(
            $type:ty => {
                builder: $builder:ty,
                array: $array:ty
            }
        ),*
    ) => {
        $(
            impl MergableDataType<()> for $type {
                fn merge(
                    col_idx: usize,
                    merge_order: &[(usize, u32)],
                    batches: &[RecordBatch],
                    _options: &(),
                ) -> Result<ArrayRef, SortingParquetError> {
                    let mut builder = <$builder>::with_capacity(merge_order.len());
                    for &(batch_idx, row_idx) in merge_order {
                        let array = batches[batch_idx]
                            .column(col_idx)
                            .as_any()
                            .downcast_ref::<$array>()
                            .ok_or_else(||{
                                CastError::new::<$array>(&batches[batch_idx].column(col_idx).data_type())
                            })?;
                        builder.append_value(array.value(row_idx as usize));
                    }
                    Ok(Arc::new(builder.finish()))
                }
            }
            impl MergableDataType<()> for Vec<$type>{
                fn merge(
                    col_idx: usize,
                    merge_order: &[(usize, u32)],
                    batches: &[RecordBatch],
                    _options: &(),
                ) -> Result<ArrayRef, SortingParquetError> {
                    let mut builder = ListBuilder::new(<$builder>::with_capacity(merge_order.len()));
                    for &(batch_idx, row_idx) in merge_order {
                        let array = batches[batch_idx]
                            .column(col_idx)
                            .as_any()
                            .downcast_ref::<arrow::array::ListArray>()
                            .ok_or_else(||{
                                CastError::new::<arrow::array::ListArray>(&batches[batch_idx].column(col_idx).data_type())
                            })?;
                        let row_array = array.value(row_idx as usize);
                        let value_array = row_array
                            .as_any()
                            .downcast_ref::<$array>()
                            .ok_or_else(||{
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
simple_type! {
    i8 => { builder: arrow::array::Int8Builder, array: arrow::array::Int8Array },
    i16 => { builder: arrow::array::Int16Builder, array: arrow::array::Int16Array },
    i32 => { builder: arrow::array::Int32Builder, array: arrow::array::Int32Array },
    i64 => { builder: arrow::array::Int64Builder, array: arrow::array::Int64Array },
    u8 => { builder: arrow::array::UInt8Builder, array: arrow::array::UInt8Array },
    u16 => { builder: arrow::array::UInt16Builder, array: arrow::array::UInt16Array },
    u32 => { builder: arrow::array::UInt32Builder, array: arrow::array::UInt32Array },
    u64 => { builder: arrow::array::UInt64Builder, array: arrow::array::UInt64Array },
    f32 => { builder: arrow::array::Float32Builder, array: arrow::array::Float32Array },
    f64 => { builder: arrow::array::Float64Builder, array: arrow::array::Float64Array },
    bool => { builder: arrow::array::BooleanBuilder, array: arrow::array::BooleanArray }
}
