use std::sync::Arc;

use arrow::array::{ArrayRef, ListBuilder, RecordBatch};

use crate::SortingParquetError;

pub trait MergableDataType<T> {
    fn merge(
        col_idx: usize,
        merge_order: &[(usize, u32)],
        batches: &[RecordBatch],
        options: &T,
    ) -> Result<ArrayRef, SortingParquetError>;
}
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
                            .ok_or_else(|| SortingParquetError::ArrowError(
                                arrow::error::ArrowError::InvalidArgumentError(
                                    "Expected correct array type".to_string()
                                )
                            ))?;
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
                            .ok_or_else(|| SortingParquetError::ArrowError(
                                arrow::error::ArrowError::InvalidArgumentError(
                                    "Expected ListArray".to_string()
                                )
                            ))?;
                        let row_array = array.value(row_idx as usize);
                        let value_array = row_array
                            .as_any()
                            .downcast_ref::<$array>()
                            .ok_or_else(|| SortingParquetError::ArrowError(
                                arrow::error::ArrowError::InvalidArgumentError(
                                    "Expected value to be of correct type".to_string()
                                )
                            ))?;
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
                .unwrap();
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
                .unwrap();
            let row_array = array.value(row_idx as usize);
            let value_array = row_array
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();
            builder.values().append_array(value_array);
            builder.append(true);
        }
        Ok(Arc::new(builder.finish()))
    }
}
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
                            .downcast_ref::<$array>()
                            .unwrap();
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
                            .unwrap();
                        let row_array = array.value(row_idx as usize);
                        let value_array = row_array
                            .as_any()
                            .downcast_ref::<$array>()
                            .unwrap();
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
