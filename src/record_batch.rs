use arrow::array::{ArrayRef, RecordBatch};
use arrow::datatypes::{DataType, Field};
use arrow_row::{RowConverter, SortField};
use parquet::format::SortingColumn;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::SortingParquetError;
use crate::record_batch::build::{
    MergableDataType, TimestampMicrosecond, TimestampMillisecond, TimestampNanosecond,
    TimestampSecond,
};
mod build;
#[derive(Eq)]
struct HeapItem<'a> {
    batch_idx: usize,
    row_idx: usize,
    row: arrow_row::Row<'a>,
}
impl<'a> Ord for HeapItem<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.row.cmp(&self.row)
    }
}
impl<'a> PartialOrd for HeapItem<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<'a> PartialEq for HeapItem<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row
    }
}
///  Merges multiple sorted RecordBatches into a single sorted RecordBatch based on the specified sorting columns.
pub fn merge_sorted_batches(
    batches: &[RecordBatch],
    sorting_columns: &[SortingColumn],
) -> Result<RecordBatch, SortingParquetError> {
    if batches.is_empty() {
        return Err(arrow::error::ArrowError::InvalidArgumentError(
            "No batches to merge".to_string(),
        )
        .into());
    }
    let schema = batches[0].schema();
    for batch in batches {
        if !batch.schema().as_ref().eq(schema.as_ref()) {
            return Err(arrow::error::ArrowError::InvalidArgumentError(
                "All batches must have the same schema".to_string(),
            )
            .into());
        }
    }

    // Prepare row converters for comparison
    let mut sort_fields = Vec::with_capacity(sorting_columns.len());
    for col in sorting_columns {
        let field = schema.field(col.column_idx as usize);
        let sort_field = SortField::new_with_options(
            field.data_type().clone(),
            arrow::compute::SortOptions {
                descending: col.descending,
                nulls_first: col.nulls_first,
            },
        );
        sort_fields.push(sort_field);
    }
    let row_converter = RowConverter::new(sort_fields)?;

    // For each batch, convert the sort columns to rows
    let mut row_columns = Vec::with_capacity(batches.len());
    let mut total_rows = 0;
    for batch in batches {
        let cols: Vec<ArrayRef> = sorting_columns
            .iter()
            .map(|col| batch.column(col.column_idx as usize).clone())
            .collect();
        let rows = row_converter.convert_columns(&cols)?;
        total_rows += batch.num_rows();
        row_columns.push(rows);
    }

    let mut heap = BinaryHeap::with_capacity(total_rows);
    let mut total_rows = 0;
    for (batch_idx, batch) in batches.iter().enumerate() {
        if batch.num_rows() > 0 {
            heap.push(HeapItem {
                batch_idx,
                row_idx: 0,
                row: row_columns[batch_idx].row(0),
            });
            total_rows += batch.num_rows();
        }
    }

    let mut merge_order: Vec<(usize, u32)> = Vec::with_capacity(total_rows);

    while let Some(HeapItem {
        batch_idx, row_idx, ..
    }) = heap.pop()
    {
        merge_order.push((batch_idx, row_idx as u32));
        let next_row_idx = row_idx + 1;
        if next_row_idx < batches[batch_idx].num_rows() {
            heap.push(HeapItem {
                batch_idx,
                row_idx: next_row_idx,
                row: row_columns[batch_idx].row(next_row_idx),
            });
        }
    }
    // Build final columns by taking rows in merge order
    let mut final_columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    for col_idx in 0..schema.fields().len() {
        // Build arrays based on data type
        let field = schema.field(col_idx);
        let array = from_field(field, col_idx, &merge_order, batches)?;
        final_columns.push(array);
    }
    let merged = RecordBatch::try_new(schema, final_columns)?;
    Ok(merged)
}
fn from_field(
    field: &Field,
    col_idx: usize,
    merge_order: &[(usize, u32)],
    batches: &[RecordBatch],
) -> Result<ArrayRef, SortingParquetError> {
    match field.data_type() {
        DataType::Boolean => {
            return <bool as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::Int8 => {
            return <i8 as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::Int16 => {
            return <i16 as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::Int32 => {
            return <i32 as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::Int64 => {
            return <i64 as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::UInt8 => {
            return <u8 as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::UInt16 => {
            return <u16 as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::UInt32 => {
            return <u32 as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::UInt64 => {
            return <u64 as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::Float32 => {
            return <f32 as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::Float64 => {
            return <f64 as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::Utf8 => {
            return <String as MergableDataType<usize>>::merge(col_idx, &merge_order, batches, &10);
        }
        DataType::Timestamp(unit, tz) => {
            return match unit {
                arrow::datatypes::TimeUnit::Second => {
                    TimestampSecond::merge(col_idx, &merge_order, batches, tz)
                }
                arrow::datatypes::TimeUnit::Millisecond => {
                    TimestampMillisecond::merge(col_idx, &merge_order, batches, tz)
                }
                arrow::datatypes::TimeUnit::Microsecond => {
                    TimestampMicrosecond::merge(col_idx, &merge_order, batches, tz)
                }
                arrow::datatypes::TimeUnit::Nanosecond => {
                    TimestampNanosecond::merge(col_idx, &merge_order, batches, tz)
                }
            };
        }
        DataType::List(data_type) => {
            from_list_field(data_type.as_ref(), col_idx, merge_order, batches)
        }
        dt => {
            return Err(arrow::error::ArrowError::InvalidArgumentError(format!(
                "Unsupported data type for merge: {:?}",
                dt
            ))
            .into());
        }
    }
}

fn from_list_field(
    field: &Field,
    col_idx: usize,
    merge_order: &[(usize, u32)],
    batches: &[RecordBatch],
) -> Result<ArrayRef, SortingParquetError> {
    match field.data_type() {
        DataType::Boolean => {
            return <Vec<bool> as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::Int8 => {
            return <Vec<i8> as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::Int16 => {
            return <Vec<i16> as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::Int32 => {
            return <Vec<i32> as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::Int64 => {
            return <Vec<i64> as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::UInt8 => {
            return <Vec<u8> as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::UInt16 => {
            return <Vec<u16> as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::UInt32 => {
            return <Vec<u32> as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::UInt64 => {
            return <Vec<u64> as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::Float32 => {
            return <Vec<f32> as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::Float64 => {
            return <Vec<f64> as MergableDataType<()>>::merge(col_idx, &merge_order, batches, &());
        }
        DataType::Utf8 => {
            return <Vec<String> as MergableDataType<usize>>::merge(col_idx, &merge_order, batches, &10);
        }
        DataType::Timestamp(unit, tz) => {
            return match unit {
                arrow::datatypes::TimeUnit::Second => {
                    Vec::<TimestampSecond>::merge(col_idx, &merge_order, batches, tz)
                }
                arrow::datatypes::TimeUnit::Millisecond => {
                    Vec::<TimestampMillisecond>::merge(col_idx, &merge_order, batches, tz)
                }
                arrow::datatypes::TimeUnit::Microsecond => {
                    Vec::<TimestampMicrosecond>::merge(col_idx, &merge_order, batches, tz)
                }
                arrow::datatypes::TimeUnit::Nanosecond => {
                    Vec::<TimestampNanosecond>::merge(col_idx, &merge_order, batches, tz)
                }
            };
        }
        DataType::List(data_type) => {
            from_list_field(data_type.as_ref(), col_idx, merge_order, batches)
        }
        dt => {
            return Err(arrow::error::ArrowError::InvalidArgumentError(format!(
                "Unsupported data type for merge: {:?}",
                dt
            ))
            .into());
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{
        array::{Int32Array, RecordBatch, StringArray},
        datatypes::{DataType, Field, Schema},
    };
    use parquet::format::SortingColumn;
    use std::sync::Arc;

    #[test]
    fn test_merge_sorted_batches() {
        // Create two sorted batches
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
        ]));
        // Batch 1: a: [1, 3], b: ["a", "c"]
        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 3])),
                Arc::new(StringArray::from(vec!["a", "c"])),
            ],
        )
        .unwrap();
        // Batch 2: a: [2, 4], b: ["b", "d"]
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![2, 4])),
                Arc::new(StringArray::from(vec!["b", "d"])),
            ],
        )
        .unwrap();
        // Both batches are sorted by column 0 ascending
        let sorting_columns = vec![SortingColumn {
            column_idx: 0,
            descending: false,
            nulls_first: false,
        }];
        let merged = merge_sorted_batches(&[batch1, batch2], &sorting_columns).unwrap();
        arrow::util::pretty::print_batches(&[merged.clone()]).unwrap();
        let a = merged
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let b = merged
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        // Should be [1, 2, 3, 4] and ["a", "b", "c", "d"]
        assert_eq!(a.values(), &[1, 2, 3, 4]);
        assert_eq!(b.value(0), "a");
        assert_eq!(b.value(1), "b");
        assert_eq!(b.value(2), "c");
        assert_eq!(b.value(3), "d");
    }
}
