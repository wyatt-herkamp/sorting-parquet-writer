use arrow::array::{ArrayRef, RecordBatch};
use arrow::compute::interleave_record_batch;
use arrow_row::{RowConverter, SortField};
use parquet::file::metadata::SortingColumn;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::SortingParquetError;
pub mod streaming_merge;
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

    merge_sorted_batches_with_row_converter(batches, sorting_columns, &row_converter)
}
pub fn merge_sorted_batches_with_row_converter(
    batches: &[RecordBatch],
    sorting_columns: &[SortingColumn],
    row_converter: &RowConverter,
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

    merge_sorted_batches_with_row_converter_unchecked(batches, sorting_columns, row_converter)
}
pub fn merge_sorted_batches_with_row_converter_unchecked(
    batches: &[RecordBatch],
    sorting_columns: &[SortingColumn],
    row_converter: &RowConverter,
) -> Result<RecordBatch, SortingParquetError> {
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

    let mut heap = BinaryHeap::with_capacity(batches.len());
    for (batch_idx, batch) in batches.iter().enumerate() {
        if batch.num_rows() > 0 {
            heap.push(HeapItem {
                batch_idx,
                row_idx: 0,
                row: row_columns[batch_idx].row(0),
            });
        }
    }

    let mut merge_order: Vec<(usize, usize)> = Vec::with_capacity(total_rows);

    while let Some(HeapItem {
        batch_idx, row_idx, ..
    }) = heap.pop()
    {
        merge_order.push((batch_idx, row_idx));
        let next_row_idx = row_idx + 1;
        if next_row_idx < batches[batch_idx].num_rows() {
            heap.push(HeapItem {
                batch_idx,
                row_idx: next_row_idx,
                row: row_columns[batch_idx].row(next_row_idx),
            });
        }
    }

    let batch_refs: Vec<&RecordBatch> = batches.iter().collect();
    let merged = interleave_record_batch(&batch_refs, &merge_order)?;
    Ok(merged)
}
#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{
        array::{Int32Array, RecordBatch, StringArray, record_batch},
        datatypes::{DataType, Field, Schema},
    };
    use parquet::file::metadata::SortingColumn;
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
        arrow::util::pretty::print_batches(std::slice::from_ref(&merged)).unwrap();
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

    #[test]
    fn test_different_sizes() {
        let batch_a =
            record_batch!(("id", Int32, [1, 3]), ("name", Utf8, ["wyatt", "evan"])).unwrap();

        let batch_b = record_batch!(
            ("id", Int32, [2, 4, 6]),
            ("name", Utf8, ["alice", "bob", "david"])
        )
        .unwrap();

        let sorting_columns = vec![SortingColumn {
            column_idx: 0,
            descending: false,
            nulls_first: false,
        }];
        let merged = merge_sorted_batches(&[batch_a, batch_b], &sorting_columns).unwrap();
        arrow::util::pretty::print_batches(std::slice::from_ref(&merged)).unwrap();

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

        assert_eq!(a.values(), &[1, 2, 3, 4, 6]);
        assert_eq!(b.value(0), "wyatt");
        assert_eq!(b.value(1), "alice");
        assert_eq!(b.value(2), "evan");
        assert_eq!(b.value(3), "bob");
        assert_eq!(b.value(4), "david");
    }

    #[test]
    fn merge_with_null() {
        let batch_a = record_batch!(
            ("id", Int32, [1, 3, 5]),
            ("name", Utf8, [Some("wyatt"), Some("evan"), None])
        )
        .unwrap();

        let batch_b = record_batch!(
            ("id", Int32, [2, 4, 6]),
            ("name", Utf8, ["alice", "bob", "david"])
        )
        .unwrap();

        let sorting_columns = vec![SortingColumn {
            column_idx: 0,
            descending: false,
            nulls_first: false,
        }];
        let merged = merge_sorted_batches(&[batch_a, batch_b], &sorting_columns).unwrap();
        arrow::util::pretty::print_batches(std::slice::from_ref(&merged)).unwrap();

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

        assert_eq!(a.values(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(b.value(0), "wyatt");
        assert_eq!(b.value(1), "alice");
        assert_eq!(b.value(2), "evan");
        assert_eq!(b.value(3), "bob");
        assert!(b.value(4).is_empty());
        assert_eq!(b.value(5), "david");
    }
}
