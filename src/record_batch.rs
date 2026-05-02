//! In-memory k-way merge of pre-sorted [`RecordBatch`]es.
//!
//! Each batch in the input must already be sorted by the supplied
//! [`SortingColumn`]s. The merge uses a binary min-heap over the head row of
//! every non-empty batch and assembles the merged output in one pass via
//! [`arrow::compute::interleave_record_batch`].
//!
//! Variants:
//!
//! - [`merge_sorted_batches`] — convenience wrapper that builds a fresh
//!   [`RowConverter`] and validates schemas.
//! - [`merge_sorted_batches_with_row_converter`] — same, but reuses a
//!   pre-built converter.
//! - [`merge_sorted_batches_with_row_converter_unchecked`] — skips the
//!   schema-equality check; intended for internal callers that have already
//!   established the invariant (e.g. all batches came from the same writer).
//! - [`merge_sorted_batches_with_row_converter_returning_extremes`] — same as
//!   the unchecked variant but additionally returns the encoded min and max
//!   sort keys for use as a [`RunInfo`](streaming_merge::RunInfo) range.
//!
//! For the file-backed merge across spilled run files used by
//! [`SortingParquetWriter`](crate::writers::SortingParquetWriter), see the
//! [`streaming_merge`] submodule.

use arrow::array::{ArrayRef, RecordBatch};
use arrow::compute::interleave_record_batch;
use arrow_row::{RowConverter, SortField};
use parquet::file::metadata::SortingColumn;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::SortingParquetError;
use crate::sorting::SortExtremes;
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
/// Merges multiple already-sorted [`RecordBatch`]es into a single sorted
/// [`RecordBatch`] over the same sort key.
///
/// Builds a fresh [`RowConverter`] from the first batch's schema and
/// `sorting_columns`. Prefer
/// [`merge_sorted_batches_with_row_converter`] when merging repeatedly with
/// the same schema.
///
/// # Errors
///
/// Returns an error if `batches` is empty, if the batches do not all share a
/// schema, or if row encoding / interleaving fails.
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

/// Like [`merge_sorted_batches`], but reuses a caller-supplied
/// [`RowConverter`] instead of building one each call.
///
/// Validates that every batch has the same schema before delegating to
/// [`merge_sorted_batches_with_row_converter_unchecked`].
///
/// # Errors
///
/// Returns an error if `batches` is empty, the schemas don't match, or row
/// encoding / interleaving fails.
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

/// Merges sorted batches without checking that all schemas match.
///
/// The caller is responsible for ensuring every batch in `batches` has a
/// schema that matches `row_converter`'s expectations and that each batch is
/// already sorted by `sorting_columns`. Empty batches are skipped from the
/// heap.
///
/// Used internally by callers that have already established these invariants
/// — for example,
/// [`SortedGroupsParquetWriter`](crate::writers::SortedGroupsParquetWriter)
/// pre-sorts each batch and shares the writer's schema across all of them.
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

/// Merges sorted batches and additionally returns the encoded min and max
/// sort keys of the merged result.
///
/// Equivalent in merge logic to
/// [`merge_sorted_batches_with_row_converter_unchecked`]. The min/max are
/// taken from the first and last entries of the merge order, avoiding a
/// second [`RowConverter::convert_columns`] pass.
///
/// Used by
/// [`SortingParquetWriter`](crate::writers::SortingParquetWriter) when
/// `merge_sort_batches` is enabled, so each spilled run file can be tagged
/// with its sort-key range for the lazy-activation step in
/// [`SortedRunMerger`](streaming_merge::SortedRunMerger).
///
/// # Errors
///
/// Returns [`SortingParquetError::UnexpectedIndexOutOfBounds`] when every
/// input batch is empty (and so the merge produces no rows from which to
/// extract a min/max). Other errors are propagated from row encoding or
/// [`interleave_record_batch`].
pub fn merge_sorted_batches_with_row_converter_returning_extremes(
    batches: &[RecordBatch],
    sorting_columns: &[SortingColumn],
    row_converter: &RowConverter,
) -> Result<(RecordBatch, SortExtremes), SortingParquetError> {
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

    // First entry in merge_order = min, last = max
    let (min_batch, min_row) = *merge_order
        .first()
        .ok_or(SortingParquetError::UnexpectedIndexOutOfBounds)?;
    let (max_batch, max_row) = *merge_order
        .last()
        .ok_or(SortingParquetError::UnexpectedIndexOutOfBounds)?;
    let min_key = row_columns[min_batch].row(min_row).as_ref().to_vec();
    let max_key = row_columns[max_batch].row(max_row).as_ref().to_vec();

    let batch_refs: Vec<&RecordBatch> = batches.iter().collect();
    let merged = interleave_record_batch(&batch_refs, &merge_order)?;
    Ok((merged, (min_key, max_key)))
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
