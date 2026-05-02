//! Single-batch sorting primitives built on top of the
//! [`arrow_row`](https://crates.io/crates/arrow_row) row format.
//!
//! The functions here all produce a permutation of the input batch's rows in
//! the order specified by a list of [`SortingColumn`]s. Variants that take a
//! pre-built [`RowConverter`] avoid rebuilding it on each call when the same
//! schema is sorted repeatedly (e.g. on every `write()` of a writer).
//!
//! The `_returning_extremes` variant additionally returns the encoded min and
//! max sort keys, which the writers use to populate
//! [`RunInfo`](crate::record_batch::streaming_merge::RunInfo) without a second
//! pass over the data.

use arrow::array::ArrayRef;
use arrow::array::RecordBatch;
use arrow::compute::{SortOptions, take};
use arrow_row::{RowConverter, SortField};
use parquet::file::metadata::SortingColumn;
pub mod buffer;
/// Sorts a [`RecordBatch`] by the given sorting columns and returns the
/// reordered batch.
///
/// Convenience wrapper over [`create_row_converter`] +
/// [`sort_record_batch_with_row_converter`]. Prefer the explicit pair when
/// sorting many batches with the same schema, since
/// [`RowConverter::new`](arrow_row::RowConverter::new) is not free.
pub fn sort_record_batch(
    batch: &RecordBatch,
    sorting_columns: &[SortingColumn],
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let schema = batch.schema();
    let row_converter = create_row_converter(sorting_columns, schema.as_ref())?;
    sort_record_batch_with_row_converter(batch, sorting_columns, &row_converter)
}

/// Builds a [`RowConverter`] for the given sorting columns against `schema`.
///
/// The returned converter encodes a row of the sort columns into a comparable
/// byte sequence honoring each column's `descending` and `nulls_first` flags.
/// Reuse the same converter across many batches that share the schema and
/// sorting column set.
pub fn create_row_converter(
    sorting_columns: &[SortingColumn],
    schema: &arrow::datatypes::Schema,
) -> Result<RowConverter, arrow::error::ArrowError> {
    let mut sort_fields = Vec::with_capacity(sorting_columns.len());
    for col in sorting_columns {
        let field = schema.field(col.column_idx as usize);
        let sort_field = SortField::new_with_options(
            field.data_type().clone(),
            SortOptions {
                descending: col.descending,
                nulls_first: col.nulls_first,
            },
        );
        sort_fields.push(sort_field);
    }
    RowConverter::new(sort_fields)
}
/// Sorts a [`RecordBatch`] using a caller-supplied [`RowConverter`].
///
/// The converter must have been built from the same `sorting_columns` and a
/// schema compatible with `batch`; otherwise [`RowConverter::convert_columns`]
/// will return an error.
pub fn sort_record_batch_with_row_converter(
    batch: &RecordBatch,
    sorting_columns: &[SortingColumn],
    row_converter: &RowConverter,
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let (batch, _) = sort_record_batch_with_row_converter_returning_extremes(
        batch,
        sorting_columns,
        row_converter,
    )?;
    Ok(batch)
}

/// `(min_sort_key, max_sort_key)` as encoded by [`RowConverter`].
///
/// Both values are owned byte vectors comparable lexicographically; together
/// they describe the sort-key range of a sorted batch and are stored on each
/// [`RunInfo`](crate::record_batch::streaming_merge::RunInfo) for use by the
/// streaming merger.
pub type SortExtremes = (Vec<u8>, Vec<u8>);

/// Sorts a [`RecordBatch`] and returns it together with the encoded min and
/// max sort keys.
///
/// Avoids a second pass over the sort columns compared to calling
/// [`sort_record_batch_with_row_converter`] and then re-encoding the first and
/// last rows separately.
///
/// # Panics
///
/// Panics if `batch.num_rows() == 0`. The writers never call this on an empty
/// buffer.
pub fn sort_record_batch_with_row_converter_returning_extremes(
    batch: &RecordBatch,
    sorting_columns: &[SortingColumn],
    row_converter: &RowConverter,
) -> Result<(RecordBatch, SortExtremes), arrow::error::ArrowError> {
    let columns: Vec<ArrayRef> = sorting_columns
        .iter()
        .map(|col| batch.column(col.column_idx as usize).clone())
        .collect();
    let rows = row_converter.convert_columns(&columns)?;
    let mut indices: Vec<u32> = (0..batch.num_rows() as u32).collect();
    // SAFETY: every value in `indices` is in `0..batch.num_rows()`, which
    // matches the row count of `rows`, so each `row_unchecked` index is in
    // bounds.
    indices.sort_unstable_by(|&a, &b| unsafe {
        let row_a = rows.row_unchecked(a as usize);
        let row_b = rows.row_unchecked(b as usize);
        row_a.cmp(&row_b)
    });

    // Extract min/max sort keys from the sorted indices (first = min, last = max).
    // SAFETY: `indices` is non-empty by the documented precondition, and every
    // entry is a valid row index for `rows` as established above.
    let min_key = unsafe { rows.row_unchecked(indices[0] as usize) }
        .as_ref()
        .to_vec();
    let max_key = unsafe { rows.row_unchecked(*indices.last().unwrap() as usize) }
        .as_ref()
        .to_vec();

    let indices_array = arrow::array::UInt32Array::from_iter(indices);
    let sorted_columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|i| take(batch.column(i).as_ref(), &indices_array, None))
        .collect::<Result<_, _>>()?;
    let sorted_batch = arrow::record_batch::RecordBatch::try_new(batch.schema(), sorted_columns)?;

    Ok((sorted_batch, (min_key, max_key)))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Int32Array, RecordBatch, StringArray},
        datatypes::{DataType, Field, Schema},
    };
    use parquet::file::metadata::SortingColumn;
    #[test]
    fn test_sort_record_batch() {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("a", DataType::Int32, false),
                Field::new("b", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(Int32Array::from(vec![3, 1, 2, 2])),
                Arc::new(StringArray::from(vec!["c", "a", "b", "b"])),
            ],
        )
        .unwrap();
        let sorting_columns = vec![
            SortingColumn {
                column_idx: 0,
                descending: true,
                nulls_first: false,
            },
            SortingColumn {
                column_idx: 1,
                descending: true,
                nulls_first: false,
            },
        ];
        let sorted_batch = super::sort_record_batch(&batch, &sorting_columns).unwrap();
        // Print the sorted batch for verification
        println!("{:?}", sorted_batch);
    }
}
