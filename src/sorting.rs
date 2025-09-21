use arrow::array::ArrayRef;
use arrow::array::RecordBatch;
use arrow::compute::{SortOptions, take};
use arrow_row::{RowConverter, SortField};
use parquet::format::SortingColumn;
pub mod buffer;
pub mod row_group;
/// Sorts a RecordBatch based on the provided sorting columns.
///
/// Using [arrow_row](https://crates.io/crates/arrow_row) to convert rows into a comparable format,
pub fn sort_record_batch(
    batch: &RecordBatch,
    sorting_columns: &[SortingColumn],
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let schema = batch.schema();
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
    let row_converter = RowConverter::new(sort_fields)?;
    let columns: Vec<ArrayRef> = sorting_columns
        .iter()
        .map(|col| batch.column(col.column_idx as usize).clone())
        .collect();
    let rows = row_converter.convert_columns(&columns)?;
    let mut indices: Vec<usize> = (0..batch.num_rows()).collect();
    indices.sort_by(|&a, &b| {
        let row_a = rows.row(a);
        let row_b = rows.row(b);
        row_a.cmp(&row_b)
    });
    let indices_array = arrow::array::UInt32Array::from_iter(indices.iter().map(|&i| i as u32));
    // Use arrow::compute::take on each column to build the sorted batch
    let sorted_columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|i| take(batch.column(i).as_ref(), &indices_array, None))
        .collect::<Result<_, _>>()?;
    let sorted_batch = arrow::record_batch::RecordBatch::try_new(batch.schema(), sorted_columns)?;

    Ok(sorted_batch)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Int32Array, RecordBatch, StringArray},
        datatypes::{DataType, Field, Schema},
    };
    use parquet::format::SortingColumn;
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
