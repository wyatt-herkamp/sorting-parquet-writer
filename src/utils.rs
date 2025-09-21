use arrow::array::RecordBatch;

pub fn split_batch(batch: &RecordBatch, at_row: usize) -> (RecordBatch, RecordBatch) {
    let total_rows = batch.num_rows();
    if total_rows <= at_row {
        return (batch.clone(), RecordBatch::new_empty(batch.schema()));
    }
    let left_batch = batch.slice(0, at_row);
    let right_batch = batch.slice(at_row, total_rows - at_row);
    (left_batch, right_batch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn test_split_batch() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let id_array = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));
        let name_array = Arc::new(StringArray::from(vec![
            "Alice", "Bob", "Charlie", "David", "Eve",
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, name_array]).unwrap();

        let (left, right) = split_batch(&batch, 3);
        assert_eq!(left.num_rows(), 3);
        assert_eq!(right.num_rows(), 2);

        let (left2, right2) = split_batch(&batch, 5);
        assert_eq!(left2.num_rows(), 5);
        assert_eq!(right2.num_rows(), 0);

        let (left3, right3) = split_batch(&batch, 10);
        assert_eq!(left3.num_rows(), 5);
        assert_eq!(right3.num_rows(), 0);
    }
}
