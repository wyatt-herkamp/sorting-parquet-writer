use std::fs::File;

use arrow::{array::RecordBatch, datatypes::SchemaRef};
use parquet::{
    arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder},
    file::properties::WriterProperties,
};

use crate::{
    SortingParquetError,
    writers::SortedGroupsParquetWriter,
};
pub struct SortingParquetWriter {
    schema: SchemaRef,
    properties: WriterProperties,
    buffer: SortedGroupsParquetWriter,
    target: ArrowWriter<File>,
}
impl SortingParquetWriter {
    pub fn try_new(
        file: File,
        schema: SchemaRef,
        properties: WriterProperties,
    ) -> Result<Self, SortingParquetError> {
        let mut temp_file = tempfile::NamedTempFile::new()?;
        temp_file.disable_cleanup(true);
        let buffer = SortedGroupsParquetWriter::try_new(
            temp_file.into_file(),
            schema.clone(),
            properties.clone(),
        )?;
        let target = ArrowWriter::try_new(file, schema.clone(), Some(properties.clone()))?;
        Ok(Self {
            schema,
            properties,
            buffer,
            target,
        })
    }

    /// Writes a RecordBatch to the buffer for sorting and eventual merging
    pub fn write(&mut self, batch: &RecordBatch) -> Result<(), SortingParquetError> {
        self.buffer.write(batch)
    }


    pub fn finish(self) -> Result<(), SortingParquetError> {
        let buffer_file = self.buffer.into_inner_writer()?;

        let sorting_columns = self.properties
            .sorting_columns()
            .ok_or(SortingParquetError::NoSortingColumnsConfigured)?
            .clone();

        let parquet_reader = ParquetRecordBatchReaderBuilder::try_new(buffer_file)?
            .build()?;
            
        let mut all_batches = Vec::new();
        for batch_result in parquet_reader {
            let batch = batch_result?;
            all_batches.push(batch);
        }
        
        let mut target = self.target;

        let chunk_concat = arrow::compute::concat_batches(&self.schema, &all_batches)?;
        let chunk_sorted = crate::sorting::sort_record_batch(&chunk_concat, &sorting_columns)?;
        target.write(&chunk_sorted)?;
        target.close()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{
        array::{Int32Array, RecordBatch, StringArray},
        datatypes::{DataType, Field, Schema},
    };
    use parquet::{
        arrow::arrow_reader::{ArrowReaderBuilder, ArrowReaderOptions},
        file::properties::WriterProperties,
        format::SortingColumn,
    };
    use std::fs::File;
    use std::sync::Arc;

    fn create_test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]))
    }

    fn create_test_batch(ids: Vec<i32>, names: Vec<&str>) -> RecordBatch {
        let schema = create_test_schema();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap()
    }
    #[test]
    fn test_sorting_parquet_writer() {
        // Create test data
        let schema = create_test_schema();

        // Create sorting columns configuration
        let sorting_columns = vec![SortingColumn {
            column_idx: 0, // Sort by id column
            descending: false,
            nulls_first: false,
        }];

        // Create writer properties with sorting
        let properties = WriterProperties::builder()
            .set_max_row_group_size(2) // Small row groups to force multiple groups
            .set_sorting_columns(Some(sorting_columns))
            .build();

        // Create temporary output file
        let test_file = File::create("output.parquet").unwrap();
        // Create the sorting writer
        let mut writer =
            SortingParquetWriter::try_new(test_file, schema.clone(), properties).unwrap();

        let test_input = vec![
            (vec![3,1], vec!["c","a"]),
            (vec![4], vec!["d"]),
            (vec![2], vec!["b"]),
            (vec![5,0], vec!["e","z"]),
            (vec![6], vec!["f"]),
            (vec![8], vec!["h"]),
            (vec![7], vec!["g"]),
            (vec![9], vec!["i"]),
            (vec![10], vec!["j"]),
        ];
        for (ids, names) in test_input {
            let batch = create_test_batch(ids, names);
            writer.write(&batch).unwrap();
        }
        // Use the new row group merge logic
        writer.finish().unwrap();
        let test_file = File::open("output.parquet").unwrap();

        let mut parquet_reader = ArrowReaderBuilder::try_new_with_options(
            test_file,
            ArrowReaderOptions::new().with_schema(schema.clone()),
        )
        .unwrap()
        .build()
        .unwrap();

        let batch = parquet_reader.next().unwrap().unwrap();
        arrow::util::pretty::print_batches(std::slice::from_ref(&batch)).unwrap();

        let expected_ids: Vec<i32> = vec![
           0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        ];
        let expected_names: Vec<&str> = vec![
            "z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j"
        ];
        let actual_ids = batch.column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .iter()
            .flatten()
            .collect::<Vec<i32>>();
        let actual_names = batch.column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .flatten()
            .collect::<Vec<&str>>();
        assert_eq!(actual_ids, expected_ids, "IDs should be sorted");
        assert_eq!(actual_names, expected_names, "Names should be sorted according to ID order");
    }
}
