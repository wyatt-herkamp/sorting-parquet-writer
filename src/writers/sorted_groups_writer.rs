use std::io::Write;

use arrow::{array::RecordBatch, datatypes::SchemaRef};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

use crate::{
    SortingParquetError, record_batch,
    sorting::{self, buffer::SortingBuffer},
};

/// Default maximum row group size when not explicitly configured (matches parquet 56 behavior).
const DEFAULT_MAX_ROW_GROUP_SIZE: usize = 1024 * 1024;

/// A Parquet Writer that sorts the individual Row Groups based on the provided sorting columns.
///
/// This will not result in a globally sorted Parquet File. But the individual Row Groups will be sorted.
///
/// This can create a Parquet file that is more efficient to read but it will not be as efficient as a fully sorted Parquet file.
pub struct SortedGroupsParquetWriter<W: Write + Send> {
    schema: SchemaRef,
    buffer: SortingBuffer,
    properties: WriterProperties,
    inner: ArrowWriter<W>,
    row_converter: arrow_row::RowConverter,
}
impl<W: Write + Send> SortedGroupsParquetWriter<W> {
    pub fn try_new(
        writer: W,
        schema: SchemaRef,
        properties: WriterProperties,
    ) -> Result<Self, SortingParquetError> {
        if properties.sorting_columns().is_none() {
            return Err(SortingParquetError::NoSortingColumnsConfigured);
        }
        let inner = ArrowWriter::try_new(writer, schema.clone(), Some(properties.clone()))?;
        let row_converter = sorting::create_row_converter(
            properties
                .sorting_columns()
                .ok_or(SortingParquetError::NoSortingColumnsConfigured)?,
            schema.as_ref(),
        )?;
        Ok(Self {
            schema,
            buffer: SortingBuffer::new(
                properties
                    .max_row_group_row_count()
                    .unwrap_or(DEFAULT_MAX_ROW_GROUP_SIZE),
            ),
            properties,
            inner,
            row_converter,
        })
    }
    fn sorting_columns(
        &self,
    ) -> Result<&Vec<parquet::file::metadata::SortingColumn>, SortingParquetError> {
        self.properties
            .sorting_columns()
            .ok_or(SortingParquetError::NoSortingColumnsConfigured)
    }
    /// Writes a RecordBatch to the Parquet file, sorting it based on the configured sorting columns.
    ///
    /// Each Batch will be sorted and then buffered until the configured maximum Row Group size is reached.
    /// At that point, the buffered batches will be merged and written as a single Row Group
    ///
    /// See: [ArrowWriter::write](parquet::arrow::ArrowWriter::write)
    ///
    pub fn write(&mut self, batch: &RecordBatch) -> Result<(), SortingParquetError> {
        if !batch.schema().as_ref().eq(self.schema.as_ref()) {
            return Err(SortingParquetError::ArrowError(
                arrow::error::ArrowError::InvalidArgumentError(
                    "Batch schema does not match writer schema".to_string(),
                ),
            ));
        }
        let sorted_batch = sorting::sort_record_batch_with_row_converter(
            batch,
            self.sorting_columns()?,
            &self.row_converter,
        )?;
        let results = self.buffer.add_batch(sorted_batch);
        if let Some(batches_to_write) = results {
            let sorted_batch = record_batch::merge_sorted_batches_with_row_converter_unchecked(
                &batches_to_write,
                self.sorting_columns()?,
                &self.row_converter,
            )?;
            self.inner.write(&sorted_batch)?;
        }

        Ok(())
    }
    /// Flushes any remaining buffered data to the Parquet file.
    ///
    /// This will result in a Row Group that may be smaller than the configured maximum size.
    ///
    /// See: [ArrowWriter::flush](parquet::arrow::ArrowWriter::flush)
    pub fn flush(&mut self) -> Result<(), SortingParquetError> {
        if let Some(batches_to_write) = self.buffer.flush() {
            let sorted_batch = record_batch::merge_sorted_batches_with_row_converter_unchecked(
                &batches_to_write,
                self.sorting_columns()?,
                &self.row_converter,
            )?;
            self.inner.write(&sorted_batch)?;
        }
        self.inner.flush()?;
        Ok(())
    }
    /// Flushes any remaining buffered data and closes the Parquet file.
    pub fn close(mut self) -> Result<(), SortingParquetError> {
        self.flush()?;
        self.inner.close()?;
        Ok(())
    }

    pub fn into_inner(mut self) -> Result<ArrowWriter<W>, SortingParquetError> {
        self.flush()?;
        Ok(self.inner)
    }

    pub fn into_inner_writer(self) -> Result<W, SortingParquetError> {
        Ok(self.into_inner()?.into_inner()?)
    }
    pub fn writer_properties(&self) -> &WriterProperties {
        &self.properties
    }
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }
}

#[cfg(test)]
mod tests {
    use crate::test::get_test_dir;

    use super::*;
    use arrow::{
        array::{Int32Array, RecordBatch, StringArray},
        datatypes::{DataType, Field, Schema},
    };
    use parquet::{
        arrow::arrow_reader::{ArrowReaderBuilder, ArrowReaderOptions},
        file::metadata::SortingColumn,
        file::{
            properties::WriterProperties,
            reader::{FileReader, SerializedFileReader},
        },
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
    fn small_row_groups() {
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
            .set_max_row_group_row_count(Some(2)) // Small row groups to force multiple groups
            .set_sorting_columns(Some(sorting_columns))
            .build();

        // Create temporary output file
        let test_file = File::create(get_test_dir().join("small_row_groups.parquet")).unwrap();
        // Create the sorting writer
        let mut writer =
            SortedGroupsParquetWriter::try_new(test_file, schema.clone(), properties).unwrap();

        let test_input = vec![
            (vec![3, 1, 4], vec!["c", "a", "d"]),
            (vec![2, 5], vec!["b", "e"]),
            (vec![6, 0], vec!["f", "z"]),
            (vec![8, 7, 9], vec!["h", "g", "i"]),
            (vec![10], vec!["j"]),
        ];
        for (ids, names) in test_input {
            let batch = create_test_batch(ids, names);
            writer.write(&batch).unwrap();
        }
        // Use the new row group merge logic
        writer.close().unwrap();
        {
            let test_file = File::open(get_test_dir().join("small_row_groups.parquet")).unwrap();

            let reader = SerializedFileReader::new(test_file).unwrap();
            assert_eq!(reader.num_row_groups(), 6, "Expected total of 6 row groups");
        }
        let test_file = File::open(get_test_dir().join("small_row_groups.parquet")).unwrap();
        let mut parquet_reader = ArrowReaderBuilder::try_new_with_options(
            test_file,
            ArrowReaderOptions::new().with_schema(schema.clone()),
        )
        .unwrap()
        .build()
        .unwrap();
        let record_batch_reader = parquet_reader.next().unwrap().unwrap();
        let expected_ids: Vec<i32> = vec![1, 3, 2, 4, 0, 5, 6, 7, 8, 9, 10];
        let expected_names: Vec<&str> = vec!["a", "c", "b", "d", "z", "e", "f", "g", "h", "i", "j"];
        let actual_ids = record_batch_reader
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .iter()
            .flat_map(|v| v)
            .collect::<Vec<i32>>();
        let actual_names = record_batch_reader
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .flat_map(|v| v)
            .collect::<Vec<&str>>();
        assert_eq!(actual_ids, expected_ids, "IDs should be sorted");
        assert_eq!(actual_names, expected_names, "Names should be sorted");
    }
}
