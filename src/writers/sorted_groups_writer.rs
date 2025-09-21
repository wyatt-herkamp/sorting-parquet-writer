use std::fs::File;

use arrow::{array::RecordBatch, datatypes::SchemaRef};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

use crate::{
    SortingParquetError, record_batch,
    sorting::{self, buffer::SortingBuffer},
};
/// A Parquet Writer that sorts the individual Row Groups based on the provided sorting columns.
///
/// This will not result in a globally sorted Parquet File.
pub struct SortedGroupsParquetWriter {
    schema: SchemaRef,
    buffer: SortingBuffer,
    properties: WriterProperties,
    inner: ArrowWriter<File>,
}
impl SortedGroupsParquetWriter {
    pub fn try_new(
        file: File,
        schema: SchemaRef,
        properties: WriterProperties,
    ) -> Result<Self, SortingParquetError> {
        if properties.sorting_columns().is_none() {
            return Err(SortingParquetError::NoSortingColumnsConfigured);
        }
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(properties.clone()))?;
        Ok(Self {
            schema,
            buffer: SortingBuffer::new(properties.max_row_group_size()),
            properties,
            inner: writer,
        })
    }
    fn sorting_columns(&self) -> Result<&Vec<parquet::format::SortingColumn>, SortingParquetError> {
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
        let results = self
            .buffer
            .add_batch(sorting::sort_record_batch(batch, self.sorting_columns()?)?);
        if let Some(batches_to_write) = results {
            let sorted_batch =
                record_batch::merge_sorted_batches(&batches_to_write, self.sorting_columns()?)?;
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
            let sorted_batch =
                record_batch::merge_sorted_batches(&batches_to_write, self.sorting_columns()?)?;
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
}
