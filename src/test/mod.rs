use arrow::{
    array::RecordBatch,
    datatypes::{DataType, SchemaRef},
};
mod ticker;
pub use ticker::*;
pub mod generation;
pub mod random_time;
#[derive(thiserror::Error, Debug)]
pub enum TestError {
    #[error(transparent)]
    SortingParquetError(#[from] crate::SortingParquetError),
    #[error(transparent)]
    ArrowError(#[from] arrow::error::ArrowError),
    #[error(transparent)]
    IOError(#[from] std::io::Error),

    #[error("Failed to downcast array {from} to {to}")]
    CastError { from: DataType, to: &'static str },
    #[error("Chrono error: {0}")]
    ChronoError(&'static str),
}
pub trait TestArrowType {
    fn random_instances(n: usize) -> Vec<Self>
    where
        Self: Sized;
    fn sorting_columns() -> Vec<parquet::format::SortingColumn>
    where
        Self: Sized;

    fn schema() -> SchemaRef;

    fn into_record_batch(records: Vec<Self>) -> Result<RecordBatch, TestError>
    where
        Self: Sized;

    fn from_record_batch(batch: &RecordBatch) -> Result<Vec<Self>, TestError>
    where
        Self: Sized;

    fn is_sorted(records: &[Self]) -> Option<&[Self]>
    where
        Self: Sized;
}
