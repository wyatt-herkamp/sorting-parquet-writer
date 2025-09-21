use arrow::error::ArrowError;
use thiserror::Error;
#[derive(Debug, Error)]
pub enum SortingParquetError {
    #[error(transparent)]
    ArrowError(#[from] ArrowError),
    #[error(transparent)]
    ParquetError(#[from] parquet::errors::ParquetError),
    #[error("No Sorting Columns Configured")]
    NoSortingColumnsConfigured,
}
