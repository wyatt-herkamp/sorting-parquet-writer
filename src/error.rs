use arrow::{datatypes::DataType, error::ArrowError};
use thiserror::Error;
#[derive(Debug, Error)]
pub enum SortingParquetError {
    #[error(transparent)]
    ArrowError(#[from] ArrowError),
    #[error(transparent)]
    ParquetError(#[from] parquet::errors::ParquetError),
    #[error("No Sorting Columns Configured")]
    NoSortingColumnsConfigured,

    #[error(transparent)]
    CastError(#[from] CastError),

    #[error("Unsupported DataType for Merging: {0}")]
    UnsupportedDataTypeForMerge(String),
}

#[derive(Debug, Error)]
#[error(
    "Failed to downcast Arrow Type from {from} to {to}. This is likely a bug in the sorting-parquet-writer crate."
)]
pub struct CastError {
    pub from: String,
    pub to: &'static str,
}
impl CastError {
    pub fn new<T>(from: &DataType) -> Self {
        Self {
            from: format!("{from}"),
            to: std::any::type_name::<T>(),
        }
    }
}
impl<'dt> From<(&'dt DataType, &'static str)> for CastError {
    fn from((dt, to): (&'dt DataType, &'static str)) -> Self {
        CastError {
            from: format!("{dt}"),
            to,
        }
    }
}
