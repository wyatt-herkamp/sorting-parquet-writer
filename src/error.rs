//! Error type returned by the sorting writers and merge utilities.

use std::io;

use arrow::error::ArrowError;
use parquet::errors::ParquetError;
use thiserror::Error;

/// The unified error type produced by this crate.
///
/// Wraps the three I/O / format error sources used during sorting and writing
/// ([`ArrowError`], [`ParquetError`], [`std::io::Error`]) and adds two
/// crate-specific variants for misconfiguration and an internal invariant
/// violation.
#[derive(Debug, Error)]
pub enum SortingParquetError {
    /// An error originating from the Arrow compute / array layer.
    #[error(transparent)]
    ArrowError(#[from] ArrowError),

    /// An error originating from the Parquet reader/writer layer.
    #[error(transparent)]
    ParquetError(#[from] ParquetError),

    /// Returned by `try_new` constructors when the supplied
    /// [`WriterProperties`](parquet::file::properties::WriterProperties) does
    /// not have sorting columns configured via
    /// [`set_sorting_columns`](parquet::file::properties::WriterPropertiesBuilder::set_sorting_columns).
    #[error("No Sorting Columns Configured")]
    NoSortingColumnsConfigured,

    /// Returned when a writer operation requires the internal row converter
    /// but the writer has already consumed it as part of a finalize step.
    #[error("Writer is already closed")]
    WriterClosed,

    /// An error reading from or writing to a file (run files or the target).
    #[error(transparent)]
    IoError(#[from] io::Error),

    /// An internal invariant was violated while computing min/max sort keys
    /// for a merged batch.
    ///
    /// Currently surfaces only from
    /// [`merge_sorted_batches_with_row_converter_returning_extremes`](crate::record_batch::merge_sorted_batches_with_row_converter_returning_extremes)
    /// when every input batch is empty, so the merge has no rows from which
    /// to extract a min or max.
    #[error("Unexpected index out of bounds during sorting")]
    UnexpectedIndexOutOfBounds,
}
