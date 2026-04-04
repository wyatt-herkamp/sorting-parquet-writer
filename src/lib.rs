//! # Sorting Parquet Writer
//!
//! A library for writing sorted Parquet files with bounded memory usage,
//! inspired by [Parquet-Go's SortingWriter](https://pkg.go.dev/github.com/parquet-go/parquet-go#SortingWriter).
//!
//! ## Writers
//!
//! - [`writers::SortingParquetWriter`] — produces a **globally sorted** Parquet file
//!   using external merge sort. Data is buffered in memory, periodically sorted and
//!   spilled to temporary run files, then merged via streaming k-way merge at finalization.
//!
//! - [`writers::SortedGroupsParquetWriter`] — sorts **individual row groups** without
//!   guaranteeing global order. No temporary files needed.
//!
//! ## Sorting utilities
//!
//! - [`sorting::sort_record_batch`] — sort a single [`RecordBatch`](arrow::array::RecordBatch)
//!   by the given sorting columns.
//! - [`record_batch::merge_sorted_batches`] — k-way merge of pre-sorted batches into one.
//!
//! ## Progress tracking
//!
//! [`writers::SortingParquetWriter::finish_with_progress`] accepts any
//! [`writers::FinishProgressHandler`] (including closures) for monitoring the merge phase.
//!
//! ## Example
//!
//! ```rust,no_run
//! use sorting_parquet_writer::writers::SortingParquetWriter;
//! use parquet::file::properties::WriterProperties;
//! use parquet::file::metadata::SortingColumn;
//! use arrow::datatypes::{Schema, Field, DataType, SchemaRef};
//! use std::sync::Arc;
//!
//! let schema: SchemaRef = Arc::new(Schema::new(vec![
//!     Field::new("id", DataType::Int32, false),
//! ]));
//! let props = WriterProperties::builder()
//!     .set_sorting_columns(Some(vec![SortingColumn {
//!         column_idx: 0, descending: false, nulls_first: false,
//!     }]))
//!     .build();
//!
//! let file = std::fs::File::create("output.parquet").unwrap();
//! let mut writer = SortingParquetWriter::try_new(file, schema, props).unwrap();
//! // writer.write(&batch)?;
//! // let file = writer.finish()?;
//! ```

mod error;
pub use error::*;
pub mod record_batch;
pub mod sorting;
#[cfg(test)]
pub mod test;
mod utils;
pub mod writers;
