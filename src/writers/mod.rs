//! Parquet writers that produce sorted output.
//!
//! Three writers are provided:
//!
//! - [`SortingParquetWriter`] — produces a **globally sorted** Parquet file
//!   via external merge sort with bounded memory. Buffers, spills sorted
//!   runs to disk, and merges them in [`SortingParquetWriter::finish`].
//! - [`ConcurrentSortingParquetWriter`] — the `async` counterpart of
//!   `SortingParquetWriter`. Offloads each spill to a background Tokio task so
//!   buffering can overlap with sorting/writing. Requires a Tokio runtime; see
//!   [`BackgroundWorker`] to bound spill concurrency.
//! - [`SortedGroupsParquetWriter`] — sorts **each row group** independently
//!   without temporary files. Cheaper to run, but does not give a global sort.
//!
//! Progress tracking for the merge phase of `SortingParquetWriter` is
//! exposed via [`FinishProgressHandler`] / [`FinishProgress`] /
//! [`FinishPhase`]; see
//! [`SortingParquetWriter::finish_with_progress`].

mod progress;
pub use progress::*;
mod sorted_groups_writer;
pub use sorted_groups_writer::*;
mod sorting_writer;
pub use sorting_writer::*;
mod concurrent_sorting_writer;
pub use concurrent_sorting_writer::*;
