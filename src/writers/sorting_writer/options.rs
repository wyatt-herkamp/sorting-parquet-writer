use std::path::{Path, PathBuf};

use parquet::file::properties::WriterProperties;

use crate::writers::{FlushThreshold, sorting_writer::DEFAULT_MAX_MEMORY_ROWS};

/// Configuration options for the sorting writer's external merge sort behavior.
///
/// These options control how data is buffered, sorted, and spilled to disk
/// during the write phase. They are separate from [`WriterProperties`], which
/// controls Parquet encoding and compression for the final output file.
///
/// Construct with [`SortingWriterOptions::builder`]; the fields are private, so
/// [`SortingWriterOptionsBuilder`] is the only way to set them. Each field has a
/// getter for reading the configuration back — see
/// [`SortingParquetWriter::sorting_options`](crate::writers::SortingParquetWriter::sorting_options).
///
/// # Example
///
/// ```rust,no_run
/// use sorting_parquet_writer::writers::SortingWriterOptions;
///
/// let options = SortingWriterOptions::builder()
///     .with_flush_after_rows_or_bytes(500_000, 256 * 1024 * 1024)
///     .with_temp_dir("/fast-ssd/tmp")
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct SortingWriterOptions {
    /// See [`SortingWriterOptionsBuilder::with_flush_threshold`].
    pub(crate) flush_threshold: FlushThreshold,

    /// See [`SortingWriterOptionsBuilder::with_temp_dir`].
    pub(crate) temp_dir: Option<PathBuf>,

    /// See [`SortingWriterOptionsBuilder::with_run_file_properties`].
    pub(crate) run_file_properties: Option<WriterProperties>,

    /// See [`SortingWriterOptionsBuilder::with_merge_sort_batches`].
    pub(crate) merge_sort_batches: bool,

    /// See [`SortingWriterOptionsBuilder::with_auto_compact_at`].
    pub(crate) auto_compact_at: Option<usize>,
}

impl SortingWriterOptions {
    /// Starts building options from the defaults.
    ///
    /// # Example
    ///
    /// ```rust
    /// use sorting_parquet_writer::writers::{FlushThreshold, SortingWriterOptions};
    ///
    /// let options = SortingWriterOptions::builder()
    ///     .with_flush_after_rows(250_000)
    ///     .with_merge_sort_batches(true)
    ///     .build();
    ///
    /// assert_eq!(options.flush_threshold(), FlushThreshold::Rows(250_000));
    /// assert!(options.merge_sort_batches());
    /// ```
    pub fn builder() -> SortingWriterOptionsBuilder {
        SortingWriterOptionsBuilder::new()
    }

    /// Turns these options back into a builder so a subset of settings can be
    /// overridden without rebuilding the rest.
    ///
    /// # Example
    ///
    /// ```rust
    /// use sorting_parquet_writer::writers::SortingWriterOptions;
    ///
    /// let base = SortingWriterOptions::builder()
    ///     .with_flush_after_rows(250_000)
    ///     .build();
    ///
    /// let tuned = base.clone().into_builder().with_auto_compact_at(64).build();
    ///
    /// assert_eq!(tuned.flush_threshold(), base.flush_threshold());
    /// assert_eq!(tuned.auto_compact_at(), Some(64));
    /// ```
    pub fn into_builder(self) -> SortingWriterOptionsBuilder {
        SortingWriterOptionsBuilder { options: self }
    }

    /// When buffered data is flushed to a sorted run file.
    ///
    /// See [`SortingWriterOptionsBuilder::with_flush_threshold`].
    pub fn flush_threshold(&self) -> FlushThreshold {
        self.flush_threshold
    }

    /// The configured parent directory for temporary run files, or `None` when
    /// the system temp directory is used.
    ///
    /// See [`SortingWriterOptionsBuilder::with_temp_dir`].
    pub fn temp_dir(&self) -> Option<&Path> {
        self.temp_dir.as_deref()
    }

    /// The configured properties for temporary run files, or `None` when the
    /// writer's defaults are used.
    ///
    /// See [`SortingWriterOptionsBuilder::with_run_file_properties`].
    pub fn run_file_properties(&self) -> Option<&WriterProperties> {
        self.run_file_properties.as_ref()
    }

    /// Whether incoming batches are sorted on `write()` and k-way merged at
    /// flush time.
    ///
    /// See [`SortingWriterOptionsBuilder::with_merge_sort_batches`].
    pub fn merge_sort_batches(&self) -> bool {
        self.merge_sort_batches
    }

    /// The peak merge fan-in above which a flush triggers an inline compaction
    /// pass, or `None` when automatic compaction is disabled.
    ///
    /// See [`SortingWriterOptionsBuilder::with_auto_compact_at`].
    pub fn auto_compact_at(&self) -> Option<usize> {
        self.auto_compact_at
    }
}

impl Default for SortingWriterOptions {
    fn default() -> Self {
        Self {
            flush_threshold: FlushThreshold::Rows(DEFAULT_MAX_MEMORY_ROWS),
            temp_dir: None,
            run_file_properties: None,
            merge_sort_batches: false,
            auto_compact_at: None,
        }
    }
}

/// Chainable builder for [`SortingWriterOptions`], and the only way to
/// configure it.
///
/// Created by [`SortingWriterOptions::builder`] (starting from the defaults) or
/// [`SortingWriterOptions::into_builder`] (starting from existing options).
/// Every setter takes and returns `self`, so calls chain and the whole thing
/// ends in [`build()`](Self::build):
///
/// ```rust,no_run
/// use parquet::basic::Compression;
/// use parquet::file::properties::WriterProperties;
/// use sorting_parquet_writer::writers::SortingWriterOptions;
///
/// let options = SortingWriterOptions::builder()
///     .with_flush_after_rows_or_bytes(500_000, 256 * 1024 * 1024)
///     .with_temp_dir("/fast-ssd/tmp")
///     .with_run_file_properties(
///         WriterProperties::builder()
///             .set_compression(Compression::LZ4_RAW)
///             .build(),
///     )
///     .with_auto_compact_at(64)
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct SortingWriterOptionsBuilder {
    options: SortingWriterOptions,
}

impl SortingWriterOptionsBuilder {
    /// Creates a builder holding the [`SortingWriterOptions::default`] values.
    pub fn new() -> Self {
        Self {
            options: SortingWriterOptions::default(),
        }
    }

    /// Sets when buffered data is flushed to a sorted run file.
    ///
    /// See [`with_flush_after_rows`](Self::with_flush_after_rows),
    /// [`with_flush_after_bytes`](Self::with_flush_after_bytes) and
    /// [`with_flush_after_rows_or_bytes`](Self::with_flush_after_rows_or_bytes)
    /// for shorthands that build the individual [`FlushThreshold`] variants.
    ///
    /// Default: `FlushThreshold::Rows(1_000_000)`
    pub fn with_flush_threshold(mut self, flush_threshold: FlushThreshold) -> Self {
        self.options.flush_threshold = flush_threshold;
        self
    }

    /// Flushes a sorted run once `max_rows` rows are buffered.
    ///
    /// Shorthand for [`FlushThreshold::Rows`].
    pub fn with_flush_after_rows(self, max_rows: usize) -> Self {
        self.with_flush_threshold(FlushThreshold::Rows(max_rows))
    }

    /// Flushes a sorted run once the buffered batches are estimated to hold
    /// `max_bytes` bytes in memory.
    ///
    /// Shorthand for [`FlushThreshold::Bytes`].
    pub fn with_flush_after_bytes(self, max_bytes: usize) -> Self {
        self.with_flush_threshold(FlushThreshold::Bytes(max_bytes))
    }

    /// Flushes a sorted run when either limit is reached, whichever comes
    /// first.
    ///
    /// Shorthand for [`FlushThreshold::Either`].
    pub fn with_flush_after_rows_or_bytes(self, max_rows: usize, max_bytes: usize) -> Self {
        self.with_flush_threshold(FlushThreshold::Either {
            max_rows,
            max_bytes,
        })
    }

    /// Sets the directory temporary sorted run files are created in.
    ///
    /// Run files are automatically cleaned up when the writer is finished or
    /// dropped.
    ///
    /// Default: the system's default temp directory.
    pub fn with_temp_dir(mut self, temp_dir: impl Into<PathBuf>) -> Self {
        self.options.temp_dir = Some(temp_dir.into());
        self
    }

    /// Sets the [`WriterProperties`] used for temporary run files, controlling
    /// compression and encoding of intermediate sorted data on disk.
    ///
    /// Tip: use fast compression (e.g., LZ4) for run files even if the final
    /// output uses stronger compression like ZSTD.
    ///
    /// Default: statistics disabled — run files are only read during the merge
    /// phase and then immediately deleted — and default compression.
    pub fn with_run_file_properties(mut self, run_file_properties: WriterProperties) -> Self {
        self.options.run_file_properties = Some(run_file_properties);
        self
    }

    /// When `true`, each incoming
    /// [`RecordBatch`](arrow::array::RecordBatch) is sorted individually on
    /// [`write()`](crate::writers::SortingParquetWriter::write) and the flush
    /// phase merges the pre-sorted batches with a streaming k-way merge instead
    /// of concatenating and re-sorting them from scratch.
    ///
    /// This trades a per-batch sort cost on the write path for a cheaper
    /// flush:
    ///
    /// - With `merge_sort_batches = false` (default), `flush_to_run`
    ///   concatenates every buffered batch and runs one *O(n log n)* sort
    ///   over the result. Peak memory transiently holds both the
    ///   concatenated input and the sorted output.
    /// - With `merge_sort_batches = true`, each `write()` does an *O(b log b)*
    ///   sort on its own batch (size `b`) and the flush is an *O(n)* k-way
    ///   merge that streams across the already-sorted batches. The
    ///   concatenation copy is avoided.
    ///
    /// Enable this when batches arrive unsorted and the flush buffer holds
    /// many batches; the per-batch sort amortizes well against a much
    /// cheaper flush. If batches are already sorted at the source, only the
    /// merge benefit applies.
    ///
    /// Default: `false`
    pub fn with_merge_sort_batches(mut self, merge_sort_batches: bool) -> Self {
        self.options.merge_sort_batches = merge_sort_batches;
        self
    }

    /// Runs one inline compaction pass after any flush that leaves the peak
    /// merge fan-in above `target_fan_in`.
    ///
    /// This is a safety valve, not a performance tuning knob. Every run that
    /// is active at the same moment during the final merge costs an open file
    /// descriptor and a decoded batch, so a long-running writer producing
    /// thousands of *overlapping* runs can exhaust file descriptors or memory
    /// at [`finish()`](crate::writers::SortingParquetWriter::finish). Setting
    /// this bounds that cost, at the price of an occasional stall inside
    /// [`write()`](crate::writers::SortingParquetWriter::write).
    ///
    /// Note this bounds *fan-in*, not run count: runs with disjoint key ranges
    /// have a fan-in of 1 however many there are, and will never trigger a
    /// compaction. See
    /// [`peak_merge_fan_in`](crate::writers::SortingParquetWriter::peak_merge_fan_in).
    ///
    /// For control over *when* the work happens — including running it on
    /// another thread — leave this unset and drive
    /// [`take_compaction_job`](crate::writers::SortingParquetWriter::take_compaction_job)
    /// yourself.
    ///
    /// Takes either a bare `usize` or an `Option<usize>`, so config that is
    /// already optional can be passed straight through; `None` disables
    /// automatic compaction.
    ///
    /// Default: unset (no automatic compaction)
    pub fn with_auto_compact_at(mut self, target_fan_in: impl Into<Option<usize>>) -> Self {
        self.options.auto_compact_at = target_fan_in.into();
        self
    }

    /// Consumes the builder and returns the configured options.
    pub fn build(self) -> SortingWriterOptions {
        self.options
    }
}
