# Sorting Parquet Writer

[![Crates.io Version](https://img.shields.io/crates/v/sorting-parquet-writer)](https://crates.io/crates/sorting-parquet-writer)
[![docs.rs](https://img.shields.io/docsrs/sorting-parquet-writer)](https://docs.rs/sorting-parquet-writer)
[![License](https://img.shields.io/crates/l/sorting-parquet-writer)](https://crates.io/crates/sorting-parquet-writer)


A Rust library for writing sorted Parquet files with bounded memory usage. Inspired by [Parquet-Go's SortingWriter](https://pkg.go.dev/github.com/parquet-go/parquet-go#SortingWriter).

## Features

- **Globally sorted output** via external merge sort (`SortingParquetWriter`)
- **Per-row-group sorting** for lighter-weight optimization (`SortedGroupsParquetWriter`)
- **Bounded memory** — configurable row buffer with automatic spill to temporary run files
- **Streaming k-way merge** — final merge reads one batch per run file at a time
- **Run compaction** — merge overlapping run files as you go to bound the cost of the final merge; jobs are `Send + 'static` so they can run on another thread
- **Progress tracking** — callback-based progress reporting during the merge phase
- Supports int, uint, float, bool, string, and list column types

## Quick Start

```rust
use sorting_parquet_writer::writers::{SortingParquetWriter, SortingWriterOptions};
use parquet::file::properties::WriterProperties;
use parquet::file::metadata::SortingColumn;
use arrow::datatypes::{Schema, Field, DataType, SchemaRef};
use std::sync::Arc;

let schema: SchemaRef = Arc::new(Schema::new(vec![
    Field::new("timestamp", DataType::Int64, false),
    Field::new("value", DataType::Float64, false),
]));

let props = WriterProperties::builder()
    .set_sorting_columns(Some(vec![SortingColumn {
        column_idx: 0,
        descending: false,
        nulls_first: false,
    }]))
    .build();

let file = std::fs::File::create("sorted_output.parquet").unwrap();
let mut writer = SortingParquetWriter::try_new(file, schema, props).unwrap();

// Write batches in any order — they will be sorted automatically
// writer.write(&batch)?;

// Finalize: merges all sorted runs into the output file
// let file = writer.finish()?;
```

## Writers

### `SortingParquetWriter`

Produces a **globally sorted** Parquet file using external merge sort:

1. **Write phase** — buffers incoming `RecordBatch`es in memory. When the configured `FlushThreshold` is reached (row count, byte size, or either), the buffer is sorted and flushed to a temporary run file on disk.
2. **Merge phase** (`finish()`) — all sorted run files are merged via a streaming k-way merge into the final output.

Configure via `SortingWriterOptions::builder()`:

```rust
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use sorting_parquet_writer::writers::SortingWriterOptions;

let options = SortingWriterOptions::builder()
    .with_flush_after_rows(500_000)      // rows before spilling (default: 1M)
    // Or byte-based:  .with_flush_after_bytes(256 * 1024 * 1024)
    // Or both:        .with_flush_after_rows_or_bytes(500_000, 256 * 1024 * 1024)
    .with_temp_dir("/fast-ssd/tmp")      // run file location
    .with_run_file_properties(           // compression for run files
        WriterProperties::builder()
            .set_compression(Compression::LZ4_RAW)
            .build(),
    )
    .build();
```

The fields are private — the builder is the only way to set them, and
`SortingWriterOptions` exposes a getter per setting for reading them back.

#### Progress Tracking

Use `finish_with_progress` to monitor the merge phase:

```rust
use sorting_parquet_writer::writers::FinishProgress;

# fn example(writer: sorting_parquet_writer::writers::SortingParquetWriter<std::fs::File>) {
writer.finish_with_progress(|p: &FinishProgress| {
    println!("{:.1}% complete ({} / {} rows)",
        p.fraction_complete() * 100.0,
        p.rows_written,
        p.total_rows,
    );
}).unwrap();
# }
```

#### Run Compaction

Long-running writers accumulate run files. The merger opens a run only once the
merge position reaches its minimum sort key, so runs with **disjoint** key ranges
cost about one open file at a time no matter how many there are. **Overlapping**
runs are the problem: they must all be open at once, each costing a file
descriptor and a decoded batch. Enough of them and `finish()` runs out of file
descriptors or memory.

`peak_merge_fan_in()` reports the worst case — the most runs the final merge will
ever hold open simultaneously. Compaction merges overlapping runs to bring it down:

```rust
use sorting_parquet_writer::writers::CompactionPolicy;

# fn example(writer: &mut sorting_parquet_writer::writers::SortingParquetWriter<std::fs::File>)
# -> Result<(), Box<dyn std::error::Error>> {
// Blocking, in place. Returns Ok(None) when nothing is worth compacting —
// which is always the case for runs with disjoint key ranges.
writer.compact(&CompactionPolicy::default())?;
# Ok(())
# }
```

To overlap compaction with writing, take the job and run it elsewhere. It owns
everything it needs and is `Send + 'static`, so this works on a plain thread, a
pool, or `tokio::task::spawn_blocking` — the crate itself stays runtime-agnostic
and depends on no async runtime:

```rust
use sorting_parquet_writer::writers::CompactionPolicy;

# fn example(writer: &mut sorting_parquet_writer::writers::SortingParquetWriter<std::fs::File>)
# -> Result<(), Box<dyn std::error::Error>> {
if let Some(job) = writer.take_compaction_job(&CompactionPolicy::default()) {
    let handle = std::thread::spawn(move || job.run());

    // ... keep calling writer.write(&batch) here ...

    match handle.join().expect("compaction thread panicked") {
        Ok(output) => writer.apply_compaction(output)?,
        Err(failure) => {
            // Runs are handed back intact; nothing is lost.
            let err = writer.abandon_compaction(failure);
            eprintln!("compaction failed, will retry: {err}");
        }
    }
}
# Ok(())
# }
```

`finish()` returns `SortingParquetError::CompactionInFlight` while any job is
outstanding, since its runs are detached from the writer.

For a hands-off safety valve, set `auto_compact_at` and the writer compacts
inline whenever a flush pushes fan-in past the limit:

```rust
use sorting_parquet_writer::writers::SortingWriterOptions;

let options = SortingWriterOptions::builder()
    .with_auto_compact_at(256)  // default: no automatic compaction
    .build();
```

### `SortedGroupsParquetWriter`

Sorts **individual row groups** without guaranteeing global sort order. Lighter weight than `SortingParquetWriter` — no temporary files needed. Useful when queries primarily filter within row groups.

## Examples

### `sort-parquet` — Sort a Parquet file

```bash
cargo run --example sort-parquet -- \
  --sort-columns "timestamp:asc:true" \
  --output sorted.parquet \
  input.parquet

# With custom memory limit
cargo run --example sort-parquet -- \
  --sort-columns "id:asc:false" \
  --max-memory-rows 500000 \
  --output sorted.parquet \
  input.parquet
```

### `sort-checker` — Verify sort order

```bash
cargo run --example sort-checker -- \
  --sort-columns "timestamp:asc:true" \
  input.parquet
```


## License

Apache-2.0 OR MIT
