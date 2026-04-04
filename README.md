# Sorting Parquet Writer

A Rust library for writing sorted Parquet files with bounded memory usage. Inspired by [Parquet-Go's SortingWriter](https://pkg.go.dev/github.com/parquet-go/parquet-go#SortingWriter).

## Features

- **Globally sorted output** via external merge sort (`SortingParquetWriter`)
- **Per-row-group sorting** for lighter-weight optimization (`SortedGroupsParquetWriter`)
- **Bounded memory** — configurable row buffer with automatic spill to temporary run files
- **Streaming k-way merge** — final merge reads one batch per run file at a time
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

1. **Write phase** — buffers incoming `RecordBatch`es in memory. When the buffer reaches `max_memory_rows`, it is sorted and flushed to a temporary run file on disk.
2. **Merge phase** (`finish()`) — all sorted run files are merged via a streaming k-way merge into the final output.

Configure via `SortingWriterOptions`:

```rust
use sorting_parquet_writer::writers::SortingWriterOptions;

let options = SortingWriterOptions {
    max_memory_rows: 500_000,                          // rows before spilling (default: 1M)
    temp_dir: Some("/fast-ssd/tmp".into()),             // run file location
    run_file_properties: None,                          // compression for run files
    ..Default::default()
};
```

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

## Limitations

- Only supports int, uint, float, bool, string, and list types. Other Arrow types will produce an error during the merge process.

## License

Apache-2.0 OR MIT
