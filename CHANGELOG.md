# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
## [0.2.1] (2026-05-12)

### Added
- Improve Documentation
- Add Check for Empty Batches in SortingParquetWriter::write() to avoid unnecessary flushes and potential schema mismatches when empty batches are written.
- Add Schema Validation in SortingParquetWriter::write() to ensure that the schema of incoming batches matches the writer's schema, preventing runtime errors during sorting and merging.

## [0.2.0] (2026-05-02)

### Added
- Replace sort with sort_unstable
- Added merge_sort_batches option to SortingParquetWriter for controlling whether to merge sort batches during the merge phase. This can reduce memory usage at the cost of potentially more disk I/O and longer merge times.
- More Benchmarks

## [0.1.0] (2026-04-04)

### Added
- `SortingParquetWriter` — globally sorted Parquet output via external merge sort with bounded memory
- `SortedGroupsParquetWriter` — per-row-group sorting without temporary files
- Streaming k-way merge with lazy run activation based on sort key ranges
- `FinishProgressHandler` trait and `finish_with_progress()` for merge-phase progress tracking
- `sort-parquet` example CLI — reads a Parquet file, sorts it, writes sorted output with indicatif progress bars
- `sort-checker` example CLI — validates that a Parquet file is correctly sorted

[0.1.0]:https://github.com/wyatt-herkamp/sorting-parquet-writer/releases/tag/0.1.0
[0.2.0]:https://github.com/wyatt-herkamp/sorting-parquet-writer/releases/tag/0.2.0