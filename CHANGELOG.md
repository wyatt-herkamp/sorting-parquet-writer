# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-04-04

### Added
- `SortingParquetWriter` — globally sorted Parquet output via external merge sort with bounded memory
- `SortedGroupsParquetWriter` — per-row-group sorting without temporary files
- Streaming k-way merge with lazy run activation based on sort key ranges
- `FinishProgressHandler` trait and `finish_with_progress()` for merge-phase progress tracking
- `sort-parquet` example CLI — reads a Parquet file, sorts it, writes sorted output with indicatif progress bars
- `sort-checker` example CLI — validates that a Parquet file is correctly sorted
