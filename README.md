# Sorting Parquet Writer

A low memory parquet writer that sorts the data before writing it to disk.

## Limitations
- Currently only sorts the individual row groups, not the entire file.
  - This still shows siginificant performance improvements when reading and filtering data.
- Only Supports int, uint, float, bool, string, and list types. Others will result in a error during the merge process.


## How does it work?
The sorting parquet writers works by taking in record batches and sorting them in memory and caching them until a certain threshold is reached.

Once the threshold is reached, the cached record batches are merged into a single sorted record batch and written to disk as a row group. This process is repeated until all record batches have been written.