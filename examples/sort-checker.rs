use anyhow::{Context, Result, bail};
use arrow::array::{ArrayRef, RecordBatch};
use arrow::compute::SortOptions;
use arrow_row::{RowConverter, SortField};
use clap::Parser;
use parquet::{
    arrow::arrow_reader::{
        ArrowReaderOptions, ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder,
    },
    file::metadata::SortingColumn,
};
use std::{
    fmt::{Debug, Display},
    path::PathBuf,
    str::FromStr,
};
#[derive(clap::Parser, Debug, Clone)]
struct Cli {
    #[arg(
        short,
        long,
        help = "Sorting columns in the format: column_name:asc|desc:nulls_first (e.g. timestamp:asc:true)"
    )]
    pub sort_columns: Vec<SortColumnArg>,
    pub file: PathBuf,
}

/// Sort Column Arg
/// Format: {column_name}:{asc|desc}:{true|false}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortColumnArg {
    pub name: String,
    pub data_type: SortDirection,
    pub nulls_first: bool,
}
impl FromStr for SortColumnArg {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() <= 2 || parts.len() > 3 {
            return Err(format!("Invalid sort column argument: {}", s));
        }
        let name = parts[0].to_string();
        let data_type = match parts[1] {
            "asc" => SortDirection::Ascending,
            "desc" => SortDirection::Descending,
            _ => return Err(format!("Invalid sort direction: {}", parts[1])),
        };
        let nulls_first = if parts.len() == 3 {
            match parts[2] {
                "true" => true,
                "false" => false,
                _ => return Err(format!("Invalid nulls first value: {}", parts[2])),
            }
        } else {
            // Default to false if not provided
            false
        };
        Ok(SortColumnArg {
            name,
            data_type,
            nulls_first,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if !cli.file.exists() {
        bail!("File does not exist: {}", cli.file.display());
    }
    let open_file = std::fs::File::open(&cli.file)
        .with_context(|| format!("Failed to open file: {}", cli.file.display()))?;

    let reader_builder =
        ParquetRecordBatchReaderBuilder::try_new_with_options(open_file, ArrowReaderOptions::new())
            .context("Failed to open parquet")?;

    let schema = reader_builder.schema().clone();
    
    // Use CLI sort columns if provided, otherwise try to read from parquet metadata
    let sort_columns = if !cli.sort_columns.is_empty() {
        println!("Using sort columns from CLI arguments...");
        convert_sort_columns(cli.sort_columns, &schema)?
    } else {
        println!("No sort columns provided, reading from parquet metadata...");
        extract_sort_columns_from_metadata(&reader_builder, &schema)?
    };

    println!("Checking sort order for {} columns...", sort_columns.len());
    for (idx, col) in sort_columns.iter().enumerate() {
        let field = schema.field(col.column_idx as usize);
        let direction = if col.descending { "DESC" } else { "ASC" };
        let nulls = if col.nulls_first {
            "NULLS FIRST"
        } else {
            "NULLS LAST"
        };
        println!(
            "  {}: {} ({}) {} {}",
            idx + 1,
            field.name(),
            field.data_type(),
            direction,
            nulls
        );
    }
    println!();

    let mut reader = reader_builder
        .build()
        .context("Failed to build parquet reader")?;

    match check_sorted(&mut reader, &sort_columns, &schema)? {
        Ok(()) => {
            println!("✅ File is correctly sorted");
            std::process::exit(0);
        }
        Err(error) => {
            eprintln!("❌ Sort violation detected:");
            eprintln!("{}", error);
            std::process::exit(1);
        }
    }
}

fn extract_sort_columns_from_metadata(
    reader_builder: &ParquetRecordBatchReaderBuilder<std::fs::File>,
    schema: &arrow::datatypes::Schema,
) -> Result<Vec<SortingColumn>> {
    let metadata = reader_builder.metadata();
    
    // Check if there are any row groups
    if metadata.num_row_groups() == 0 {
        bail!("Parquet file contains no row groups");
    }
    
    // Get sorting metadata from the first row group
    // Note: In a properly sorted parquet file, all row groups should have the same sorting specification
    let row_group_meta = metadata.row_group(0);
    
    match row_group_meta.sorting_columns() {
        Some(sorting_cols) => {
            if sorting_cols.is_empty() {
                bail!(
                    "Parquet file has no sorting metadata. {}",
                    "Please specify sort columns using --sort-columns or ensure the parquet file was written with sorting metadata."
                );
            }
            
            println!("Found {} sorting column(s) in parquet metadata", sorting_cols.len());
            
            // Validate that all row groups have the same sorting specification
            for (idx, rg_meta) in metadata.row_groups().iter().enumerate().skip(1) {
                match rg_meta.sorting_columns() {
                    Some(rg_sorting_cols) if rg_sorting_cols != sorting_cols => {
                        eprintln!(
                            "⚠️  Warning: Row group {} has different sorting specification than row group 0",
                            idx
                        );
                    }
                    None => {
                        eprintln!(
                            "⚠️  Warning: Row group {} has no sorting metadata while row group 0 does",
                            idx
                        );
                    }
                    _ => {} // Same sorting specification, which is expected
                }
            }
            
            // Validate that all column indices exist in the schema
            for col in sorting_cols.iter() {
                if col.column_idx < 0 || (col.column_idx as usize) >= schema.fields().len() {
                    bail!(
                        "Invalid column index {} in sorting metadata. Schema has {} columns",
                        col.column_idx,
                        schema.fields().len()
                    );
                }
            }
            
            Ok(sorting_cols.clone())
        }
        None => {
            bail!(
                "Parquet file has no sorting metadata. {}",
                "Please specify sort columns using --sort-columns or ensure the parquet file was written with sorting metadata."
            );
        }
    }
}

fn convert_sort_columns(
    sort_columns: Vec<SortColumnArg>,
    schema: &arrow::datatypes::Schema,
) -> Result<Vec<SortingColumn>> {
    let mut result = Vec::new();
    for col in sort_columns {
        let index = schema
            .index_of(&col.name)
            .with_context(|| format!("Column not found in schema: {}", col.name))?;
        result.push(SortingColumn {
            column_idx: index as i32,
            descending: matches!(col.data_type, SortDirection::Descending),
            nulls_first: col.nulls_first,
        });
    }
    Ok(result)
}

struct SortError {
    batch_number: usize,
    row_number: usize,
    column_name: String,
    previous_value: String,
    current_value: String,
    sort_direction: String,
}

impl Debug for SortError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SortError at batch {} row {}: column '{}' violates {} order (prev: {}, current: {})",
            self.batch_number,
            self.row_number,
            self.column_name,
            self.sort_direction,
            self.previous_value,
            self.current_value
        )
    }
}

impl Display for SortError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Sort violation in column '{}':", self.column_name)?;
        writeln!(
            f,
            "  Location: batch {}, row {}",
            self.batch_number, self.row_number
        )?;
        writeln!(f, "  Expected: {} order", self.sort_direction)?;
        writeln!(f, "  Previous value: {}", self.previous_value)?;
        writeln!(f, "  Current value:  {}", self.current_value)?;
        write!(f, "  Issue: Values are not in correct order")
    }
}

impl std::error::Error for SortError {}
fn check_sorted(
    reader: &mut ParquetRecordBatchReader,
    sort_columns: &[SortingColumn],
    schema: &arrow::datatypes::Schema,
) -> Result<Result<(), SortError>> {
    if sort_columns.is_empty() {
        return Ok(Ok(()));
    }

    // Create sort fields for RowConverter
    let mut sort_fields = Vec::with_capacity(sort_columns.len());
    for col in sort_columns {
        let field = schema.field(col.column_idx as usize);
        let sort_field = SortField::new_with_options(
            field.data_type().clone(),
            SortOptions {
                descending: col.descending,
                nulls_first: col.nulls_first,
            },
        );
        sort_fields.push(sort_field);
    }

    let row_converter = RowConverter::new(sort_fields).context("Failed to create row converter")?;

    let mut previous_last_row_data: Option<(Vec<u8>, RecordBatch, usize)> = None;
    let mut batch_number = 0;
    let mut total_rows_processed = 0;

    for batch_result in reader {
        let batch = batch_result.context("Failed to read record batch")?;
        batch_number += 1;

        if batch.num_rows() == 0 {
            continue;
        }

        print!(
            "Processing batch {} ({} rows)",
            batch_number,
            batch.num_rows()
        );
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        // Extract the sort key columns for this batch
        let columns: Vec<ArrayRef> = sort_columns
            .iter()
            .map(|col| batch.column(col.column_idx as usize).clone())
            .collect();

        let rows = row_converter
            .convert_columns(&columns)
            .context("Failed to convert columns to rows")?;

        // Check first row of this batch against last row of previous batch
        if let Some((prev_row_bytes, prev_batch, prev_row_idx)) = &previous_last_row_data {
            let current_row = rows.row(0);
            if prev_row_bytes.as_slice() > current_row.as_ref() {
                println!();
                let error = create_sort_error(
                    batch_number,
                    1,                // First row of current batch
                    &sort_columns[0], // Report on first sort column for simplicity
                    schema,
                    prev_batch,
                    *prev_row_idx,
                    &batch,
                    0,
                );
                return Ok(Err(error));
            }
        }

        // Check all consecutive rows within this batch
        for row_idx in 1..batch.num_rows() {
            let prev_row = rows.row(row_idx - 1);
            let current_row = rows.row(row_idx);

            if prev_row > current_row {
                println!();
                let error = create_sort_error(
                    batch_number,
                    row_idx + 1,      // 1-based row number
                    &sort_columns[0], // Report on first sort column for simplicity
                    schema,
                    &batch,
                    row_idx - 1,
                    &batch,
                    row_idx,
                );
                return Ok(Err(error));
            }
        }

        // Store the last row from this batch for comparison with next batch
        let last_row_idx = batch.num_rows() - 1;
        let last_row = rows.row(last_row_idx);
        let last_row_bytes = last_row.as_ref().to_vec();
        previous_last_row_data = Some((last_row_bytes, batch.clone(), last_row_idx));

        total_rows_processed += batch.num_rows();
        println!(" ✓");
    }

    println!(
        "\nValidated {} rows across {} batches",
        total_rows_processed, batch_number
    );
    Ok(Ok(()))
}

fn create_sort_error(
    batch_number: usize,
    row_number: usize,
    sort_column: &SortingColumn,
    schema: &arrow::datatypes::Schema,
    prev_batch: &RecordBatch,
    prev_row_idx: usize,
    current_batch: &RecordBatch,
    current_row_idx: usize,
) -> SortError {
    let field = schema.field(sort_column.column_idx as usize);
    let column_name = field.name().to_string();

    let prev_column = prev_batch.column(sort_column.column_idx as usize);
    let current_column = current_batch.column(sort_column.column_idx as usize);

    let previous_value = format_array_value(prev_column, prev_row_idx);
    let current_value = format_array_value(current_column, current_row_idx);

    let sort_direction = if sort_column.descending {
        "descending".to_string()
    } else {
        "ascending".to_string()
    };

    SortError {
        batch_number,
        row_number,
        column_name,
        previous_value,
        current_value,
        sort_direction,
    }
}

fn format_array_value(array: &ArrayRef, row_idx: usize) -> String {
    if array.is_null(row_idx) {
        "NULL".to_string()
    } else {
        // Use Arrow's display formatting
        let single_value = array.slice(row_idx, 1);
        format!("{:?}", single_value)
    }
}
