use anyhow::{Context, Result, bail};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::{
    arrow::arrow_reader::{ArrowReaderOptions, ParquetRecordBatchReaderBuilder},
    file::{metadata::SortingColumn, properties::WriterProperties},
};
use sorting_parquet_writer::writers::{
    FinishProgress, FlushThreshold, SortingParquetWriter, SortingWriterOptions,
};
use std::{fmt::Debug, path::PathBuf, str::FromStr};

#[derive(clap::Parser, Debug, Clone)]
#[command(name = "sort-parquet", about = "Sort a parquet file")]
struct Cli {
    #[arg(
        short,
        long,
        help = "Sorting columns in the format: column_name:asc|desc:nulls_first (e.g. timestamp:asc:true)"
    )]
    pub sort_columns: Vec<SortColumnArg>,

    #[arg(short, long, help = "Output file path")]
    pub output: PathBuf,

    #[arg(
        long,
        help = "Maximum rows to buffer in memory before spilling to disk"
    )]
    pub max_memory_rows: Option<usize>,
    #[arg(
        long,
        help = "Maximum memory to use before spilling to disk (e.g. 500MB, 2GB, etc.)"
    )]
    pub max_memory_bytes: Option<MemorySize>,
    #[arg(long, help = "Use merge sort for sorting batches", action = clap::ArgAction::SetTrue)]
    pub merge_sort: bool,
    pub file: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortColumnArg {
    pub name: String,
    pub direction: SortDirection,
    pub nulls_first: bool,
}

impl FromStr for SortColumnArg {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() < 2 || parts.len() > 3 {
            return Err(format!("Invalid sort column argument: {}", s));
        }
        let name = parts[0].to_string();
        let direction = match parts[1] {
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
            false
        };
        Ok(SortColumnArg {
            name,
            direction,
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
    let metadata = reader_builder.metadata().clone();
    let total_rows: u64 = metadata
        .row_groups()
        .iter()
        .map(|rg| rg.num_rows() as u64)
        .sum();

    // Resolve sort columns
    let sort_columns = if !cli.sort_columns.is_empty() {
        println!("Using sort columns from CLI arguments...");
        convert_sort_columns(&cli.sort_columns, &schema)?
    } else {
        println!("No sort columns provided, reading from parquet metadata...");
        extract_sort_columns_from_metadata(&reader_builder, &schema)?
    };

    println!("Sorting by {} column(s):", sort_columns.len());
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

    // Build writer properties
    let properties = WriterProperties::builder()
        .set_sorting_columns(Some(sort_columns.clone()))
        .build();
    let flush_threshold = match (cli.max_memory_rows, cli.max_memory_bytes) {
        (Some(rows), Some(bytes)) => {
            println!(
                "Using flush threshold: {} rows or {} bytes (whichever comes first)",
                rows, bytes.0
            );
            FlushThreshold::Either {
                max_rows: rows,
                max_bytes: bytes.0,
            }
        }
        (Some(rows), None) => {
            println!("Using flush threshold: {} rows", rows);
            FlushThreshold::Rows(rows)
        }
        (None, Some(bytes)) => {
            println!("Using flush threshold: {} bytes", bytes.0);
            FlushThreshold::Bytes(bytes.0)
        }
        (None, None) => {
            println!("Using default flush threshold: 1,000,000 rows");
            FlushThreshold::Rows(1_000_000)
        }
    };
    let options = SortingWriterOptions {
        flush_threshold,
        merge_sort_batches: cli.merge_sort,
        ..Default::default()
    };
    println!("Sorting Writer Options: {:?}", options);

    let output_file = std::fs::File::create(&cli.output)
        .with_context(|| format!("Failed to create output file: {}", cli.output.display()))?;

    let mut writer = SortingParquetWriter::try_new_with_options(
        output_file,
        schema.clone(),
        properties,
        options,
    )
    .context("Failed to create sorting writer")?;

    // Read phase — feed all batches into the writer
    let read_bar = ProgressBar::new(total_rows);
    read_bar.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} rows ({eta})")
            .unwrap()
            .progress_chars("##-"),
    );
    read_bar.set_message("Reading");

    let mut reader = reader_builder
        .build()
        .context("Failed to build parquet reader")?;
    let start = std::time::Instant::now();
    for batch_result in &mut reader {
        let batch = batch_result.context("Failed to read record batch")?;
        let num_rows = batch.num_rows() as u64;
        writer
            .write(&batch)
            .context("Failed to write batch to sorting writer")?;
        read_bar.inc(num_rows);
    }
    read_bar.finish_with_message("Read complete");

    // Finish phase — sort and merge
    let num_runs = writer.num_run_files();
    println!(
        "Finishing: {} run file(s) to merge, {} total rows",
        num_runs, total_rows
    );

    let finish_bar = ProgressBar::new(total_rows);
    finish_bar.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40.green/white}] {pos}/{len} rows ({eta})")
            .unwrap()
            .progress_chars("##-"),
    );
    finish_bar.set_message("Sorting/Merging");

    writer
        .finish_with_progress(|p: &FinishProgress| {
            finish_bar.set_position(p.rows_written);
        })
        .context("Failed to finish sorting writer")?;

    finish_bar.finish_with_message("Done");

    let output_size = std::fs::metadata(&cli.output).map(|m| m.len()).unwrap_or(0);
    println!(
        "\nWrote {} rows to {} ({:.2} MB) in {:.2} seconds",
        total_rows,
        cli.output.display(),
        output_size as f64 / (1024.0 * 1024.0),
        start.elapsed().as_secs_f64()
    );

    Ok(())
}

fn extract_sort_columns_from_metadata(
    reader_builder: &ParquetRecordBatchReaderBuilder<std::fs::File>,
    schema: &arrow::datatypes::Schema,
) -> Result<Vec<SortingColumn>> {
    let metadata = reader_builder.metadata();

    if metadata.num_row_groups() == 0 {
        bail!("Parquet file contains no row groups");
    }

    let row_group_meta = metadata.row_group(0);

    match row_group_meta.sorting_columns() {
        Some(sorting_cols) if !sorting_cols.is_empty() => {
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
        _ => {
            bail!(
                "Parquet file has no sorting metadata. \
                 Please specify sort columns using --sort-columns."
            );
        }
    }
}

fn convert_sort_columns(
    sort_columns: &[SortColumnArg],
    schema: &arrow::datatypes::Schema,
) -> Result<Vec<SortingColumn>> {
    let mut result = Vec::new();
    for col in sort_columns {
        let index = schema
            .index_of(&col.name)
            .with_context(|| format!("Column not found in schema: {}", col.name))?;
        result.push(SortingColumn {
            column_idx: index as i32,
            descending: matches!(col.direction, SortDirection::Descending),
            nulls_first: col.nulls_first,
        });
    }
    Ok(result)
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemorySize(pub usize);
impl FromStr for MemorySize {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim().to_uppercase();
        let multiplier = if s.ends_with("KB") {
            1024
        } else if s.ends_with("MB") {
            1024 * 1024
        } else if s.ends_with("GB") {
            1024 * 1024 * 1024
        } else if s.ends_with("B") {
            1
        } else {
            return Err(format!("Invalid memory size suffix in '{}'", s));
        };
        let num_part = s.trim_end_matches(|c: char| !c.is_ascii_digit()).trim();
        let num = num_part.parse::<usize>().map_err(|e| {
            format!(
                "Failed to parse numeric part of memory size '{}': {}",
                num_part, e
            )
        })?;
        Ok(MemorySize(num * multiplier))
    }
}
