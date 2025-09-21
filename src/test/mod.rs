use arrow::{array::RecordBatch, datatypes::SchemaRef};
mod ticker;
pub use ticker::*;
pub mod random_time;
pub trait TestArrowType {
    fn random_instances(n: usize) -> Vec<Self>
    where
        Self: Sized;
    fn sorting_columns() -> Vec<parquet::format::SortingColumn>
    where
        Self: Sized;

    fn schema() -> SchemaRef;

    fn into_record_batch(records: Vec<Self>) -> anyhow::Result<RecordBatch>
    where
        Self: Sized;

    fn from_record_batch(batch: &RecordBatch) -> anyhow::Result<Vec<Self>>
    where
        Self: Sized;

    fn is_sorted(records: &[Self]) -> Option<&[Self]>
    where
        Self: Sized;
}
