use parquet::arrow::ArrowWriter;

use crate::test::{TestArrowType, TickerItem};
fn base_dir() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.join("test_data")
}
#[test]
#[ignore]
fn generate_generic_unsorted_data() -> anyhow::Result<()> {
    let file = std::fs::File::create(base_dir().join("unsorted_generic_data.parquet"))?;
    let props = parquet::file::properties::WriterProperties::builder()
        .set_created_by("sorting-parquet-writer test".to_string())
        .build();
    let schema = TickerItem::schema();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;

    for i in 0..10 {
        eprintln!("Writing batch {}/10", i + 1);
        let items = TickerItem::random_instances(1024 * 1024);
        for chunk in items.chunks(128) {
            let batch = TickerItem::into_record_batch(chunk.to_vec())?;
            writer.write(&batch)?;
        }
    }
    writer.close()?;
    Ok(())
}
