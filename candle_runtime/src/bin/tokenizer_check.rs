use std::path::Path;

use audio8_candle_runtime::prompt::PromptBuilder;

fn main() -> anyhow::Result<()> {
    let tokenizer_dir = Path::new("../onnx_runtime/model/tokenizer");
    // semantic_begin_id/num_codebooks are irrelevant to plain encode_text
    // parity checks; use the same values rust_runtime uses elsewhere.
    let builder = PromptBuilder::new(tokenizer_dir, 151936, 10)?;

    let cases: Vec<(&str, &str)> = vec![
        ("english", "Hello, world! This is a test sentence for tokenizer parity."),
        ("chinese", "你好，世界！这是一个用于分词器一致性检查的测试句子。"),
        ("japanese", "こんにちは、世界！これはトークナイザーの整合性テストです。"),
        ("mixed", "Hello 你好 こんにちは — mixed script test 123."),
    ];

    for (label, text) in &cases {
        let ids = builder.encode_text(text)?;
        println!("{label}: {:?}", ids);
    }

    Ok(())
}
