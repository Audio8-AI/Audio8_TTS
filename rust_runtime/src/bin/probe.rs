use std::path::PathBuf;

use ort::session::Session;

fn model_dir() -> PathBuf {
    let manifest_dir = std::env::var("ARKTTS_MODEL_DIR")
        .unwrap_or_else(|_| "../onnx_runtime/model".to_string());
    PathBuf::from(manifest_dir)
}

fn load_session(name: &str) -> ort::Result<Session> {
    let path = model_dir().join(name);
    println!("[probe] loading {}", path.display());
    let session = Session::builder()?.commit_from_file(&path)?;
    println!("[probe] loaded {}", name);
    for i in session.inputs() {
        println!("  in  {} : {:?}", i.name(), i.dtype());
    }
    for o in session.outputs() {
        println!("  out {} : {:?}", o.name(), o.dtype());
    }
    Ok(session)
}

fn main() -> ort::Result<()> {
    println!("[probe] ort init, CPU execution provider (no CUDA yet)");

    let _slow = load_session("slow_ar_int4.onnx")?;
    let _fast = load_session("fast_ar_int4.onnx")?;
    let _codec = load_session("codec_decoder_fp16.onnx")?;

    println!("[probe] all three sessions loaded successfully on CPU EP");
    Ok(())
}
