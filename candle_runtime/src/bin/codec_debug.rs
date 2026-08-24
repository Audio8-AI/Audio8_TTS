use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

fn stats(name: &str, t: &Tensor) -> anyhow::Result<()> {
    let t32 = t.to_dtype(DType::F32)?;
    let v: Vec<f32> = t32.flatten_all()?.to_vec1()?;
    let n = v.len() as f64;
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
    println!("{name} shape={:?} mean={mean:.6} std={:.6}", t.shape(), var.sqrt());
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/candle_intermediates");
    std::fs::create_dir_all(&out_dir)?;
    t32.write_npy(out_dir.join(format!("{name}.npy")))?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let weights_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/codec_decoder.npz");
    let codes_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/probe_codes_i64.npy");
    let vb = VarBuilder::from_npz(&weights_path, DType::F32, &device)?;
    let codes = Tensor::read_npy(&codes_path)?;

    // Re-implement forward with prints at each stage using the library's
    // internal pieces via a small re-derivation (module fields are private,
    // so this mirrors CodecDecoder::forward exactly, stage by stage).
    use audio8_candle_runtime::codec_decoder::CodecDecoder;
    let model = CodecDecoder::new(vb)?;
    let audio = model.forward_debug(&codes, &mut |name, t| {
        stats(name, t).expect("stats/dump failed");
    })?;
    stats("final_audio", &audio)?;
    Ok(())
}
