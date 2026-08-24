use audio8_candle_runtime::codec_decoder::CodecDecoder;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let weights_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/codec_decoder.npz");
    let codes_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/probe_codes_i64.npy");

    println!("Loading weights from {:?}", weights_path);
    let vb = VarBuilder::from_npz(&weights_path, DType::F32, &device)?;
    let model = CodecDecoder::new(vb)?;
    println!("Model constructed.");

    let codes = Tensor::read_npy(&codes_path)?; // [1, 10, frames], int64
    println!("codes shape: {:?}, dtype: {:?}", codes.shape(), codes.dtype());

    let audio = model.forward(&codes)?;
    println!("audio shape: {:?}, dtype: {:?}", audio.shape(), audio.dtype());

    let audio = audio.to_dtype(DType::F32)?;
    let samples: Vec<f32> = audio.flatten_all()?.to_vec1()?;
    println!("num_samples: {}", samples.len());

    let n = samples.len() as f64;
    let mean = samples.iter().map(|&x| x as f64).sum::<f64>() / n;
    let rms = (samples.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / n).sqrt();
    let max_abs = samples.iter().fold(0f32, |a, &b| a.max(b.abs()));
    let nan_count = samples.iter().filter(|x| x.is_nan()).count();
    println!("mean={mean:.6} rms={rms:.6} max_abs={max_abs:.6} nan_count={nan_count}");

    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("weights/candle_codec_out.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&out_path, spec)?;
    for &s in &samples {
        let clamped = s.clamp(-1.0, 1.0);
        let v = (clamped * i16::MAX as f32) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    println!("Wrote WAV to {:?}", out_path);

    Ok(())
}
