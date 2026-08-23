use std::path::Path;

use half::f16;
use ndarray::{Array2, ArrayD, IxDyn};
use ort::session::Session;
use ort::value::Tensor;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Deserialize;

use crate::prompt::PromptBuilder;
use crate::sampling::sample;

#[derive(Deserialize)]
pub struct RuntimeManifest {
    pub max_seq_len: usize,
    pub num_layers: usize,
    pub num_fast_layers: usize,
    pub num_codebooks: usize,
    pub n_local_heads: usize,
    pub fast_n_local_heads: usize,
    pub head_dim: usize,
    pub fast_head_dim: usize,
    pub semantic_begin_id: i64,
    pub semantic_end_id: i64,
    pub im_end_id: i64,
    pub codebook_size: usize,
    pub codec_hop_length: usize,
    #[serde(default)]
    pub slow_logits_layout: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProviderChoice {
    Cpu,
    Cuda,
}

pub struct ArkTtsRuntime {
    slow: Session,
    fast: Session,
    decoder: Session,
    prompt_builder: PromptBuilder,
    manifest: RuntimeManifest,
}

impl ArkTtsRuntime {
    pub fn load(model_dir: &Path) -> anyhow::Result<Self> {
        Self::load_with_ep(model_dir, ExecutionProviderChoice::Cpu)
    }

    pub fn load_with_ep(model_dir: &Path, ep: ExecutionProviderChoice) -> anyhow::Result<Self> {
        let manifest_text = std::fs::read_to_string(model_dir.join("runtime_manifest.json"))?;
        let manifest: RuntimeManifest = serde_json::from_str(&manifest_text)?;

        let build_session = |file: &str| -> anyhow::Result<Session> {
            let builder = Session::builder().map_err(|e| anyhow::anyhow!("session builder: {e}"))?;
            let mut builder = match ep {
                ExecutionProviderChoice::Cpu => builder,
                ExecutionProviderChoice::Cuda => builder
                    .with_execution_providers([ort::ep::CUDA::default().build()])
                    .map_err(|e| anyhow::anyhow!("cuda ep: {e}"))?,
            };
            builder
                .commit_from_file(model_dir.join(file))
                .map_err(|e| anyhow::anyhow!("commit {file}: {e}"))
        };

        let slow = build_session("slow_ar_int4.onnx")?;
        let fast = build_session("fast_ar_int4.onnx")?;
        let decoder = build_session("codec_decoder_fp16.onnx")?;

        let prompt_builder = PromptBuilder::new(
            &model_dir.join("tokenizer"),
            manifest.semantic_begin_id,
            manifest.num_codebooks,
        )?;

        Ok(Self { slow, fast, decoder, prompt_builder, manifest })
    }

    fn empty_slow_caches(&self) -> Vec<ArrayD<f16>> {
        let shape = IxDyn(&[1, self.manifest.n_local_heads, self.manifest.max_seq_len, self.manifest.head_dim]);
        (0..2 * self.manifest.num_layers).map(|_| ArrayD::from_elem(shape.clone(), f16::ZERO)).collect()
    }

    fn empty_fast_caches(&self) -> Vec<ArrayD<f16>> {
        let shape = IxDyn(&[1, self.manifest.fast_n_local_heads, self.manifest.num_codebooks, self.manifest.fast_head_dim]);
        (0..2 * self.manifest.num_fast_layers).map(|_| ArrayD::from_elem(shape.clone(), f16::ZERO)).collect()
    }

    /// codes: [1, num_codebooks+1, T], positions: [T]
    /// Returns (logits [vocab_ish], hidden [1,1,hidden_dim]) and mutates caches in place.
    fn slow_step(
        &mut self,
        codes: &ArrayD<i64>,
        positions: &[i64],
        caches: &mut [ArrayD<f16>],
    ) -> anyhow::Result<(Vec<f32>, ArrayD<f16>)> {
        let mut inputs = ort::inputs![
            "codes" => Tensor::from_array(codes.clone())?,
            "input_pos" => Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[positions.len()]), positions.to_vec())?)?,
        ];
        for i in 0..self.manifest.num_layers {
            inputs.push((format!("cache_key_{i}").into(), Tensor::from_array(caches[2 * i].clone())?.into()));
            inputs.push((format!("cache_value_{i}").into(), Tensor::from_array(caches[2 * i + 1].clone())?.into()));
        }

        let outputs = self.slow.run(inputs)?;
        let logits_view = outputs["logits"].try_extract_array::<f32>()?;
        let hidden_view = outputs["slow_hidden"].try_extract_array::<f16>()?;

        // logits: [1, T, vocab]; take last timestep row.
        let logits_shape = logits_view.shape();
        let last_t = logits_shape[1] - 1;
        let vocab = logits_shape[2];
        let logits: Vec<f32> = (0..vocab).map(|v| logits_view[[0, last_t, v]]).collect();

        // slow_hidden: [1, T, hidden]; take last timestep, keep as [1,1,hidden].
        let hidden_shape = hidden_view.shape();
        let hidden_dim = hidden_shape[2];
        let hidden_last_t = hidden_shape[1] - 1;
        let mut hidden = ArrayD::<f16>::from_elem(IxDyn(&[1, 1, hidden_dim]), f16::ZERO);
        for h in 0..hidden_dim {
            hidden[[0, 0, h]] = hidden_view[[0, hidden_last_t, h]];
        }

        for i in 0..self.manifest.num_layers {
            let key_delta = outputs[format!("key_delta_{i}")].try_extract_array::<f16>()?;
            let value_delta = outputs[format!("value_delta_{i}")].try_extract_array::<f16>()?;
            update_cache_at_positions(&mut caches[2 * i], &key_delta, positions);
            update_cache_at_positions(&mut caches[2 * i + 1], &value_delta, positions);
        }

        Ok((logits, hidden))
    }

    fn fast_step(
        &mut self,
        hidden: &ArrayD<f16>,
        token_id: i64,
        use_hidden: bool,
        position: i64,
        caches: &mut [ArrayD<f16>],
    ) -> anyhow::Result<Vec<f32>> {
        let mut inputs = ort::inputs![
            "slow_hidden" => Tensor::from_array(hidden.clone())?,
            "token_id" => Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![token_id])?)?,
            "use_slow_hidden" => Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[1]), vec![use_hidden])?)?,
            "input_pos" => Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[1]), vec![position])?)?,
        ];
        for i in 0..self.manifest.num_fast_layers {
            inputs.push((format!("cache_key_{i}").into(), Tensor::from_array(caches[2 * i].clone())?.into()));
            inputs.push((format!("cache_value_{i}").into(), Tensor::from_array(caches[2 * i + 1].clone())?.into()));
        }

        let outputs = self.fast.run(inputs)?;
        let logits_view = outputs["logits"].try_extract_array::<f32>()?;
        let logits_shape = logits_view.shape();
        let last_t = logits_shape[1] - 1;
        let vocab = logits_shape[2];
        let logits: Vec<f32> = (0..vocab).map(|v| logits_view[[0, last_t, v]]).collect();

        for i in 0..self.manifest.num_fast_layers {
            let key_delta = outputs[format!("key_delta_{i}")].try_extract_array::<f16>()?;
            let value_delta = outputs[format!("value_delta_{i}")].try_extract_array::<f16>()?;
            update_cache_at_positions(&mut caches[2 * i], &key_delta, &[position]);
            update_cache_at_positions(&mut caches[2 * i + 1], &value_delta, &[position]);
        }

        Ok(logits)
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_semantic(
        &self,
        logits: &[f32],
        previous: &[i64],
        temperature: f32,
        top_p: f32,
        top_k: usize,
        rng: &mut StdRng,
    ) -> i64 {
        let begin = self.manifest.semantic_begin_id;
        let end = self.manifest.semantic_end_id;
        let stop = self.manifest.im_end_id;
        let allowed_ids: Vec<i64> = (begin..=end).chain(std::iter::once(stop)).collect();
        let allowed_logits: Vec<f32> = if self.manifest.slow_logits_layout.as_deref() == Some("semantic_then_eos") {
            logits.to_vec()
        } else {
            allowed_ids.iter().map(|&id| logits[id as usize]).collect()
        };
        debug_assert_eq!(allowed_logits.len(), allowed_ids.len());

        let normal_idx = sample(&allowed_logits, temperature, top_p, top_k, rng);
        let normal = allowed_ids[normal_idx];
        let high_idx = sample(&allowed_logits, 1.0, 0.9, top_k, rng);
        let high = allowed_ids[high_idx];

        if normal >= begin && normal <= end && previous.contains(&normal) {
            high
        } else {
            normal
        }
    }

    /// Runs the full autoregressive loop for one utterance (no reference
    /// voice), returning generated codec frames as [num_codebooks, T].
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_no_reference(
        &mut self,
        text: &str,
        reference_text: &str,
        reference_codes: &Array2<i64>,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        seed: u64,
    ) -> anyhow::Result<Array2<i64>> {
        let prompt = self.prompt_builder.build(text, reference_text, reference_codes)?;
        let prompt_len = prompt.shape()[2];
        if prompt_len >= self.manifest.max_seq_len {
            anyhow::bail!("prompt length {} exceeds max sequence length {}", prompt_len, self.manifest.max_seq_len);
        }
        let max_new_tokens = max_new_tokens.min(self.manifest.max_seq_len - prompt_len);

        let mut rng = StdRng::seed_from_u64(seed);
        let mut slow_caches = self.empty_slow_caches();
        let positions: Vec<i64> = (0..prompt_len as i64).collect();

        let prompt_dyn = prompt.into_dyn();
        let (mut logits, mut hidden) = self.slow_step(&prompt_dyn, &positions, &mut slow_caches)?;

        let mut previous: Vec<i64> = Vec::new();
        let begin = self.manifest.semantic_begin_id;
        let stop = self.manifest.im_end_id;
        let codebook_size = self.manifest.codebook_size as i64;

        let mut frames: Vec<Vec<i64>> = Vec::new();

        for step in 0..max_new_tokens {
            let semantic = self.sample_semantic(&logits, &previous, temperature, top_p, top_k, &mut rng);
            if semantic == stop {
                break;
            }
            previous.push(semantic);
            if previous.len() > 10 {
                previous.remove(0);
            }

            let mut fast_caches = self.empty_fast_caches();
            self.fast_step(&hidden, 0, true, 0, &mut fast_caches)?;

            let token0 = (semantic - begin).clamp(0, codebook_size - 1);
            let mut codebooks = vec![token0];
            for fast_pos in 1..self.manifest.num_codebooks as i64 {
                let fast_logits = self.fast_step(&hidden, *codebooks.last().unwrap(), false, fast_pos, &mut fast_caches)?;
                let token = sample(&fast_logits, temperature, top_p, top_k, &mut rng) as i64;
                codebooks.push(token);
            }

            frames.push(codebooks.clone());

            if step + 1 >= max_new_tokens {
                break;
            }

            let mut column = vec![semantic];
            column.extend(&codebooks);
            let column_arr = ArrayD::from_shape_vec(IxDyn(&[1, column.len(), 1]), column)?;
            let position = [prompt_len as i64 + step as i64];
            let (next_logits, next_hidden) = self.slow_step(&column_arr, &position, &mut slow_caches)?;
            logits = next_logits;
            hidden = next_hidden;
        }

        if frames.is_empty() {
            anyhow::bail!("model produced no codec frames");
        }

        let num_codebooks = self.manifest.num_codebooks;
        let t = frames.len();
        let mut codes = Array2::<i64>::zeros((num_codebooks, t));
        for (col, frame) in frames.iter().enumerate() {
            for row in 0..num_codebooks {
                codes[[row, col]] = frame[row];
            }
        }
        Ok(codes)
    }

    pub fn decode_codes(&mut self, codes: &Array2<i64>) -> anyhow::Result<Vec<f32>> {
        let shaped = codes.clone().insert_axis(ndarray::Axis(0)).into_dyn();
        let inputs = ort::inputs!["codes" => Tensor::from_array(shaped)?];
        let outputs = self.decoder.run(inputs)?;
        let audio_view = outputs["audio"].try_extract_array::<f32>()?;
        Ok(audio_view.iter().copied().collect())
    }
}

fn update_cache_at_positions(cache: &mut ArrayD<f16>, delta: &ndarray::ArrayViewD<f16>, positions: &[i64]) {
    // cache: [1, heads, max_seq_len, head_dim]; delta: [1, heads, len(positions), head_dim]
    let heads = cache.shape()[1];
    let head_dim = cache.shape()[3];
    for (i, &pos) in positions.iter().enumerate() {
        for h in 0..heads {
            for d in 0..head_dim {
                cache[[0, h, pos as usize, d]] = delta[[0, h, i, d]];
            }
        }
    }
}
