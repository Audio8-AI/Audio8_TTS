use std::path::Path;

use half::f16;
use ndarray::{Array2, ArrayD, IxDyn};
use ort::environment::Environment;
use ort::session::{IoBinding, Session};
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

/// Registers a real console logger with the ORT Environment BEFORE any
/// session is created - SessionBuilder::with_log_level alone only sets a
/// severity filter and produces zero output without this, since ort has no
/// default logger wired to a Rust `eprintln!`/console destination. Must be
/// called before the first Session::builder() commit in the process; ort's
/// global environment is a OnceLock, so a second call after the environment
/// is already initialized is a silent no-op (commit() returns false).
fn init_verbose_ort_logging() -> anyhow::Result<()> {
    let logger: ort::logging::LoggerFunction = std::sync::Arc::new(
        |level: ort::logging::LogLevel, category: &str, id: &str, code_location: &str, message: &str| {
            eprintln!("[ort:{level:?}] [{category}] [{id}] {code_location}: {message}");
        },
    );
    let committed = ort::init().with_logger(logger).commit();
    if !committed {
        eprintln!("[diag] ORT environment already initialized before verbose logging was requested - logger not installed");
    }
    let env = Environment::current().map_err(|e| anyhow::anyhow!("Environment::current: {e}"))?;
    env.set_log_level(ort::logging::LogLevel::Verbose);
    Ok(())
}

// NOTE on device residency: the natural fix here would be allocating these
// persistent tensors on CUDA_PINNED memory (DMA-able by the GPU, still
// host-writable) via ort::memory::Allocator::new(session, MemoryInfo::new(
// AllocationDevice::CUDA_PINNED, ...)) - matching ort's own IoBinding docs
// example. As of ort 2.0.0-rc.13 this consistently fails with ORT's own
// "No requested allocator available" for both AllocationDevice::CUDA and
// CUDA_PINNED on this session; ort exposes no binding for the
// CreateAndRegisterAllocator / use_env_allocators mechanism ONNX Runtime's
// own issue tracker names as the actual fix (registering a shared allocator
// at the Environment level before session creation). session.allocator()
// (the session's default CPU allocator) is used instead below - real device
// residency for these tensors is not currently reachable through ort's
// public API. The fix implemented instead (this session) targets the
// measured bottleneck directly: writing only the changed delta into the
// persistent tensor each step rather than re-copying the whole cache.

/// Persistent, address-stable device buffers + IoBinding for the fast-AR
/// session, reused across every fast_step call within a frame and across
/// frames - required for CUDA graph capture, which replays the exact same
/// memory addresses on every call after the first.
struct FastGraphState {
    binding: IoBinding,
    hidden: Tensor<f16>,
    token_id: Tensor<i64>,
    use_hidden: Tensor<bool>,
    input_pos: Tensor<i64>,
    cache_keys: Vec<Tensor<f16>>,
    cache_values: Vec<Tensor<f16>>,
}

/// Same idea as FastGraphState, but for slow_step's steady-state decode call
/// (T=1, one new column per step after the initial variable-length prefill).
/// The prefill call itself keeps using the plain allocate-per-call path since
/// its shape varies with prompt length - CUDA graph capture only applies to
/// the fixed-shape decode calls that follow it.
struct SlowGraphState {
    binding: IoBinding,
    codes: Tensor<i64>,
    input_pos: Tensor<i64>,
    cache_keys: Vec<Tensor<f16>>,
    cache_values: Vec<Tensor<f16>>,
}

pub struct ArkTtsRuntime {
    slow: Session,
    fast: Session,
    decoder: Session,
    prompt_builder: PromptBuilder,
    manifest: RuntimeManifest,
    fast_graph: Option<FastGraphState>,
    slow_graph: Option<SlowGraphState>,
}

impl ArkTtsRuntime {
    pub fn load(model_dir: &Path) -> anyhow::Result<Self> {
        Self::load_with_ep(model_dir, ExecutionProviderChoice::Cpu)
    }

    pub fn load_with_ep(model_dir: &Path, ep: ExecutionProviderChoice) -> anyhow::Result<Self> {
        if std::env::var("ARKTTS_VERBOSE_ORT").is_ok() {
            init_verbose_ort_logging()?;
        }

        let manifest_text = std::fs::read_to_string(model_dir.join("runtime_manifest.json"))?;
        let manifest: RuntimeManifest = serde_json::from_str(&manifest_text)?;

        let build_session = |file: &str| -> anyhow::Result<Session> {
            let builder = Session::builder().map_err(|e| anyhow::anyhow!("session builder: {e}"))?;
            let builder = match ep {
                ExecutionProviderChoice::Cpu => builder,
                ExecutionProviderChoice::Cuda => {
                    // Graph capture is off by default: ORT hard-errors ("all
                    // compute graph nodes have not been partitioned to the
                    // CUDAExecutionProvider") because this model has a real,
                    // confirmed mix of CUDA- and CPU-assigned ops (INT4
                    // quantization-adjacent and shape ops land on CPU even
                    // with the CUDA EP active) - not fixable from this crate.
                    // ARKTTS_CUDA_GRAPH=1 re-enables it, e.g. to test a future
                    // model export with full CUDA op coverage.
                    let use_graph = std::env::var("ARKTTS_CUDA_GRAPH").is_ok();
                    let cuda_ep = ort::ep::CUDA::default()
                        .with_cuda_graph(use_graph)
                        .with_tf32(true)
                        .with_attention_backend(
                            ort::ep::cuda::AttentionBackend::FLASH_ATTENTION
                                | ort::ep::cuda::AttentionBackend::EFFICIENT_ATTENTION
                                | ort::ep::cuda::AttentionBackend::CUDNN_FLASH_ATTENTION,
                        )
                        .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Exhaustive)
                        .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested);
                    let builder = builder
                        .with_execution_providers([cuda_ep.build()])
                        .map_err(|e| anyhow::anyhow!("cuda ep: {e}"))?;
                    if std::env::var("ARKTTS_FORCE_CUDA_ONLY").is_ok() {
                        builder.with_disable_cpu_fallback().map_err(|e| anyhow::anyhow!("disable cpu fallback: {e}"))?
                    } else {
                        builder
                    }
                }
            };
            // On the CUDA path, the only ops still running on CPU are cheap
            // shape/indexing ones (Gather/Concat/Unsqueeze/Slice - ORT's own
            // deliberate placement, not a gap; see the CUDA-path note above)
            // in a single-sequence autoregressive loop where parallelizing
            // within one such tiny op buys nothing and only adds scheduling
            // overhead - pin to a small fixed thread count there. The CPU-EP
            // path is different: it runs the real INT4 matmuls and genuinely
            // benefits from ORT's default (physical-core-count) threading -
            // pinning it the same way measurably regressed RTF (3.48-4.42 ->
            // 4.63), so it keeps ORT's default.
            let mut builder = match ep {
                ExecutionProviderChoice::Cuda => {
                    builder.with_intra_threads(2).map_err(|e| anyhow::anyhow!("intra threads: {e}"))?
                }
                ExecutionProviderChoice::Cpu => builder,
            };
            builder
                .commit_from_file(model_dir.join(file))
                .map_err(|e| anyhow::anyhow!("commit {file}: {e}"))
        };

        let mut slow = build_session("slow_ar_int4.onnx")?;
        let mut fast = build_session("fast_ar_int4.onnx")?;
        let decoder = build_session("codec_decoder_fp16.onnx")?;

        let prompt_builder = PromptBuilder::new(
            &model_dir.join("tokenizer"),
            manifest.semantic_begin_id,
            manifest.num_codebooks,
        )?;

        let (fast_graph, slow_graph) = match ep {
            ExecutionProviderChoice::Cpu => (None, None),
            ExecutionProviderChoice::Cuda => (
                Some(Self::build_fast_graph_state(&mut fast, &manifest)?),
                Some(Self::build_slow_graph_state(&mut slow, &manifest)?),
            ),
        };

        Ok(Self { slow, fast, decoder, prompt_builder, manifest, fast_graph, slow_graph })
    }

    fn build_fast_graph_state(fast: &mut Session, manifest: &RuntimeManifest) -> anyhow::Result<FastGraphState> {
        let allocator = fast.allocator();

        let hidden = Tensor::<f16>::new(&allocator, [1usize, 1, 896]).map_err(|e| anyhow::anyhow!("alloc hidden: {e}"))?;
        let token_id = Tensor::<i64>::new(&allocator, [1usize, 1]).map_err(|e| anyhow::anyhow!("alloc token_id: {e}"))?;
        let use_hidden = Tensor::<bool>::new(&allocator, [1usize]).map_err(|e| anyhow::anyhow!("alloc use_hidden: {e}"))?;
        let input_pos = Tensor::<i64>::new(&allocator, [1usize]).map_err(|e| anyhow::anyhow!("alloc input_pos: {e}"))?;

        let cache_shape = [1usize, manifest.fast_n_local_heads, manifest.num_codebooks, manifest.fast_head_dim];
        let mut cache_keys = Vec::with_capacity(manifest.num_fast_layers);
        let mut cache_values = Vec::with_capacity(manifest.num_fast_layers);
        for _ in 0..manifest.num_fast_layers {
            cache_keys.push(Tensor::<f16>::new(&allocator, cache_shape).map_err(|e| anyhow::anyhow!("alloc cache_key: {e}"))?);
            cache_values.push(Tensor::<f16>::new(&allocator, cache_shape).map_err(|e| anyhow::anyhow!("alloc cache_value: {e}"))?);
        }

        let mut binding = fast.create_binding().map_err(|e| anyhow::anyhow!("create_binding: {e}"))?;
        binding.bind_input("slow_hidden", &hidden).map_err(|e| anyhow::anyhow!("bind slow_hidden: {e}"))?;
        binding.bind_input("token_id", &token_id).map_err(|e| anyhow::anyhow!("bind token_id: {e}"))?;
        binding.bind_input("use_slow_hidden", &use_hidden).map_err(|e| anyhow::anyhow!("bind use_slow_hidden: {e}"))?;
        binding.bind_input("input_pos", &input_pos).map_err(|e| anyhow::anyhow!("bind input_pos: {e}"))?;
        for i in 0..manifest.num_fast_layers {
            binding
                .bind_input(format!("cache_key_{i}"), &cache_keys[i])
                .map_err(|e| anyhow::anyhow!("bind cache_key_{i}: {e}"))?;
            binding
                .bind_input(format!("cache_value_{i}"), &cache_values[i])
                .map_err(|e| anyhow::anyhow!("bind cache_value_{i}: {e}"))?;
        }
        binding
            .bind_output_to_device("logits", &allocator.memory_info())
            .map_err(|e| anyhow::anyhow!("bind logits output: {e}"))?;
        for i in 0..manifest.num_fast_layers {
            binding
                .bind_output_to_device(format!("key_delta_{i}"), &allocator.memory_info())
                .map_err(|e| anyhow::anyhow!("bind key_delta_{i} output: {e}"))?;
            binding
                .bind_output_to_device(format!("value_delta_{i}"), &allocator.memory_info())
                .map_err(|e| anyhow::anyhow!("bind value_delta_{i} output: {e}"))?;
        }

        Ok(FastGraphState { binding, hidden, token_id, use_hidden, input_pos, cache_keys, cache_values })
    }

    /// Binds only the fixed T=1 decode-step shape - the initial variable-
    /// length prefill call never goes through this binding.
    fn build_slow_graph_state(slow: &mut Session, manifest: &RuntimeManifest) -> anyhow::Result<SlowGraphState> {
        let allocator = slow.allocator();

        let codes = Tensor::<i64>::new(&allocator, [1usize, manifest.num_codebooks + 1, 1])
            .map_err(|e| anyhow::anyhow!("alloc codes: {e}"))?;
        let input_pos = Tensor::<i64>::new(&allocator, [1usize]).map_err(|e| anyhow::anyhow!("alloc input_pos: {e}"))?;

        let cache_shape = [1usize, manifest.n_local_heads, manifest.max_seq_len, manifest.head_dim];
        let mut cache_keys = Vec::with_capacity(manifest.num_layers);
        let mut cache_values = Vec::with_capacity(manifest.num_layers);
        for _ in 0..manifest.num_layers {
            cache_keys.push(Tensor::<f16>::new(&allocator, cache_shape).map_err(|e| anyhow::anyhow!("alloc cache_key: {e}"))?);
            cache_values.push(Tensor::<f16>::new(&allocator, cache_shape).map_err(|e| anyhow::anyhow!("alloc cache_value: {e}"))?);
        }

        let mut binding = slow.create_binding().map_err(|e| anyhow::anyhow!("create_binding: {e}"))?;
        binding.bind_input("codes", &codes).map_err(|e| anyhow::anyhow!("bind codes: {e}"))?;
        binding.bind_input("input_pos", &input_pos).map_err(|e| anyhow::anyhow!("bind input_pos: {e}"))?;
        for i in 0..manifest.num_layers {
            binding
                .bind_input(format!("cache_key_{i}"), &cache_keys[i])
                .map_err(|e| anyhow::anyhow!("bind cache_key_{i}: {e}"))?;
            binding
                .bind_input(format!("cache_value_{i}"), &cache_values[i])
                .map_err(|e| anyhow::anyhow!("bind cache_value_{i}: {e}"))?;
        }
        binding
            .bind_output_to_device("logits", &allocator.memory_info())
            .map_err(|e| anyhow::anyhow!("bind logits output: {e}"))?;
        binding
            .bind_output_to_device("slow_hidden", &allocator.memory_info())
            .map_err(|e| anyhow::anyhow!("bind slow_hidden output: {e}"))?;
        for i in 0..manifest.num_layers {
            binding
                .bind_output_to_device(format!("key_delta_{i}"), &allocator.memory_info())
                .map_err(|e| anyhow::anyhow!("bind key_delta_{i} output: {e}"))?;
            binding
                .bind_output_to_device(format!("value_delta_{i}"), &allocator.memory_info())
                .map_err(|e| anyhow::anyhow!("bind value_delta_{i} output: {e}"))?;
        }

        Ok(SlowGraphState { binding, codes, input_pos, cache_keys, cache_values })
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
        if positions.len() == 1 {
            if let Some(graph) = self.slow_graph.as_mut() {
                return Self::slow_step_graph(&mut self.slow, graph, codes, positions[0], self.manifest.num_layers);
            }
        }

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

        // Prefill ran on the plain path (variable T); seed the persistent
        // graph-path cache tensors once here so the first T=1 decode call
        // that follows has correct history to build on.
        if let Some(graph) = self.slow_graph.as_mut() {
            for i in 0..self.manifest.num_layers {
                graph.cache_keys[i].extract_array_mut().assign(&caches[2 * i].view());
                graph.cache_values[i].extract_array_mut().assign(&caches[2 * i + 1].view());
            }
        }

        Ok((logits, hidden))
    }

    /// IoBinding-based slow_step for the fixed T=1 decode-step shape only.
    /// The KV cache lives ONLY in graph.cache_keys/cache_values (persistent
    /// across calls) - the host-side `caches` parameter is written once by
    /// the prefill call before entering this path and is never re-read or
    /// re-synced here, so the per-step cost is one small delta write instead
    /// of a full max_seq_len-length copy.
    fn slow_step_graph(
        slow: &mut Session,
        graph: &mut SlowGraphState,
        codes: &ArrayD<i64>,
        position: i64,
        num_layers: usize,
    ) -> anyhow::Result<(Vec<f32>, ArrayD<f16>)> {
        let t_prep = std::time::Instant::now();
        graph.codes.extract_array_mut().assign(&codes.view());
        graph.input_pos.extract_array_mut()[[0]] = position;
        let prep_us = t_prep.elapsed().as_micros();

        let t_sync = std::time::Instant::now();
        graph.binding.synchronize_inputs().map_err(|e| anyhow::anyhow!("sync inputs: {e}"))?;
        let sync_us = t_sync.elapsed().as_micros();

        let t_run = std::time::Instant::now();
        let outputs = slow.run_binding(&graph.binding).map_err(|e| anyhow::anyhow!("run_binding: {e}"))?;
        let run_us = t_run.elapsed().as_micros();

        let t_post = std::time::Instant::now();
        let logits_view = outputs["logits"].try_extract_array::<f32>()?;
        let hidden_view = outputs["slow_hidden"].try_extract_array::<f16>()?;

        let logits_shape = logits_view.shape();
        let last_t = logits_shape[1] - 1;
        let vocab = logits_shape[2];
        let logits: Vec<f32> = (0..vocab).map(|v| logits_view[[0, last_t, v]]).collect();

        let hidden_shape = hidden_view.shape();
        let hidden_dim = hidden_shape[2];
        let hidden_last_t = hidden_shape[1] - 1;
        let mut hidden = ArrayD::<f16>::from_elem(IxDyn(&[1, 1, hidden_dim]), f16::ZERO);
        for h in 0..hidden_dim {
            hidden[[0, 0, h]] = hidden_view[[0, hidden_last_t, h]];
        }

        for i in 0..num_layers {
            let key_delta = outputs[format!("key_delta_{i}")].try_extract_array::<f16>()?;
            let value_delta = outputs[format!("value_delta_{i}")].try_extract_array::<f16>()?;
            write_delta_into_tensor(&mut graph.cache_keys[i], &key_delta, position);
            write_delta_into_tensor(&mut graph.cache_values[i], &value_delta, position);
        }
        let post_us = t_post.elapsed().as_micros();
        if std::env::var("ARKTTS_TIMING").is_ok() {
            eprintln!("[timing] slow_step_graph prep={prep_us}us sync={sync_us}us run={run_us}us post={post_us}us");
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
        if let Some(graph) = self.fast_graph.as_mut() {
            if use_hidden {
                // Start of a new frame: reset the persistent device cache to
                // zero directly (not via a host round-trip) instead of
                // copying a freshly zeroed host array in every call.
                for i in 0..self.manifest.num_fast_layers {
                    graph.cache_keys[i].extract_array_mut().fill(f16::ZERO);
                    graph.cache_values[i].extract_array_mut().fill(f16::ZERO);
                }
            }
            return Self::fast_step_graph(&mut self.fast, graph, hidden, token_id, use_hidden, position, self.manifest.num_fast_layers);
        }

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

    /// IoBinding-based fast_step: writes into the SAME device tensors bound
    /// at setup (address-stable across calls, required for CUDA graph replay)
    /// instead of allocating+copying fresh tensors every call. The KV cache
    /// lives ONLY in graph.cache_keys/cache_values - callers must zero them
    /// (fast_step does this at use_hidden=true, the start of each frame)
    /// rather than pass a host-side cache array in.
    fn fast_step_graph(
        fast: &mut Session,
        graph: &mut FastGraphState,
        hidden: &ArrayD<f16>,
        token_id: i64,
        use_hidden: bool,
        position: i64,
        num_fast_layers: usize,
    ) -> anyhow::Result<Vec<f32>> {
        graph.hidden.extract_array_mut().assign(&hidden.view());
        graph.token_id.extract_array_mut()[[0, 0]] = token_id;
        graph.use_hidden.extract_array_mut()[[0]] = use_hidden;
        graph.input_pos.extract_array_mut()[[0]] = position;

        graph.binding.synchronize_inputs().map_err(|e| anyhow::anyhow!("sync inputs: {e}"))?;
        let outputs = fast.run_binding(&graph.binding).map_err(|e| anyhow::anyhow!("run_binding: {e}"))?;

        let logits_view = outputs["logits"].try_extract_array::<f32>()?;
        let logits_shape = logits_view.shape();
        let last_t = logits_shape[1] - 1;
        let vocab = logits_shape[2];
        let logits: Vec<f32> = (0..vocab).map(|v| logits_view[[0, last_t, v]]).collect();

        for i in 0..num_fast_layers {
            let key_delta = outputs[format!("key_delta_{i}")].try_extract_array::<f16>()?;
            let value_delta = outputs[format!("value_delta_{i}")].try_extract_array::<f16>()?;
            write_delta_into_tensor(&mut graph.cache_keys[i], &key_delta, position);
            write_delta_into_tensor(&mut graph.cache_values[i], &value_delta, position);
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

/// Same as update_cache_at_positions, but writes a single position's delta
/// directly into a persistent ort Tensor (the graph-path cache) instead of a
/// plain host array - avoids ever copying the whole cache tensor per step.
fn write_delta_into_tensor(cache: &mut Tensor<f16>, delta: &ndarray::ArrayViewD<f16>, position: i64) {
    let mut view = cache.extract_array_mut();
    let heads = view.shape()[1];
    let head_dim = view.shape()[3];
    for h in 0..heads {
        for d in 0..head_dim {
            view[[0, h, position as usize, d]] = delta[[0, h, 0, d]];
        }
    }
}
