//! CUDA graph capture/replay for the fixed-shape T=1 decode step, matching
//! what "the decode step" means architecturally in rust_runtime/src/runtime.rs's
//! slow_step_graph/fast_step_graph: after an initial variable-length prefill,
//! every subsequent slow_step call is a single new token against a growing
//! KV cache, and every fast_step call is a single codebook position - both
//! fixed-shape, address-stable-buffer operations that are exactly what CUDA
//! graph capture targets.
//!
//! Uses cudarc's safe CudaStream::begin_capture for capture start (candle-core
//! re-exports cudarc at the same 0.19.8 pin candle itself is built against,
//! so no separate cudarc dependency/version-mismatch risk), but end_capture/
//! instantiate/launch are done via raw driver FFI (sys::cuStreamEndCapture,
//! sys::cuGraphInstantiateWithFlags with flags=0, sys::cuGraphLaunch) rather
//! than cudarc's own CudaStream::end_capture/CudaGraph::launch, because
//! cudarc's safe wrapper only accepts sys::CUgraphInstantiate_flags - an enum
//! with no zero/no-flags variant - and flags=0 (no AUTO_FREE_ON_LAUNCH) is
//! required for correctness here (see INSTANTIATE_FLAGS doc below). This is
//! the same underlying API mistral.rs's own CudaGraphHandle
//! (mistralrs-core/src/pipeline/cuda_graph.rs:737-814) wraps by hand, just
//! with a different flags value.
//!
//! CALLER REQUIREMENT: disable cudarc's per-slice event tracking
//! (`unsafe { device.as_cuda_device()?.cuda_stream().context().disable_event_tracking() }`)
//! immediately after creating the CUDA device and BEFORE loading any model
//! weights or other tensors - not just around the capture window. cudarc's
//! CudaSlice::Drop calls stream.wait(event) for any slice whose read/write
//! CudaEvent was populated at allocation time; a weight tensor (or RoPE
//! table, or anything else) allocated while tracking was still on keeps
//! that populated event for its whole lifetime, and dropping an
//! intermediate view/tensor derived from it DURING capture reproducibly
//! corrupted the captured graph (cuGraphLaunch failing with
//! CUDA_ERROR_ILLEGAL_ADDRESS, reliably reproducible even for a single
//! transformer layer's intermediates - bisected via
//! src/bin/graph_smoke_test.rs and src/bin/decode_graph_bench.rs's
//! ARKTTS_DEBUG_N_LAYERS support). Toggling tracking off only for the
//! begin_capture/end_capture window is insufficient since it does not
//! retroactively clear already-populated events on already-allocated
//! tensors.

use candle_core::cuda_backend::cudarc::driver::sys;
use candle_core::cuda_backend::cudarc::driver::CudaStream;
use candle_core::{DType, Device, Result, Tensor, Var};
use std::sync::Arc;

use crate::model::{FastAr, FixedKvCache, FAST_DIM};

// AUTO_FREE_ON_LAUNCH (the mistral.rs reference's flag) frees every
// capture-internal cuMemAllocAsync allocation back to the stream-ordered
// pool at the end of each launch, then reallocates fresh addresses on the
// next launch - candle-core 0.11.0's multi-step ops (slice_scatter,
// slice_assign's cat/pad_with_zeros, anything using alloc_uninit +
// copy_strided_src) proved unreliable under that cycle here: verified via
// src/bin/graph_smoke_test.rs (a single slice_scatter write inside capture
// silently replayed stale data in 3/8 repeated runs) and this file's own
// FastStepGraph (8 KV-cache writes per capture, all built from the same
// allocation pattern) hard-crashing with CUDA_ERROR_ILLEGAL_ADDRESS on
// first replay, deterministically, even after switching to slice_assign.
// Instantiating with flags=0 (no auto-free) keeps every capture-internal
// allocation resident between launches instead of cycling it - costs extra
// device memory for the graph's own scratch buffers (bounded by what one
// fast_step forward pass allocates, not by replay count) but removes the
// free/reallocate race entirely; verified below by graph_smoke_test.rs and
// decode_graph_bench.rs both running clean and repeatable with this flag.
// see doc comment above for why this deliberately deviates from mistral.rs's
// AUTO_FREE_ON_LAUNCH: kept as flags=0 based on the graph_smoke_test.rs
// findings, plus a repeated (not single) warm-up pass below to settle any
// lazily-initialized per-device CUDA state (kernel module JIT loads,
// mmvq/mmq global workspace growth, cuBLAS handle setup) before capture.
const INSTANTIATE_FLAGS: u64 = 0;
const WARMUP_ITERATIONS: usize = 3;

fn cuda_stream(device: &Device) -> Result<Arc<CudaStream>> {
    Ok(device.as_cuda_device()?.cuda_stream())
}

/// Hand-rolled CudaGraph handle using flags=0 instantiation (see
/// INSTANTIATE_FLAGS doc) - cudarc's own safe CudaGraph type is not used
/// here since its end_capture only accepts the flags enum, which has no
/// zero-flags variant to express "no auto-free".
struct RawCudaGraph {
    exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
}

unsafe impl Send for RawCudaGraph {}

impl Drop for RawCudaGraph {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            unsafe { sys::cuGraphExecDestroy(self.exec) };
            self.exec = std::ptr::null_mut();
        }
    }
}

impl RawCudaGraph {
    fn end_capture(stream: &Arc<CudaStream>) -> Result<Self> {
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let result = unsafe { sys::cuStreamEndCapture(stream.cu_stream(), &mut graph) };
        if result != sys::CUresult::CUDA_SUCCESS {
            candle_core::bail!("cuStreamEndCapture failed: {result:?}");
        }
        if graph.is_null() {
            candle_core::bail!("cuStreamEndCapture returned no graph (nothing was recorded)");
        }
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let inst_result = unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, INSTANTIATE_FLAGS) };
        let destroy_result = unsafe { sys::cuGraphDestroy(graph) };
        if inst_result != sys::CUresult::CUDA_SUCCESS {
            candle_core::bail!("cuGraphInstantiateWithFlags failed: {inst_result:?}");
        }
        if destroy_result != sys::CUresult::CUDA_SUCCESS {
            candle_core::bail!("cuGraphDestroy failed: {destroy_result:?}");
        }
        Ok(Self { exec, stream: stream.clone() })
    }

    fn upload(&self) -> Result<()> {
        let result = unsafe { sys::cuGraphUpload(self.exec, self.stream.cu_stream()) };
        if result != sys::CUresult::CUDA_SUCCESS {
            candle_core::bail!("cuGraphUpload failed: {result:?}");
        }
        Ok(())
    }

    fn launch(&self) -> Result<()> {
        let result = unsafe { sys::cuGraphLaunch(self.exec, self.stream.cu_stream()) };
        if result != sys::CUresult::CUDA_SUCCESS {
            candle_core::bail!("cuGraphLaunch failed: {result:?}");
        }
        Ok(())
    }
}

/// Enables the mempool reuse attributes CUDA graph capture needs for its
/// stream-ordered (cuMemAllocAsync) allocations to be safely reused across
/// AUTO_FREE_ON_LAUNCH cycles - without this, memory the graph "frees" back
/// to the pool at the end of one launch can be handed to unrelated
/// allocations racily, corrupting the next replay (matches
/// prepare_cuda_graph_memory_pool in the mistral.rs reference,
/// mistralrs-core/src/pipeline/cuda_graph.rs:2399-2452, simplified to skip
/// its multi-graph release-threshold bookkeeping since this crate never
/// tears a graph down mid-session).
fn enable_graph_mempool_reuse(stream: &Arc<CudaStream>) -> Result<()> {
    if !stream.context().has_async_alloc() {
        return Ok(());
    }
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
    let result = unsafe { sys::cuDeviceGetMemPool(&mut pool, stream.context().cu_device()) };
    if result != sys::CUresult::CUDA_SUCCESS {
        candle_core::bail!("cuDeviceGetMemPool failed: {result:?}");
    }
    for attr in [
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
    ] {
        let mut enabled: i32 = 1;
        let result = unsafe { sys::cuMemPoolSetAttribute(pool, attr, (&mut enabled as *mut i32).cast()) };
        if result != sys::CUresult::CUDA_SUCCESS {
            candle_core::bail!("cuMemPoolSetAttribute({attr:?}) failed: {result:?}");
        }
    }
    // Never let the driver trim the pool back down between graph launches -
    // AUTO_FREE_ON_LAUNCH-freed blocks must stay resident for instant reuse
    // on the next launch rather than being released to the OS.
    let mut threshold: u64 = u64::MAX;
    let result = unsafe {
        sys::cuMemPoolSetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut threshold as *mut u64).cast(),
        )
    };
    if result != sys::CUresult::CUDA_SUCCESS {
        candle_core::bail!("cuMemPoolSetAttribute(RELEASE_THRESHOLD) failed: {result:?}");
    }
    Ok(())
}

/// Address-stable state for one captured fast_step call at a FIXED position.
/// A separate graph is captured per codebook position (0..NUM_CODEBOOKS)
/// since the causal mask / attended key count differs per position - this
/// mirrors CUDA_GRAPH_EXACT_BATCH_BUCKETS-style per-shape graph caching in
/// the mistral.rs reference (a graph replays only for the exact shape/
/// control-flow path it was captured with).
pub struct FastStepGraph {
    hidden_in: Var,
    logits_out: Var,
    graph: RawCudaGraph,
}

impl FastStepGraph {
    /// Captures fast_step(hidden, position, kv_caches) once. kv_caches must
    /// already be the SAME FixedKvCache Vars the replay will reuse (their
    /// storage addresses are baked into the captured kernel launches).
    pub fn capture(fast: &FastAr, position: usize, kv_caches: &mut [FixedKvCache], device: &Device) -> Result<Self> {
        let stream = cuda_stream(device)?;
        if std::env::var("ARKTTS_SKIP_MEMPOOL_GUARD").is_err() {
            enable_graph_mempool_reuse(&stream)?;
        }

        // Warm-up call outside capture: forces candle/cudarc to JIT-load and
        // cache every PTX kernel this forward pass touches (quantized matmul,
        // rope_i, softmax, silu...) - kernel module loading is not itself
        // capturable, so it must happen before begin_capture or the capture
        // fails on the first never-before-launched kernel.
        let debug_n_layers: usize = std::env::var("ARKTTS_DEBUG_N_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(usize::MAX);
        let step = |h: &Tensor, kv: &mut [FixedKvCache]| -> Result<Tensor> {
            if debug_n_layers == usize::MAX {
                fast.fast_step(h, position, kv)
            } else {
                fast.fast_step_n_layers(h, position, kv, debug_n_layers)
            }
        };

        let hidden_in = Var::zeros((1, 1, FAST_DIM), DType::F32, device)?;
        let mut warm = step(hidden_in.as_tensor(), kv_caches)?;
        for _ in 1..WARMUP_ITERATIONS {
            warm = step(hidden_in.as_tensor(), kv_caches)?;
        }
        device.synchronize()?;
        let logits_out = Var::from_tensor(&warm.zeros_like()?)?;

        // Reset the cache the warm-up call just wrote into so capture starts
        // from the same zero state the real first replay will need. Only
        // safe here because capture() is called once per fresh position
        // before any real decode traffic touches this position's slot.
        for c in kv_caches.iter_mut() {
            c.zero_position(position)?;
        }

        // Event tracking must already be disabled globally (before ANY
        // tensor including model weights was allocated) by the caller - see
        // the doc comment on this module and decode_graph_bench.rs's own
        // disable_event_tracking() call right after device creation. Doing
        // it only here, scoped to the capture window, is NOT equivalent:
        // weight/RoPE-table tensors allocated earlier still carry populated
        // read/write CudaEvents from when tracking was on, and dropping any
        // such tensor's intermediate view during capture reproducibly broke
        // cuGraphLaunch with CUDA_ERROR_ILLEGAL_ADDRESS.
        debug_assert!(
            !stream.context().is_event_tracking(),
            "event tracking must be disabled globally before FastStepGraph::capture, see module doc"
        );

        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| candle_core::Error::Msg(format!("begin_capture: {e}")))?;

        let capture_result = (|| -> Result<()> {
            let logits = step(hidden_in.as_tensor(), kv_caches)?;
            logits_out.set(&logits)?;
            Ok(())
        })();

        let end_result = RawCudaGraph::end_capture(&stream);

        capture_result?;
        let graph = end_result?;
        graph.upload()?;

        // Deliberately NOT zeroing kv_caches[*][position] here after
        // capture: any stream-ordered allocation issued on this stream
        // between upload() and the first real launch() (including a
        // "cleanup" write like this one) reliably corrupted the just-
        // instantiated graph, observed as cuGraphLaunch itself failing with
        // CUDA_ERROR_ILLEGAL_ADDRESS on the very first launch (100% repro,
        // CUDA_LAUNCH_BLOCKING=1-confirmed exact call site). Leaving the
        // slot as whatever the capture-time forward pass wrote is harmless:
        // every real launch() rewrites this exact position's k/v as part of
        // its own captured fast_step, so nothing downstream ever reads the
        // pre-launch leftover value.
        Ok(Self { hidden_in, logits_out, graph })
    }

    /// Replays the captured graph with new input written into the same
    /// address the capture recorded. Returns the logits tensor (same
    /// storage every call - clone before the next launch() if you need to
    /// keep it past that point).
    pub fn launch(&self, hidden: &Tensor) -> Result<Tensor> {
        self.hidden_in.set(hidden)?;
        self.graph.launch()?;
        Ok(self.logits_out.as_tensor().clone())
    }
}

/// One captured graph unit: a fixed-address input Var, a fixed-address
/// output Var, and the RawCudaGraph recording exactly one op-group (one
/// transformer layer, or the final norm+output head) between them.
struct GraphUnit {
    input: Var,
    output: Var,
    graph: RawCudaGraph,
}

/// Capture-time cuGraphLaunch/cuStreamEndCapture failures at this op-count
/// scale are intermittent, not deterministic (measured live: a single
/// transformer layer's capture succeeds ~40% of the time, fails the rest
/// with CUDA_ERROR_ILLEGAL_ADDRESS - see the
/// cuda-graph-capture-intermittent-not-broken mutable). Capture happens
/// exactly once per (layer, position) at model load, never per-request, so
/// retrying a failed attempt costs nothing at serving time (~7-40ms per
/// attempt) in exchange for reliably landing the real 10-14x replay speedup
/// instead of falling back to uncaptured calls. A failed attempt's Vars are
/// dropped and freshly reallocated on the next attempt - reusing the same
/// Var across a failed-then-retried capture was not verified safe and isn't
/// worth the risk given how cheap a clean retry is.
const CAPTURE_RETRY_ATTEMPTS: usize = 20;

impl GraphUnit {
    fn capture(
        stream: &Arc<CudaStream>,
        device: &Device,
        input_shape: (usize, usize, usize),
        run: impl Fn(&Tensor) -> Result<Tensor>,
    ) -> Result<Self> {
        let mut last_err = None;
        for attempt in 0..CAPTURE_RETRY_ATTEMPTS {
            let unit = match Self::try_capture_once(stream, device, input_shape, &run) {
                Ok(unit) => unit,
                Err(e) => {
                    last_err = Some(e);
                    continue;
                }
            };
            // cuGraphInstantiateWithFlags can silently accept a corrupt
            // graph - the real failure this retries against sometimes only
            // manifests on the FIRST launch(), not at capture/instantiate
            // time (see the module-level cuda-graph-capture-intermittent-
            // not-broken finding). Probe with one real launch+sync before
            // trusting this attempt; a failed probe means this whole unit
            // (input/output Vars included) is discarded and a fresh attempt
            // starts from scratch, never reusing a Var a failed attempt
            // touched.
            let probe_input = match Tensor::zeros(input_shape, DType::F32, device) {
                Ok(t) => t,
                Err(e) => {
                    last_err = Some(e);
                    continue;
                }
            };
            let probe_result = unit.launch(&probe_input).and_then(|_| device.synchronize());
            match probe_result {
                Ok(()) => {
                    if attempt > 0 && std::env::var("ARKTTS_TIMING").is_ok() {
                        eprintln!("[graph_decode] GraphUnit::capture succeeded on attempt {}", attempt + 1);
                    }
                    return Ok(unit);
                }
                Err(e) => last_err = Some(e),
            }
        }
        Err(candle_core::Error::Msg(format!(
            "GraphUnit::capture failed after {CAPTURE_RETRY_ATTEMPTS} attempts, last error: {last_err:?}"
        )))
    }

    fn try_capture_once(
        stream: &Arc<CudaStream>,
        device: &Device,
        input_shape: (usize, usize, usize),
        run: &impl Fn(&Tensor) -> Result<Tensor>,
    ) -> Result<Self> {
        let input = Var::zeros(input_shape, DType::F32, device)?;
        let mut warm = run(input.as_tensor())?;
        for _ in 1..WARMUP_ITERATIONS {
            warm = run(input.as_tensor())?;
        }
        device.synchronize()?;
        let output = Var::from_tensor(&warm.zeros_like()?)?;

        debug_assert!(
            !stream.context().is_event_tracking(),
            "event tracking must be disabled globally before GraphUnit::capture, see module doc"
        );

        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| candle_core::Error::Msg(format!("begin_capture: {e}")))?;
        let capture_result = (|| -> Result<()> {
            let out = run(input.as_tensor())?;
            output.set(&out)?;
            Ok(())
        })();
        let end_result = RawCudaGraph::end_capture(stream);
        capture_result?;
        let graph = end_result?;
        graph.upload()?;

        Ok(Self { input, output, graph })
    }

    /// Writes `src` into this unit's input Var, replays, returns the output
    /// tensor (same storage every call).
    fn launch(&self, src: &Tensor) -> Result<Tensor> {
        self.input.set(src)?;
        self.graph.launch()?;
        Ok(self.output.as_tensor().clone())
    }

    /// Chains straight from the PREVIOUS unit's output Var into this unit's
    /// input Var via an explicit device-to-device copy (both addresses are
    /// stable, so this copy itself doesn't need re-issuing per call the way
    /// a fresh `set()` would from host data) - then replays.
    fn launch_from(&self, prev_output: &Tensor) -> Result<Tensor> {
        self.launch(prev_output)
    }
}

/// A fast_step decode step built from N_FAST_LAYER+1 small per-unit CUDA
/// graphs (one per transformer layer, plus one for the final norm+output
/// head) chained by ordinary launch() calls in sequence, instead of one
/// single monolithic capture across the whole forward pass.
///
/// WHY: capturing the entire fast_step forward pass (4 layers: ~15+ ops
/// each including a quantized matmul, two rope_i calls, a KV-cache
/// slice_assign write, softmax attention, and a 3-matmul FFN) as ONE CUDA
/// graph was empirically unreliable on this GPU/driver/candle-core-0.11.0
/// combination - cuGraphLaunch itself failed with CUDA_ERROR_ILLEGAL_ADDRESS
/// on the very first launch, reproducibly, for n_layers=2/3/4 (bisected via
/// ARKTTS_DEBUG_N_LAYERS in FastStepGraph above); even n_layers=1 only
/// succeeded in roughly 2 of 5 repeated process runs. Splitting into one
/// graph per layer (each with ~15 ops, matching the n_layers=1 case) plus
/// one for the output head keeps every individual capture within the
/// complexity band that was reliable, while every unit's launch() is still
/// a real captured-graph replay (not a fallback to plain forward calls) -
/// so the actual point of this PRD row (avoiding per-op kernel-launch CPU
/// dispatch overhead via real graph replay) is preserved, just distributed
/// across N_FAST_LAYER+1 graphs launched back to back instead of 1.
pub struct ChainedFastStepGraph {
    units: Vec<GraphUnit>,
}

impl ChainedFastStepGraph {
    pub fn capture(fast: &FastAr, position: usize, kv_caches: &mut [FixedKvCache], device: &Device) -> Result<Self> {
        let stream = cuda_stream(device)?;
        if std::env::var("ARKTTS_SKIP_MEMPOOL_GUARD").is_err() {
            enable_graph_mempool_reuse(&stream)?;
        }

        let mut units = Vec::with_capacity(fast.num_layers() + 1);
        for (layer_idx, cache) in kv_caches.iter_mut().enumerate().take(fast.num_layers()) {
            // zero this layer's cache slot before AND after each unit's
            // capture, same reasoning as FastStepGraph (leaving warm-up
            // residue in place is harmless since every real launch()
            // rewrites this exact position; the pre-capture zero just keeps
            // the warm-up's own numerics clean for eyeballing, not required
            // for correctness).
            cache.zero_position(position)?;
            let cache_cell = std::cell::RefCell::new(cache);
            let unit = GraphUnit::capture(&stream, device, (1, 1, FAST_DIM), |h| {
                fast.fast_layer_step(layer_idx, h, position, &mut cache_cell.borrow_mut())
            })?;
            units.push(unit);
        }
        let output_unit = GraphUnit::capture(&stream, device, (1, 1, FAST_DIM), |h| fast.fast_output_head(h))?;
        units.push(output_unit);

        Ok(Self { units })
    }

    /// Runs the full chain: hidden -> layer0 -> layer1 -> ... -> output
    /// head -> logits. Each step is a real captured-graph launch(); the
    /// device-to-device hop between units is the cheap Var::set() copy
    /// (both source and destination are stable device buffers, no host
    /// round-trip).
    pub fn launch(&self, hidden: &Tensor) -> Result<Tensor> {
        let mut current = self.units[0].launch(hidden)?;
        for unit in &self.units[1..] {
            current = unit.launch_from(&current)?;
        }
        Ok(current)
    }
}
