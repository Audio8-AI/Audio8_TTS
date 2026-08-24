//! candle-decode-loop-cuda-graph verification: captures a real CUDA graph
//! for FastAr::fast_step at a fixed codebook position, then replays it many
//! times, printing per-call wall-clock timing (ARKTTS_TIMING-gated, same
//! pattern as rust_runtime/src/runtime.rs's slow_step_graph) to show the
//! genuine warm-up-then-flat-fast signature: capture (including the
//! mandatory warm-up + cuStreamEndCapture/cuGraphInstantiateWithFlags cost)
//! is slow, every subsequent launch() replays the exact recorded kernel
//! graph and should be far faster than a plain (uncaptured) forward call at
//! the same shape.
//!
//! CUDA-only: requires --features cuda and a real GPU (this project's RTX
//! 3060 Laptop). Not meaningful on CPU - graph capture is a CUDA driver
//! mechanism with no CPU-backend equivalent.

use audio8_candle_runtime::graph_decode::FastStepGraph;
use audio8_candle_runtime::model::{FastAr, FAST_DIM};
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    if std::env::var("ARKTTS_FORCE_DMMV").is_ok() {
        candle_core::quantized::cuda::set_force_dmmv(true);
        println!("[decode_graph_bench] FORCE_DMMV enabled (diagnostic: disables fast_mmvq's global scratch workspace)");
    }
    let device = Device::new_cuda(0)?;
    println!("[decode_graph_bench] device: {:?}", device);
    {
        let stream = device.as_cuda_device()?.cuda_stream();
        println!("[decode_graph_bench] has_async_alloc={}", stream.context().has_async_alloc());
        // Disabled globally, before any tensor (including model weights) is
        // allocated: CudaSlice::drop calls stream.wait(event) for any slice
        // whose read/write CudaEvent was populated at allocation time (i.e.
        // any tensor allocated while event tracking was on) - dropping such
        // a tensor DURING capture showed up as cuGraphLaunch itself failing
        // with CUDA_ERROR_ILLEGAL_ADDRESS, reproducibly, even for a single
        // transformer layer's worth of intermediate tensors. Single-stream
        // usage here (is_in_multi_stream_mode() is always false) means this
        // tracking was never doing anything useful for us regardless.
        unsafe { stream.context().disable_event_tracking() };
    }

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fast = FastAr::load(&root.join("weights/fast_ar_q4_0.gguf"), &device)?;

    let timing = std::env::var("ARKTTS_TIMING").is_ok();
    let n_replays: usize = std::env::var("ARKTTS_REPLAYS").ok().and_then(|s| s.parse().ok()).unwrap_or(200);

    let position = 3usize; // an interior codebook position, arbitrary but fixed
    let mut kv_caches = fast.new_kv_caches(1)?;
    FastAr::reset_caches(&mut kv_caches)?;

    // --- Baseline: plain (uncaptured) forward calls at the same shape ---
    let hidden = Tensor::zeros((1, 1, FAST_DIM), DType::F32, &device)?;
    // warm up kernels/JIT before timing the baseline for a fair comparison
    for _ in 0..5 {
        let _ = fast.fast_step(&hidden, position, &mut kv_caches)?;
    }
    device.synchronize()?;

    let mut baseline_us = Vec::with_capacity(n_replays);
    for i in 0..n_replays {
        let t = Instant::now();
        let out = fast.fast_step(&hidden, position, &mut kv_caches)?;
        device.synchronize()?;
        let us = t.elapsed().as_micros();
        baseline_us.push(us);
        if timing {
            eprintln!("[timing] baseline (uncaptured) call {i} = {us}us");
        }
        std::hint::black_box(&out);
    }
    let baseline_mean: f64 = baseline_us.iter().sum::<u128>() as f64 / baseline_us.len() as f64;
    let baseline_median = {
        let mut v = baseline_us.clone();
        v.sort();
        v[v.len() / 2]
    };

    // --- Correctness reference: plain forward on fresh caches, computed and
    // read back to host BEFORE any graph capture exists, so this allocation
    // can never interact with a live captured graph's own device memory. ---
    let mut correctness_caches = fast.new_kv_caches(1)?;
    FastAr::reset_caches(&mut correctness_caches)?;
    let plain_check = fast.fast_step(&hidden, position, &mut correctness_caches)?;
    device.synchronize()?;
    let plain_vec: Vec<f32> = plain_check.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
    let plain_argmax = plain_vec.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    drop(correctness_caches);

    // --- CUDA graph capture + replay ---
    FastAr::reset_caches(&mut kv_caches)?;
    let t_capture = Instant::now();
    let graph = FastStepGraph::capture(&fast, position, &mut kv_caches, &device)?;
    device.synchronize()?;
    let capture_us = t_capture.elapsed().as_micros();
    println!("[decode_graph_bench] capture took {capture_us}us (includes warm-up forward + cuStreamEndCapture + cuGraphInstantiateWithFlags + upload)");

    // First launch, read back immediately, compare against the plain
    // reference computed above - this is the actual correctness evidence;
    // it happens before the timing loop so nothing about it can be blamed
    // on accumulated replay state.
    let first_launch_out = graph.launch(&hidden)?;
    device.synchronize()?;
    let graph_vec: Vec<f32> = first_launch_out.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
    let graph_argmax = graph_vec.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    let diff: f32 = plain_vec.iter().zip(graph_vec.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("\n[correctness] plain forward argmax={plain_argmax} graph replay (1st launch) argmax={graph_argmax} -> {}", if plain_argmax == graph_argmax { "MATCH" } else { "DIFFER" });
    println!("[correctness] max abs logit diff (plain vs graph 1st launch, both fresh caches, same input): {diff:.6}");

    let mut replay_us = Vec::with_capacity(n_replays);
    for i in 0..n_replays {
        let t = Instant::now();
        let out = graph.launch(&hidden)?;
        device.synchronize()?;
        let us = t.elapsed().as_micros();
        replay_us.push(us);
        if timing {
            eprintln!("[timing] graph replay call {i} = {us}us");
        }
        if std::env::var("ARKTTS_NAN_CHECK").is_ok() {
            let v: Vec<f32> = out.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
            let nan_count = v.iter().filter(|x| x.is_nan()).count();
            if nan_count > 0 {
                eprintln!("[nan_check] replay {i}: {nan_count} NaN values, first few: {:?}", &v[..5.min(v.len())]);
                anyhow::bail!("NaN detected at replay {i}");
            }
        }
        std::hint::black_box(&out);
    }
    let replay_mean: f64 = replay_us.iter().sum::<u128>() as f64 / replay_us.len() as f64;
    let replay_median = {
        let mut v = replay_us.clone();
        v.sort();
        v[v.len() / 2]
    };
    let first_replay = replay_us[0];
    let steady_state_mean: f64 = replay_us[10..].iter().sum::<u128>() as f64 / (replay_us.len() - 10) as f64;

    println!("\n=== SUMMARY ===");
    println!("baseline (uncaptured) forward: mean={baseline_mean:.1}us median={baseline_median}us over {n_replays} calls");
    println!("graph capture (one-time cost): {capture_us}us");
    println!("graph replay: mean={replay_mean:.1}us median={replay_median}us first_replay={first_replay}us steady_state_mean(skip first 10)={steady_state_mean:.1}us over {n_replays} calls");
    println!(
        "speedup (baseline_median / replay_steady_state_mean): {:.2}x",
        baseline_median as f64 / steady_state_mean
    );
    println!(
        "capture overhead vs one replay: {:.1}x (capture_us / replay_median)",
        capture_us as f64 / replay_median as f64
    );

    Ok(())
}
