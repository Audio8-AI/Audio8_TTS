//! Verifies ChainedFastStepGraph: per-layer CUDA graph capture (each layer's
//! ~15-op forward pass captured as its own small graph, with retry-until-
//! success inside GraphUnit::capture) chained together at replay time,
//! instead of one monolithic capture across the whole 4-layer forward pass.
//! See graph_decode.rs's module doc and the cuda-graph-capture-intermittent-
//! not-broken finding for why: single-graph capture at full model
//! complexity failed 0/10 times, but a single layer's capture succeeds
//! ~40% of the time - retried until success (cheap, since capture happens
//! once per position at load, never per-request), this should reliably
//! land the full speedup via 5 small graphs (4 layers + output head)
//! instead of 1 large one.

use audio8_candle_runtime::graph_decode::ChainedFastStepGraph;
use audio8_candle_runtime::model::{FastAr, FAST_DIM};
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    println!("[chained_decode_bench] device: {:?}", device);
    {
        let stream = device.as_cuda_device()?.cuda_stream();
        unsafe { stream.context().disable_event_tracking() };
    }

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fast = FastAr::load(&root.join("weights/fast_ar_q4_0.gguf"), &device)?;

    let n_replays: usize = std::env::var("ARKTTS_REPLAYS").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let position = 3usize;

    let mut kv_caches = fast.new_kv_caches(1)?;
    FastAr::reset_caches(&mut kv_caches)?;

    let hidden = Tensor::zeros((1, 1, FAST_DIM), DType::F32, &device)?;
    for _ in 0..5 {
        let _ = fast.fast_step(&hidden, position, &mut kv_caches)?;
    }
    device.synchronize()?;

    let mut baseline_us = Vec::with_capacity(n_replays);
    for _ in 0..n_replays {
        let t = Instant::now();
        let out = fast.fast_step(&hidden, position, &mut kv_caches)?;
        device.synchronize()?;
        baseline_us.push(t.elapsed().as_micros());
        std::hint::black_box(&out);
    }
    let baseline_median = {
        let mut v = baseline_us.clone();
        v.sort();
        v[v.len() / 2]
    };

    let mut correctness_caches = fast.new_kv_caches(1)?;
    FastAr::reset_caches(&mut correctness_caches)?;
    let plain_check = fast.fast_step(&hidden, position, &mut correctness_caches)?;
    device.synchronize()?;
    let plain_vec: Vec<f32> = plain_check.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
    let plain_argmax = plain_vec.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    drop(correctness_caches);

    FastAr::reset_caches(&mut kv_caches)?;
    let t_capture = Instant::now();
    let graph = ChainedFastStepGraph::capture(&fast, position, &mut kv_caches, &device)?;
    device.synchronize()?;
    let capture_us = t_capture.elapsed().as_micros();
    println!("[chained_decode_bench] chained capture (all layers + head) took {capture_us}us");

    let first_launch_out = graph.launch(&hidden)?;
    device.synchronize()?;
    let graph_vec: Vec<f32> = first_launch_out.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
    let graph_argmax = graph_vec.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    let diff: f32 = plain_vec.iter().zip(graph_vec.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("[correctness] plain argmax={plain_argmax} chained-graph argmax={graph_argmax} -> {}", if plain_argmax == graph_argmax { "MATCH" } else { "DIFFER" });
    println!("[correctness] max abs diff: {diff:.6}");

    let mut replay_us = Vec::with_capacity(n_replays);
    for _ in 0..n_replays {
        let t = Instant::now();
        let out = graph.launch(&hidden)?;
        device.synchronize()?;
        replay_us.push(t.elapsed().as_micros());
        std::hint::black_box(&out);
    }
    let replay_median = {
        let mut v = replay_us.clone();
        v.sort();
        v[v.len() / 2]
    };
    let steady_state_mean: f64 = replay_us[10..].iter().sum::<u128>() as f64 / (replay_us.len() - 10) as f64;

    println!("\n=== SUMMARY ===");
    println!("baseline (uncaptured) median: {baseline_median}us");
    println!("chained-graph replay median: {replay_median}us steady_state_mean: {steady_state_mean:.1}us");
    println!("speedup (baseline_median / replay_steady_state_mean): {:.2}x", baseline_median as f64 / steady_state_mean);

    Ok(())
}
