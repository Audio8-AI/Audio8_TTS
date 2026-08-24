//! Minimal CUDA graph capture/replay smoke test, mirroring the mistral.rs
//! reference's own test (mistralrs-core/src/pipeline/cuda_graph.rs:3908-3924)
//! almost verbatim: capture a single affine op on a Var, replay it with a
//! new input, verify the output. Used to isolate whether a bug is in the
//! basic CUDA graph capture mechanism itself vs in the DualAR model's
//! forward pass when graph_decode.rs's FastStepGraph produces NaN.

use candle_core::cuda_backend::cudarc::driver::sys;
use candle_core::{DType, Device, Tensor, Var};

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    println!("device: {:?}", device);
    let stream = device.as_cuda_device()?.cuda_stream();

    // Same mempool reuse guard as graph_decode.rs::enable_graph_mempool_reuse -
    // without this, AUTO_FREE_ON_LAUNCH-freed graph-internal allocations can
    // race with other stream-ordered allocations between launches.
    if stream.context().has_async_alloc() {
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        unsafe { sys::cuDeviceGetMemPool(&mut pool, stream.context().cu_device()) };
        for attr in [
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
        ] {
            let mut enabled: i32 = 1;
            unsafe { sys::cuMemPoolSetAttribute(pool, attr, (&mut enabled as *mut i32).cast()) };
        }
        let mut threshold: u64 = u64::MAX;
        unsafe {
            sys::cuMemPoolSetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut threshold as *mut u64).cast(),
            )
        };
        println!("mempool reuse guard enabled");
    }

    let input = Var::from_tensor(&Tensor::from_vec(vec![1f32, 2.0], 2, &device)?)?;

    // warm-up (not captured)
    let warm = input.as_tensor().affine(2.0, 1.0)?;
    device.synchronize()?;
    println!("warm-up output: {:?}", warm.to_vec1::<f32>()?);

    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
    let output = input.as_tensor().affine(2.0, 1.0)?;
    let graph = stream
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?
        .expect("capture returned no graph");
    graph.upload()?;

    graph.launch()?;
    device.synchronize()?;
    let v1: Vec<f32> = output.to_dtype(DType::F32)?.to_vec1()?;
    println!("replay 1 (input=[1,2], expect [3,5]): {:?}", v1);

    input.set(&Tensor::from_vec(vec![3f32, 5.0], 2, &device)?)?;
    graph.launch()?;
    device.synchronize()?;
    let v2: Vec<f32> = output.to_dtype(DType::F32)?.to_vec1()?;
    println!("replay 2 (input=[3,5], expect [7,11]): {:?}", v2);

    for i in 0..20 {
        input.set(&Tensor::from_vec(vec![i as f32, (i * 2) as f32], 2, &device)?)?;
        graph.launch()?;
        device.synchronize()?;
        let v: Vec<f32> = output.to_dtype(DType::F32)?.to_vec1()?;
        let expect = vec![i as f32 * 2.0 + 1.0, (i * 2) as f32 * 2.0 + 1.0];
        println!("replay {}: input=[{},{}] got={:?} expect={:?} match={}", i + 3, i, i * 2, v, expect, v == expect);
    }

    println!("\n=== test 2: multi-Var capture with slice_scatter (mimics KV cache write) ===");
    let cache = Var::zeros((1, 2, 8, 4), DType::F32, &device)?; // [B,H,max_len,D]
    let new_kv_in = Var::zeros((1, 2, 1, 4), DType::F32, &device)?;
    let out2 = Var::zeros((1, 2, 8, 4), DType::F32, &device)?;
    let position = 3usize;

    // warm-up
    {
        let new_kv = new_kv_in.as_tensor();
        let updated = cache.as_tensor().slice_scatter(new_kv, 2, position)?;
        cache.set(&updated)?;
        out2.set(cache.as_tensor())?;
    }
    device.synchronize()?;
    // reset cache to zero before capturing so capture starts clean
    cache.set(&Tensor::zeros((1, 2, 8, 4), DType::F32, &device)?)?;

    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
    let capture_result: anyhow::Result<()> = (|| {
        let new_kv = new_kv_in.as_tensor();
        let updated = cache.as_tensor().slice_scatter(new_kv, 2, position)?;
        cache.set(&updated)?;
        out2.set(cache.as_tensor())?;
        Ok(())
    })();
    let graph2 = stream
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?
        .expect("capture 2 returned no graph");
    capture_result?;
    graph2.upload()?;

    for i in 0..5 {
        let val = (i + 1) as f32;
        new_kv_in.set(&Tensor::from_vec(vec![val; 8], (1, 2, 1, 4), &device)?)?;
        graph2.launch()?;
        device.synchronize()?;
        let v: Vec<f32> = out2.as_tensor().flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
        let nan_count = v.iter().filter(|x| x.is_nan()).count();
        // shape [1,2,8,4]: head0 at offset 0..32, head1 at offset 32..64.
        // position 3 within head0 is elements [12..16), within head1 [44..48).
        let pos3_head0 = &v[12..16];
        let pos3_head1 = &v[44..48];
        println!("test2 replay {i}: nan_count={nan_count} pos3_head0={:?} pos3_head1={:?} expect_all={val}", pos3_head0, pos3_head1);
    }

    println!("\n=== test 3: two chained Vars WITHOUT slice_scatter (plain affine chain) ===");
    let a = Var::zeros(4, DType::F32, &device)?;
    let b = Var::zeros(4, DType::F32, &device)?;
    let a_in = Var::zeros(4, DType::F32, &device)?;
    {
        let updated = a_in.as_tensor().affine(1.0, 0.0)?;
        a.set(&updated)?;
        let updated_b = a.as_tensor().affine(10.0, 0.0)?;
        b.set(&updated_b)?;
    }
    device.synchronize()?;
    a.set(&Tensor::zeros(4, DType::F32, &device)?)?;
    b.set(&Tensor::zeros(4, DType::F32, &device)?)?;

    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
    let cap3: anyhow::Result<()> = (|| {
        let updated = a_in.as_tensor().affine(1.0, 0.0)?;
        a.set(&updated)?;
        let updated_b = a.as_tensor().affine(10.0, 0.0)?;
        b.set(&updated_b)?;
        Ok(())
    })();
    let graph3 = stream
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?
        .expect("capture 3 returned no graph");
    cap3?;
    graph3.upload()?;

    for i in 0..3 {
        let val = (i + 1) as f32;
        a_in.set(&Tensor::from_vec(vec![val; 4], 4, &device)?)?;
        graph3.launch()?;
        device.synchronize()?;
        let av: Vec<f32> = a.as_tensor().to_dtype(DType::F32)?.to_vec1()?;
        let bv: Vec<f32> = b.as_tensor().to_dtype(DType::F32)?.to_vec1()?;
        println!("test3 replay {i}: input={val} a={:?} (expect {val}) b={:?} (expect {})", av, bv, val * 10.0);
    }

    println!("\n=== test 4: slice_scatter on dim=0 (no transpose path) ===");
    let cache0 = Var::zeros((8, 4), DType::F32, &device)?;
    let new_in0 = Var::zeros((1, 4), DType::F32, &device)?;
    let out0 = Var::zeros((8, 4), DType::F32, &device)?;
    {
        let updated = cache0.as_tensor().slice_scatter(new_in0.as_tensor(), 0, 3)?;
        cache0.set(&updated)?;
        out0.set(cache0.as_tensor())?;
    }
    device.synchronize()?;
    cache0.set(&Tensor::zeros((8, 4), DType::F32, &device)?)?;

    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
    let cap4: anyhow::Result<()> = (|| {
        let updated = cache0.as_tensor().slice_scatter(new_in0.as_tensor(), 0, 3)?;
        cache0.set(&updated)?;
        out0.set(cache0.as_tensor())?;
        Ok(())
    })();
    let graph4 = stream
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?
        .expect("capture 4 returned no graph");
    cap4?;
    graph4.upload()?;

    for i in 0..3 {
        let val = (i + 1) as f32;
        new_in0.set(&Tensor::from_vec(vec![val; 4], (1, 4), &device)?)?;
        graph4.launch()?;
        device.synchronize()?;
        let v: Vec<f32> = out0.as_tensor().flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
        println!("test4 replay {i}: input={val} row3={:?} (expect [{val},{val},{val},{val}])", &v[12..16]);
    }

    println!("\n=== test 5: slice_assign instead of slice_scatter (dim=2, mimics KV cache) ===");
    let cache5 = Var::zeros((1, 2, 8, 4), DType::F32, &device)?;
    let new_kv5 = Var::zeros((1, 2, 1, 4), DType::F32, &device)?;
    let out5 = Var::zeros((1, 2, 8, 4), DType::F32, &device)?;
    let pos5 = 3usize;
    {
        let updated = cache5.as_tensor().slice_assign(&[0..1, 0..2, pos5..pos5 + 1, 0..4], new_kv5.as_tensor())?;
        cache5.set(&updated)?;
        out5.set(cache5.as_tensor())?;
    }
    device.synchronize()?;
    cache5.set(&Tensor::zeros((1, 2, 8, 4), DType::F32, &device)?)?;

    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
    let cap5: anyhow::Result<()> = (|| {
        let updated = cache5.as_tensor().slice_assign(&[0..1, 0..2, pos5..pos5 + 1, 0..4], new_kv5.as_tensor())?;
        cache5.set(&updated)?;
        out5.set(cache5.as_tensor())?;
        Ok(())
    })();
    let graph5 = stream
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?
        .expect("capture 5 returned no graph");
    cap5?;
    graph5.upload()?;

    for i in 0..5 {
        let val = (i + 1) as f32;
        new_kv5.set(&Tensor::from_vec(vec![val; 8], (1, 2, 1, 4), &device)?)?;
        graph5.launch()?;
        device.synchronize()?;
        let v: Vec<f32> = out5.as_tensor().flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
        let pos3_head0 = &v[12..16];
        let pos3_head1 = &v[44..48];
        println!("test5 replay {i}: input={val} pos3_head0={:?} pos3_head1={:?} expect_all={val}", pos3_head0, pos3_head1);
    }

    println!("\n=== test 6: rope_i custom op inside capture ===");
    use candle_nn::rotary_emb::rope_i;
    let q_in = Var::zeros((1, 2, 1, 8), DType::F32, &device)?; // [B,H,T,D]
    let cos6 = Tensor::from_vec(vec![1f32; 4], (1, 4), &device)?; // [T, D/2]
    let sin6 = Tensor::from_vec(vec![0f32; 4], (1, 4), &device)?;
    let rope_out = Var::zeros((1, 2, 1, 8), DType::F32, &device)?;
    {
        let r = rope_i(&q_in.as_tensor().contiguous()?, &cos6, &sin6)?;
        rope_out.set(&r)?;
    }
    device.synchronize()?;
    rope_out.set(&Tensor::zeros((1, 2, 1, 8), DType::F32, &device)?)?;

    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
    let cap6: anyhow::Result<()> = (|| {
        let r = rope_i(&q_in.as_tensor().contiguous()?, &cos6, &sin6)?;
        rope_out.set(&r)?;
        Ok(())
    })();
    let graph6 = stream
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?
        .expect("capture 6 returned no graph");
    cap6?;
    graph6.upload()?;

    for i in 0..3 {
        let val = (i + 1) as f32;
        q_in.set(&Tensor::from_vec(vec![val; 16], (1, 2, 1, 8), &device)?)?;
        graph6.launch()?;
        device.synchronize()?;
        let v: Vec<f32> = rope_out.as_tensor().flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
        let nan_count = v.iter().filter(|x| x.is_nan()).count();
        println!("test6 replay {i}: input={val} nan_count={nan_count} output={:?}", v);
    }

    println!("\n=== test 7: 4-layer chain of (matmul + rope_i + slice_assign) mimicking fast_step depth ===");
    let hidden7 = Var::zeros((1, 1, 16), DType::F32, &device)?;
    let w7 = Tensor::rand(0f32, 1f32, (16, 16), &device)?;
    let cache7 = Var::zeros((1, 2, 8, 8), DType::F32, &device)?;
    let out7 = Var::zeros((1, 1, 16), DType::F32, &device)?;
    let cos7 = Tensor::from_vec(vec![1f32; 2], (1, 2), &device)?;
    let sin7 = Tensor::from_vec(vec![0f32; 2], (1, 2), &device)?;
    let pos7 = 3usize;
    let run7 = |hidden_in: &Tensor| -> anyhow::Result<Tensor> {
        let mut h = hidden_in.clone();
        for _layer in 0..4 {
            let qkv = h.broadcast_matmul(&w7.t()?)?; // [1,1,16]
            let q = qkv.narrow(2, 0, 8)?.reshape((1, 1, 2, 4))?.transpose(1, 2)?.contiguous()?;
            let k = qkv.narrow(2, 8, 8)?.reshape((1, 1, 2, 4))?.transpose(1, 2)?.contiguous()?;
            let q = rope_i(&q, &cos7, &sin7)?;
            let k = rope_i(&k, &cos7, &sin7)?;
            let ranges = [0..1usize, 0..2usize, pos7..pos7 + 1, 0..4usize];
            let k_new = cache7.as_tensor().slice_assign(&ranges, &k)?;
            cache7.set(&k_new)?;
            let v_read = cache7.as_tensor().narrow(2, 0, pos7 + 1)?;
            let attn = q.matmul(&v_read.transpose(2, 3)?.contiguous()?)?; // [1,2,1,pos7+1]
            let summed = attn.sum(3)?; // [1,2,1]
            let flat = summed.reshape((1, 1, 2))?;
            h = flat.broadcast_as((1, 1, 16))?.contiguous()?;
        }
        Ok(h)
    };
    let _ = run7(hidden7.as_tensor())?;
    device.synchronize()?;
    cache7.set(&Tensor::zeros((1, 2, 8, 8), DType::F32, &device)?)?;

    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
    let cap7: anyhow::Result<()> = (|| {
        let r = run7(hidden7.as_tensor())?;
        out7.set(&r)?;
        Ok(())
    })();
    let graph7 = stream
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?
        .expect("capture 7 returned no graph");
    cap7?;
    graph7.upload()?;

    for i in 0..5 {
        let val = (i + 1) as f32;
        hidden7.set(&Tensor::from_vec(vec![val; 16], (1, 1, 16), &device)?)?;
        graph7.launch()?;
        device.synchronize()?;
        let v: Vec<f32> = out7.as_tensor().flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
        let nan_count = v.iter().filter(|x| x.is_nan()).count();
        println!("test7 replay {i}: input={val} nan_count={nan_count} sample={:?}", &v[..4]);
    }

    Ok(())
}
