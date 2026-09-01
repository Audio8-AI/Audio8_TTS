use rand::Rng;

/// Mirrors onnx_runtime/arktts_runtime/runtime.py's _sample: top-p + top-k
/// filtering, temperature scaling, then Gumbel-max sampling (argmax of
/// probs / -ln(U)) instead of cumulative-distribution sampling - this is
/// numerically the same "sample without building a CDF" trick, matched here
/// call-for-call so a fixed seed's *decision sequence* lines up with Python's
/// even though the underlying RNG streams differ.
pub fn sample(logits: &[f32], temperature: f32, top_p: f32, top_k: usize, rng: &mut impl Rng) -> usize {
    let n = logits.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

    let max_val = logits[order[0]];
    let mut exp_sum = 0.0f64;
    let mut exp_vals = vec![0.0f64; n];
    for (rank, &idx) in order.iter().enumerate() {
        let e = ((logits[idx] as f64) - (max_val as f64)).exp();
        exp_vals[rank] = e;
        exp_sum += e;
    }

    let mut cumulative = 0.0f64;
    let mut remove = vec![false; n];
    for rank in 0..n {
        cumulative += exp_vals[rank] / exp_sum;
        if cumulative > top_p as f64 || rank >= top_k {
            remove[rank] = true;
        }
    }
    remove[0] = false;

    let mut masked = logits.to_vec();
    for (rank, &idx) in order.iter().enumerate() {
        if remove[rank] {
            masked[idx] = f32::NEG_INFINITY;
        }
    }

    let temp = temperature.max(1e-5);
    let mut scaled: Vec<f64> = masked.iter().map(|&v| (v as f64) / (temp as f64)).collect();
    let max_scaled = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    for v in scaled.iter_mut() {
        *v -= max_scaled;
    }
    let mut probs: Vec<f64> = scaled.iter().map(|&v| v.exp()).collect();
    let prob_sum: f64 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= prob_sum;
    }

    let mut best_idx = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (idx, &p) in probs.iter().enumerate() {
        let u: f64 = rng.gen_range(1e-12..1.0);
        let noise = -u.ln();
        let score = p / noise;
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }
    best_idx
}
