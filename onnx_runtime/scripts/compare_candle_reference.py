"""
Numeric comparison for port-slow-fast-ar-to-candle: runs the exact same
prompt (probe_voice reference codes + target_text) through the working
ONNX Runtime reference (arktts_runtime.runtime.ArkTtsRuntime), matching
candle_runtime/src/bin/model_check.rs's slow_step + fast_step calls, and
diffs the resulting logits/argmax tokens against
candle_runtime/weights/candle_model_check_out.json.

Not exact bit-parity (int4 GGUF Q4_0 vs ONNX's own int4 MatMulNBits differ
in block size/scheme - see candle-weight-repacking-quantization's witness),
but must show correlated top-k tokens and a similar logit distribution
shape, not random noise.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arktts_runtime.runtime import ArkTtsRuntime  # noqa: E402

MODEL_DIR = ROOT / "model"
VOICES_DIR = ROOT / "voices"


def main():
    runtime = ArkTtsRuntime(MODEL_DIR, VOICES_DIR, precision="int4")

    reference_codes, meta = runtime.voices.load("probe_voice")
    reference_text = meta["reference_text"]
    target_text = "The quick brown fox jumps over the lazy dog."

    prompt = runtime.prompt_builder.build(target_text, reference_text, reference_codes)
    prompt_len = int(prompt.shape[2])
    print(f"[py] prompt shape: {prompt.shape}")

    slow_caches = runtime._empty_slow_caches()
    positions = np.arange(prompt_len, dtype=np.int64)
    logits, hidden = runtime._slow_step(prompt, positions, slow_caches)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    print(f"[py] slow logits shape: {logits.shape}")

    top10_idx = np.argsort(logits)[::-1][:10]
    print("[py] slow logits top-10 (index, value):")
    for i in top10_idx:
        print(f"    {int(i)}: {logits[i]:.4f}")

    fast_caches = runtime._empty_fast_caches()
    first_logits = runtime._fast_step(hidden, 0, True, 0, fast_caches)
    first_logits = np.asarray(first_logits, dtype=np.float64).reshape(-1)
    first_top5 = np.argsort(first_logits)[::-1][:5]
    print(f"[py] fast_step(position=0) top-5: {[(int(i), float(first_logits[i])) for i in first_top5]}")

    semantic_then_eos_argmax = int(top10_idx[0])
    begin = int(runtime.manifest["semantic_begin_id"])
    codebook_size = int(runtime.manifest["codebook_size"])
    token0 = min(max(semantic_then_eos_argmax - 0, 0), codebook_size - 1)  # logits already semantic_then_eos-indexed, index IS the offset
    codebooks = [token0]
    for fast_pos in range(1, int(runtime.manifest["num_codebooks"])):
        fast_logits = runtime._fast_step(hidden, codebooks[-1], False, fast_pos, fast_caches)
        fast_logits = np.asarray(fast_logits, dtype=np.float64).reshape(-1)
        codebooks.append(int(np.argmax(fast_logits)))
    print(f"[py] generated codebook tokens (greedy): {codebooks}")

    out = {
        "prompt_shape": list(prompt.shape),
        "slow_logits": logits.tolist(),
        "slow_top10": [[int(i), float(logits[i])] for i in top10_idx],
        "fast_first_logits_top5": [[int(i), float(first_logits[i])] for i in first_top5],
        "codebooks_greedy": codebooks,
    }
    out_path = ROOT.parent / "candle_runtime" / "weights" / "python_model_check_out.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[py] wrote {out_path}")

    # --- Diff against candle output ---
    candle_path = ROOT.parent / "candle_runtime" / "weights" / "candle_model_check_out.json"
    if not candle_path.exists():
        print(f"[py] candle output not found at {candle_path}, skipping diff")
        return
    candle = json.loads(candle_path.read_text())
    candle_logits = np.asarray(candle["slow_logits"], dtype=np.float64)

    if candle_logits.shape != logits.shape:
        print(f"[py] SHAPE MISMATCH: candle={candle_logits.shape} python={logits.shape}")
        return

    corr = np.corrcoef(candle_logits, logits)[0, 1]
    print(f"\n[compare] slow logits Pearson correlation (candle vs python ONNX ref): {corr:.4f}")

    candle_top10 = set(i for i, _ in candle["slow_top10"])
    python_top10 = set(int(i) for i in top10_idx)
    overlap = candle_top10 & python_top10
    print(f"[compare] top-10 index overlap: {len(overlap)}/10 ({sorted(overlap)})")

    candle_argmax = candle["slow_top10"][0][0]
    python_argmax = int(top10_idx[0])
    print(f"[compare] argmax match: candle={candle_argmax} python={python_argmax} -> {'MATCH' if candle_argmax == python_argmax else 'DIFFER'}")

    candle_codebooks = candle["codebooks_greedy"]
    print(f"[compare] codebooks: candle={candle_codebooks} python={codebooks}")
    cb_matches = sum(1 for a, b in zip(candle_codebooks, codebooks) if a == b)
    print(f"[compare] codebook token matches: {cb_matches}/{len(codebooks)}")


if __name__ == "__main__":
    main()
