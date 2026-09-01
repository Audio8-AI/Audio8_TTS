// Port of onnx_runtime/arktts_runtime/runtime.py's ArkTtsRuntime against
// onnxruntime-web InferenceSession objects. Verified: prompt-tensor construction
// and the first generated codec frame are byte-identical to the Python
// implementation (same fixed seed, replayed RNG draws); later frames diverge only
// by small cross-execution-provider floating point differences (WASM SIMD vs
// native CPU EP), never structurally (call count/order/array sizes all match,
// all generated values stay within valid codebook range).

// top-p + top-k + Gumbel-noise argmax sampling trick, mirrors Python's _sample().
export function sampleTopPTopK(logitsArr, temperature, topP, topK, rng) {
  const n = logitsArr.length;
  const values = Array.from(logitsArr, Number);
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => values[b] - values[a]);
  const sortedValues = order.map((i) => values[i]);
  const maxSorted = Math.max(...sortedValues);
  const base = sortedValues.map((v) => Math.exp(v - maxSorted));
  const baseSum = base.reduce((a, b) => a + b, 0);
  const baseNorm = base.map((v) => v / baseSum);
  const cumulative = [];
  {
    let acc = 0;
    for (const v of baseNorm) { acc += v; cumulative.push(acc); }
  }
  const remove = cumulative.map((c, i) => c > topP || i >= topK);
  remove[0] = false;

  const masked = values.slice();
  for (let i = 0; i < n; i++) {
    if (remove[i]) masked[order[i]] = -Infinity;
  }
  const temp = Math.max(temperature, 1e-5);
  const scaled = masked.map((v) => v / temp);
  const maxScaled = Math.max(...scaled);
  const probsRaw = scaled.map((v) => Math.exp(v - maxScaled));
  const probsSum = probsRaw.reduce((a, b) => a + b, 0);
  const probs = probsRaw.map((v) => v / probsSum);

  const noise = new Array(n);
  for (let i = 0; i < n; i++) {
    const u = Math.min(Math.max(rng(), 1e-12), 1.0);
    noise[i] = -Math.log(u);
  }
  const scores = probs.map((p, i) => p / noise[i]);
  let best = 0;
  for (let i = 1; i < n; i++) if (scores[i] > scores[best]) best = i;
  return best;
}

export class ArkTtsRuntime {
  constructor({ ort, slowSession, fastSession, manifest }) {
    this.ort = ort;
    this.slow = slowSession;
    this.fast = fastSession;
    this.manifest = manifest;
  }

  emptySlowCaches() {
    const { ort, manifest } = this;
    const shape = [1, manifest.n_local_heads, manifest.max_seq_len, manifest.head_dim];
    const size = shape.reduce((a, b) => a * b, 1);
    const caches = [];
    for (let i = 0; i < 2 * manifest.num_layers; i++) {
      caches.push(new ort.Tensor('float16', new Uint16Array(size), shape));
    }
    return caches;
  }

  emptyFastCaches() {
    const { ort, manifest } = this;
    const shape = [1, manifest.fast_n_local_heads, manifest.num_codebooks, manifest.fast_head_dim];
    const size = shape.reduce((a, b) => a * b, 1);
    const caches = [];
    for (let i = 0; i < 2 * manifest.num_fast_layers; i++) {
      caches.push(new ort.Tensor('float16', new Uint16Array(size), shape));
    }
    return caches;
  }

  // In-place cache write at explicit positions: caches[i][:, :, positions, :] = deltas[i]
  static updateCaches(caches, positions, deltas) {
    for (let idx = 0; idx < deltas.length; idx++) {
      const cache = caches[idx];
      const delta = deltas[idx];
      const [, H, S, D] = cache.dims;
      const deltaLen = delta.dims[2];
      const cacheData = cache.data;
      const deltaData = delta.data;
      for (let h = 0; h < H; h++) {
        for (let t = 0; t < deltaLen; t++) {
          const pos = Number(positions[t]);
          const cacheBase = (h * S + pos) * D;
          const deltaBase = (h * deltaLen + t) * D;
          for (let d = 0; d < D; d++) {
            cacheData[cacheBase + d] = deltaData[deltaBase + d];
          }
        }
      }
    }
  }

  sliceLastPosition(tensor) {
    const [, T, H] = tensor.dims;
    const out = new tensor.data.constructor(H);
    const base = (T - 1) * H;
    for (let i = 0; i < H; i++) out[i] = tensor.data[base + i];
    return new this.ort.Tensor(tensor.type, out, [1, 1, H]);
  }

  async slowStep(codesTensor, positionsArr, caches) {
    const { ort, manifest } = this;
    const feeds = {
      codes: codesTensor,
      input_pos: new ort.Tensor('int64', BigInt64Array.from(positionsArr.map(BigInt)), [positionsArr.length]),
    };
    for (let i = 0; i < manifest.num_layers; i++) {
      feeds[`cache_key_${i}`] = caches[2 * i];
      feeds[`cache_value_${i}`] = caches[2 * i + 1];
    }
    const outputs = await this.slow.run(feeds);
    const outNames = this.slow.outputNames;
    const logits = outputs[outNames[0]];
    const slowHidden = outputs[outNames[1]];
    const deltas = outNames.slice(2).map((n) => outputs[n]);
    ArkTtsRuntime.updateCaches(caches, positionsArr, deltas);
    return { logits: this.sliceLastPosition(logits), hiddenLast: this.sliceLastPosition(slowHidden) };
  }

  async fastStep(hiddenTensor, tokenId, useHidden, position, caches) {
    const { ort, manifest } = this;
    const feeds = {
      slow_hidden: hiddenTensor,
      token_id: new ort.Tensor('int64', BigInt64Array.from([BigInt(tokenId)]), [1, 1]),
      use_slow_hidden: new ort.Tensor('bool', Uint8Array.from([useHidden ? 1 : 0]), [1]),
      input_pos: new ort.Tensor('int64', BigInt64Array.from([BigInt(position)]), [1]),
    };
    for (let i = 0; i < manifest.num_fast_layers; i++) {
      feeds[`cache_key_${i}`] = caches[2 * i];
      feeds[`cache_value_${i}`] = caches[2 * i + 1];
    }
    const outputs = await this.fast.run(feeds);
    const outNames = this.fast.outputNames;
    const logits = outputs[outNames[0]];
    const deltas = outNames.slice(1).map((n) => outputs[n]);
    ArkTtsRuntime.updateCaches(caches, [position], deltas);
    return logits;
  }

  sampleSemantic(logitsTensor, previous, temperature, topP, topK, rng) {
    const { manifest } = this;
    const begin = manifest.semantic_begin_id;
    const end = manifest.semantic_end_id;
    const stop = manifest.im_end_id;
    const values = Array.from(logitsTensor.data, Number);

    let allowedIds, allowedLogits;
    if (manifest.slow_logits_layout === 'semantic_then_eos') {
      allowedIds = [];
      for (let i = begin; i <= end; i++) allowedIds.push(i);
      allowedIds.push(stop);
      allowedLogits = values;
    } else {
      allowedIds = [];
      for (let i = begin; i <= end; i++) allowedIds.push(i);
      allowedIds.push(stop);
      allowedLogits = allowedIds.map((id) => values[id]);
    }
    const normalIndex = sampleTopPTopK(allowedLogits, temperature, topP, topK, rng);
    const normal = allowedIds[normalIndex];
    const highIndex = sampleTopPTopK(allowedLogits, 1.0, 0.9, topK, rng);
    const high = allowedIds[highIndex];
    if (normal >= begin && normal <= end && previous.includes(normal)) return high;
    return normal;
  }

  /**
   * Async generator yielding { semantic, frame } per step.
   * @param {{prompt: ort.Tensor, promptLen: number, maxNewTokens: number, temperature: number, topP: number, topK: number, rng: () => number, stopSignal?: {aborted: boolean}}} opts
   */
  async *iterCodes(opts) {
    const { prompt, promptLen, maxNewTokens, temperature, topP, topK, rng, stopSignal } = opts;
    const { manifest } = this;
    const slowCaches = this.emptySlowCaches();
    const positions = Array.from({ length: promptLen }, (_, i) => i);
    let { logits, hiddenLast } = await this.slowStep(prompt, positions, slowCaches);

    let previous = [];
    const begin = manifest.semantic_begin_id;
    const stop = manifest.im_end_id;
    const codebookSize = manifest.codebook_size;
    const numCodebooks = manifest.num_codebooks;

    for (let step = 0; step < maxNewTokens; step++) {
      if (stopSignal && stopSignal.aborted) return;
      const semantic = this.sampleSemantic(logits, previous, temperature, topP, topK, rng);
      if (semantic === stop) return;
      previous.push(semantic);
      if (previous.length > 10) previous = previous.slice(-10);

      const fastCaches = this.emptyFastCaches();
      await this.fastStep(hiddenLast, 0, true, 0, fastCaches);
      let token = Math.min(Math.max(semantic - begin, 0), codebookSize - 1);
      const codebooks = [token];
      for (let fastPos = 1; fastPos < numCodebooks; fastPos++) {
        const fastLogits = await this.fastStep(hiddenLast, token, false, fastPos, fastCaches);
        token = sampleTopPTopK(fastLogits.data, temperature, topP, topK, rng);
        codebooks.push(token);
      }
      yield { semantic, frame: codebooks };
      if (step + 1 >= maxNewTokens) return;

      const column = new this.ort.Tensor(
        'int64',
        BigInt64Array.from([semantic, ...codebooks].map(BigInt)),
        [1, numCodebooks + 1, 1]
      );
      const position = [promptLen + step];
      ({ logits, hiddenLast } = await this.slowStep(column, position, slowCaches));
    }
  }
}

// Deterministic-seedable PRNG (mulberry32) for reproducible browser-side sampling.
// Not bit-parity with numpy's PCG64 (not required - see PRD postcondition), only
// used to make a given seed reproducible run-to-run within the browser itself.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
