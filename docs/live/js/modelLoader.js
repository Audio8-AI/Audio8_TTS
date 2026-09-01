// Progressive model fetch + Cache API caching for the Audio8-TTS-Preview-0.6B-ONNX-INT4
// model set (~572MiB base). Model files are fetched directly from the Hugging Face Hub
// CDN at runtime (confirmed access-control-allow-origin: * and accept-ranges: bytes on
// both small config files and the large .onnx.data weight files) rather than committed
// to this repository - this keeps the git repo and GitHub Pages deploy free of ~600MiB
// of binary weights while still serving a fully static page.

const HF_REPO = 'Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4';
const DEFAULT_BASE = `https://huggingface.co/${HF_REPO}/resolve/main`;
const CACHE_NAME = 'audio8-tts-model-v1';

// Optional self-hosted mirror override, e.g. for a corporate proxy or a faster
// regional mirror: /live/index.html?modelBase=https://mirror.example.com/model
function resolveModelBase() {
  try {
    const override = new URLSearchParams(window.location.search).get('modelBase');
    if (override) return override.replace(/\/+$/, '');
  } catch {
    // window/location unavailable (non-browser context) - fall through to default.
  }
  return DEFAULT_BASE;
}

const HF_BASE = resolveModelBase();

export const MODEL_FILES = [
  { key: 'slow_ar_int4.onnx', path: 'slow_ar_int4.onnx' },
  { key: 'slow_ar_int4.onnx.data', path: 'slow_ar_int4.onnx.data' },
  { key: 'fast_ar_int4.onnx', path: 'fast_ar_int4.onnx' },
  { key: 'fast_ar_int4.onnx.data', path: 'fast_ar_int4.onnx.data' },
  { key: 'codec_decoder_fp16.onnx', path: 'codec_decoder_fp16.onnx' },
  { key: 'codec_decoder_fp16.onnx.data', path: 'codec_decoder_fp16.onnx.data' },
  { key: 'tokenizer.json', path: 'tokenizer/tokenizer.json' },
  { key: 'runtime_manifest.json', path: 'runtime_manifest.json' },
];

export const REGISTRATION_FILES = [
  { key: 'codec_encoder_fp16.onnx', path: 'registration/codec_encoder_fp16.onnx' },
  { key: 'codec_encoder_fp16.onnx.data', path: 'registration/codec_encoder_fp16.onnx.data' },
  { key: 'registration_manifest.json', path: 'registration/registration_manifest.json' },
];

// The Cache Storage API's backing store is a disk-mediated browser service; on
// some automated/ephemeral browser profiles (observed: CDP-launched Chrome with
// a non-persistent storage partition) its promises never settle rather than
// rejecting. caches.open()/match() resolve fine there, but cache.put() and
// navigator.storage.estimate() hang indefinitely - confirmed by isolated testing
// (localStorage, a synchronous API, works in the same environment; every async
// Storage-Manager-backed call does not). A hard timeout keeps a broken storage
// backend from blocking model loading - caching becomes best-effort, never a
// load-correctness requirement.
const CACHE_OP_TIMEOUT_MS = 4000;

function withTimeout(promise, ms, label) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
    promise.then(
      (v) => { clearTimeout(timer); resolve(v); },
      (e) => { clearTimeout(timer); reject(e); }
    );
  });
}

/**
 * Fetch one file with progress reporting, using the Cache API to skip re-download
 * on repeat visits where the browser's storage backend supports it. Returns an
 * ArrayBuffer. Caching is strictly best-effort: any Cache API failure or hang
 * degrades to an uncached fetch rather than blocking the load.
 * @param {{key:string,path:string}} file
 * @param {(loaded:number, total:number)=>void} onProgress
 */
async function fetchWithCache(file, onProgress) {
  const url = `${HF_BASE}/${file.path}`;

  let cache = null;
  try {
    cache = await withTimeout(caches.open(CACHE_NAME), CACHE_OP_TIMEOUT_MS, 'caches.open');
    const cached = await withTimeout(cache.match(url), CACHE_OP_TIMEOUT_MS, 'cache.match');
    if (cached) {
      const buf = await cached.arrayBuffer();
      onProgress(buf.byteLength, buf.byteLength);
      return buf;
    }
  } catch (err) {
    console.warn('model cache read unavailable (continuing without cache):', err.message);
    cache = null;
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to fetch ${file.path}: HTTP ${response.status}`);
  }
  const total = Number(response.headers.get('content-length')) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(loaded, total);
  }
  const buf = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buf.set(chunk, offset);
    offset += chunk.byteLength;
  }

  if (cache) {
    try {
      await withTimeout(
        cache.put(url, new Response(buf, { headers: { 'Content-Length': String(loaded) } })),
        CACHE_OP_TIMEOUT_MS,
        'cache.put'
      );
    } catch (err) {
      console.warn('model cache write failed or timed out (continuing without cache):', err.message);
    }
  }

  return buf.buffer;
}

/**
 * Load the full model file set with aggregate progress reporting.
 * @param {Array<{key:string,path:string}>} fileList
 * @param {(info: {key:string, loaded:number, total:number, fileIndex:number, fileCount:number, aggregateLoaded:number, aggregateTotal:number}) => void} onProgress
 * @returns {Promise<Record<string, ArrayBuffer>>}
 */
export async function loadModelFiles(fileList, onProgress) {
  const results = {};
  const fileTotals = new Array(fileList.length).fill(0);
  const fileLoaded = new Array(fileList.length).fill(0);

  // Pre-fetch content-length via HEAD for an accurate aggregate total where possible.
  await Promise.all(
    fileList.map(async (file, i) => {
      try {
        const res = await fetch(`${HF_BASE}/${file.path}`, { method: 'HEAD' });
        fileTotals[i] = Number(res.headers.get('content-length')) || 0;
      } catch {
        fileTotals[i] = 0;
      }
    })
  );

  for (let i = 0; i < fileList.length; i++) {
    const file = fileList[i];
    const buf = await fetchWithCache(file, (loaded, total) => {
      fileLoaded[i] = loaded;
      if (total) fileTotals[i] = total;
      const aggregateLoaded = fileLoaded.reduce((a, b) => a + b, 0);
      const aggregateTotal = fileTotals.reduce((a, b) => a + b, 0);
      onProgress({
        key: file.key,
        loaded,
        total: fileTotals[i],
        fileIndex: i,
        fileCount: fileList.length,
        aggregateLoaded,
        aggregateTotal,
      });
    });
    results[file.key] = buf;
  }
  return results;
}

/** Clear the cached model set (for a "re-download" / troubleshooting action). */
export async function clearModelCache() {
  try {
    await withTimeout(caches.delete(CACHE_NAME), CACHE_OP_TIMEOUT_MS, 'caches.delete');
  } catch (err) {
    console.warn('model cache clear unavailable:', err.message);
  }
}

/** Check whether the full base model set is already cached (for skip-download UI). */
export async function isModelCached() {
  let cache;
  try {
    cache = await withTimeout(caches.open(CACHE_NAME), CACHE_OP_TIMEOUT_MS, 'caches.open');
  } catch (err) {
    console.warn('model cache unavailable, assuming uncached:', err.message);
    return false;
  }
  for (const file of MODEL_FILES) {
    const url = `${HF_BASE}/${file.path}`;
    let match;
    try {
      match = await withTimeout(cache.match(url), CACHE_OP_TIMEOUT_MS, 'cache.match');
    } catch {
      return false;
    }
    if (!match) return false;
  }
  return true;
}
