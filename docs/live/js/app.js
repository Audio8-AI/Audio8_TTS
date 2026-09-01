import ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.min.mjs';
import { PreTrainedTokenizer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js';
import { PromptBuilder } from './promptBuilder.js';
import { ArkTtsRuntime, mulberry32 } from './runtime.js';
import { framesToCodes, decodeCodes, pcmToAudioBuffer, pcmToWavBlob } from './codecDecoder.js';
import { loadModelFiles, MODEL_FILES, REGISTRATION_FILES, isModelCached } from './modelLoader.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';

// Optional generation-length cap override, e.g. ?maxNewTokens=64 for a quicker
// synthesis on lower-powered devices. Defaults to 256 (~12s of audio headroom
// at this model's codec_hop_length/sample_rate).
const MAX_NEW_TOKENS = Number(new URLSearchParams(window.location.search).get('maxNewTokens')) || 256;

const els = {
  status: document.getElementById('status'),
  progressBar: document.getElementById('progress-bar'),
  progressLabel: document.getElementById('progress-label'),
  loadButton: document.getElementById('load-model'),
  synthesizeButton: document.getElementById('synthesize'),
  targetText: document.getElementById('target-text'),
  referenceText: document.getElementById('reference-text'),
  referenceAudio: document.getElementById('reference-audio'),
  registerButton: document.getElementById('register-voice'),
  player: document.getElementById('player'),
  downloadLink: document.getElementById('download-link'),
  epSelect: document.getElementById('ep-select'),
  errorBox: document.getElementById('error-box'),
  timingBox: document.getElementById('timing-box'),
};

let state = {
  manifest: null,
  tokenizer: null,
  promptBuilder: null,
  slowSession: null,
  fastSession: null,
  decoderSession: null,
  encoderSession: null,
  registrationManifest: null,
  referenceCodes: null, // number[numCodebooks][T]
  referenceText: '',
  audioCtx: null,
  loaded: false,
};

function setStatus(text) {
  els.status.textContent = text;
}

function setError(text) {
  if (!text) {
    els.errorBox.hidden = true;
    els.errorBox.textContent = '';
    return;
  }
  els.errorBox.hidden = false;
  els.errorBox.textContent = text;
}

function setProgress(fraction, label) {
  els.progressBar.style.width = `${Math.round(fraction * 100)}%`;
  els.progressLabel.textContent = label;
}

function formatBytes(n) {
  if (!n) return '0 B';
  const units = ['B', 'KiB', 'MiB', 'GiB'];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(1)} ${units[i]}`;
}

async function loadModel(executionProvider) {
  setError(null);
  els.loadButton.disabled = true;
  const cached = await isModelCached();
  setStatus(cached ? 'Loading cached model...' : 'Downloading model (~572 MiB, first visit only)...');
  setProgress(0, '');

  const files = await loadModelFiles(MODEL_FILES, (info) => {
    const fraction = info.aggregateTotal ? info.aggregateLoaded / info.aggregateTotal : 0;
    setProgress(
      fraction,
      `${info.key} (${info.fileIndex + 1}/${info.fileCount}) - ${formatBytes(info.aggregateLoaded)} / ${formatBytes(info.aggregateTotal)}`
    );
  });

  setStatus('Parsing manifest and tokenizer...');
  const manifest = JSON.parse(new TextDecoder().decode(files['runtime_manifest.json']));
  const tokenizerJSON = JSON.parse(new TextDecoder().decode(files['tokenizer.json']));
  const tokenizer = new PreTrainedTokenizer(tokenizerJSON, {});
  const promptBuilder = new PromptBuilder(tokenizer, manifest.semantic_begin_id, manifest.num_codebooks);

  // NOTE (web-webgpu-acceleration, 2026-08-23): WebGPU session creation succeeds
  // for these graphs, but session.run() currently fails on both slow_ar_int4.onnx
  // and fast_ar_int4.onnx with "[WebGPU] Kernel [MatMulNBits] .../feed_forward/w2/
  // MatMul_Q4 failed. Error: cannot convert shape" - a reproducible upstream bug in
  // onnxruntime-web's WebGPU MatMulNBits kernel (matches microsoft/onnxruntime
  // issue #28029, int64->uint32 shape casting gaps in the WebGPU EP). Confirmed
  // still present on the 1.30.0-dev nightly (2 days old at investigation time) and
  // not recoverable via the executionProviders:['webgpu','wasm'] fallback array
  // (that only applies at session-creation time, not to a runtime kernel failure).
  // WASM is the recommended and default execution provider until this is fixed
  // upstream; the WebGPU option remains selectable for forward compatibility.
  setStatus(`Creating inference sessions (${executionProvider})...`);
  const sessionOpts = { executionProviders: [executionProvider] };

  function withExternalData(name, buf) {
    const dataBuf = files[`${name}.data`];
    const opts = { ...sessionOpts };
    if (dataBuf) opts.externalData = [{ path: `${name}.data`, data: new Uint8Array(dataBuf) }];
    return opts;
  }

  const slowSession = await ort.InferenceSession.create(
    new Uint8Array(files['slow_ar_int4.onnx']),
    withExternalData('slow_ar_int4.onnx', files['slow_ar_int4.onnx'])
  );
  const fastSession = await ort.InferenceSession.create(
    new Uint8Array(files['fast_ar_int4.onnx']),
    withExternalData('fast_ar_int4.onnx', files['fast_ar_int4.onnx'])
  );
  const decoderSession = await ort.InferenceSession.create(
    new Uint8Array(files['codec_decoder_fp16.onnx']),
    withExternalData('codec_decoder_fp16.onnx', files['codec_decoder_fp16.onnx'])
  );

  state = {
    ...state,
    manifest,
    tokenizer,
    promptBuilder,
    slowSession,
    fastSession,
    decoderSession,
    loaded: true,
  };

  setStatus('Model ready.');
  setProgress(1, 'Loaded.');
  els.synthesizeButton.disabled = false;
  els.registerButton.disabled = false;
  els.loadButton.disabled = false;
  els.loadButton.textContent = 'Reload model';
}

async function ensureEncoderLoaded() {
  if (state.encoderSession) return;
  setStatus('Downloading voice-registration encoder (~415 MiB)...');
  const files = await loadModelFiles(REGISTRATION_FILES, (info) => {
    const fraction = info.aggregateTotal ? info.aggregateLoaded / info.aggregateTotal : 0;
    setProgress(fraction, `registration: ${info.key}`);
  });
  const registrationManifest = JSON.parse(new TextDecoder().decode(files['registration_manifest.json']));
  const opts = { executionProviders: [els.epSelect.value] };
  const dataBuf = files['codec_encoder_fp16.onnx.data'];
  if (dataBuf) opts.externalData = [{ path: 'codec_encoder_fp16.onnx.data', data: new Uint8Array(dataBuf) }];
  const encoderSession = await ort.InferenceSession.create(new Uint8Array(files['codec_encoder_fp16.onnx']), opts);
  state.encoderSession = encoderSession;
  state.registrationManifest = registrationManifest;
  setStatus('Voice-registration encoder ready.');
}

async function decodeAudioFileToMono(file, targetRate) {
  const arrayBuf = await file.arrayBuffer();
  const audioCtx = state.audioCtx || (state.audioCtx = new (window.AudioContext || window.webkitAudioContext)());
  const decoded = await audioCtx.decodeAudioData(arrayBuf.slice(0));
  const duration = decoded.duration;
  if (duration < 0.5 || duration > 30) {
    throw new Error('reference audio duration must be between 0.5 and 30 seconds');
  }
  let mono;
  if (decoded.numberOfChannels === 1) {
    mono = decoded.getChannelData(0).slice();
  } else {
    mono = new Float32Array(decoded.length);
    for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
      const data = decoded.getChannelData(ch);
      for (let i = 0; i < data.length; i++) mono[i] += data[i] / decoded.numberOfChannels;
    }
  }
  if (decoded.sampleRate !== targetRate) {
    mono = await resampleLinear(mono, decoded.sampleRate, targetRate);
  }
  const padded = padTo(mono, 2048);
  return padded;
}

function padTo(arr, multiple) {
  const rem = arr.length % multiple;
  if (rem === 0) return arr;
  const out = new Float32Array(arr.length + (multiple - rem));
  out.set(arr);
  return out;
}

async function resampleLinear(samples, fromRate, toRate) {
  const ratio = toRate / fromRate;
  const outLength = Math.round(samples.length * ratio);
  const out = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const srcPos = i / ratio;
    const i0 = Math.floor(srcPos);
    const i1 = Math.min(i0 + 1, samples.length - 1);
    const frac = srcPos - i0;
    out[i] = samples[i0] * (1 - frac) + samples[i1] * frac;
  }
  return out;
}

async function registerVoice() {
  setError(null);
  const file = els.referenceAudio.files[0];
  const text = els.referenceText.value.trim();
  if (!file) { setError('Choose a reference audio file first.'); return; }
  if (!text) { setError('Enter the exact transcript of the reference audio.'); return; }
  els.registerButton.disabled = true;
  try {
    await ensureEncoderLoaded();
    const targetRate = state.registrationManifest.sample_rate;
    setStatus('Decoding reference audio...');
    const pcm = await decodeAudioFileToMono(file, targetRate);
    const inputTensor = new ort.Tensor('float32', pcm, [1, 1, pcm.length]);
    setStatus('Encoding reference voice...');
    const inputMeta = state.encoderSession.inputMetadata[0];
    let feedTensor = inputTensor;
    if (inputMeta.type === 'float16') {
      const half = new Uint16Array(pcm.length);
      for (let i = 0; i < pcm.length; i++) half[i] = float32ToFloat16Bits(pcm[i]);
      feedTensor = new ort.Tensor('float16', half, [1, 1, pcm.length]);
    }
    const outputs = await state.encoderSession.run({ [state.encoderSession.inputNames[0]]: feedTensor });
    const codesTensor = outputs[state.encoderSession.outputNames[0]];
    const numCodebooks = state.manifest.num_codebooks;
    const dims = codesTensor.dims;
    const T = dims[dims.length - 1];
    const flat = codesTensor.data;
    const codes = [];
    for (let c = 0; c < numCodebooks; c++) {
      const row = new Array(T);
      for (let t = 0; t < T; t++) row[t] = Number(flat[c * T + t]);
      codes.push(row);
    }
    state.referenceCodes = codes;
    state.referenceText = text;
    setStatus('Voice registered. Ready to synthesize with your reference voice.');
  } catch (err) {
    console.error(err);
    setError(`Voice registration failed: ${err.message}`);
  } finally {
    els.registerButton.disabled = false;
  }
}

function float32ToFloat16Bits(value) {
  const floatView = new Float32Array(1);
  const int32View = new Int32Array(floatView.buffer);
  floatView[0] = value;
  const x = int32View[0];
  let bits = (x >> 16) & 0x8000;
  let m = (x >> 12) & 0x07ff;
  const e = (x >> 23) & 0xff;
  if (e >= 103) {
    bits |= ((e - 112) << 10) | (m >> 1);
    bits += m & 1;
  }
  return bits;
}

function defaultReferenceCodes(manifest) {
  // Deterministic placeholder reference so synthesis works before any voice is
  // registered (matches the shape contract PromptBuilder requires: [numCodebooks, T>0]).
  const numCodebooks = manifest.num_codebooks;
  const T = 8;
  const codes = [];
  for (let c = 0; c < numCodebooks; c++) {
    const row = [];
    for (let t = 0; t < T; t++) row.push((c * 37 + t * 13) % manifest.codebook_size);
    codes.push(row);
  }
  return codes;
}

async function synthesize() {
  setError(null);
  const text = els.targetText.value.trim();
  if (!text) { setError('Enter text to synthesize.'); return; }
  if (!state.loaded) { setError('Load the model first.'); return; }

  els.synthesizeButton.disabled = true;
  setStatus('Synthesizing...');
  const t0 = performance.now();
  try {
    const referenceCodes = state.referenceCodes || defaultReferenceCodes(state.manifest);
    const referenceText = state.referenceText || 'Reference voice sample.';
    const built = state.promptBuilder.build(text, referenceText, referenceCodes);
    const [, rows, promptLen] = built.shape;
    const flatCodes = new BigInt64Array(rows * promptLen);
    for (let r = 0; r < rows; r++) {
      for (let t = 0; t < promptLen; t++) flatCodes[r * promptLen + t] = BigInt(built.values[r][t]);
    }
    const promptTensor = new ort.Tensor('int64', flatCodes, [1, rows, promptLen]);

    const runtime = new ArkTtsRuntime({ ort, slowSession: state.slowSession, fastSession: state.fastSession, manifest: state.manifest });
    const rng = mulberry32(Date.now() & 0xffffffff);

    const frames = [];
    const genT0 = performance.now();
    for await (const { frame } of runtime.iterCodes({
      prompt: promptTensor,
      promptLen,
      maxNewTokens: MAX_NEW_TOKENS,
      temperature: 0.7,
      topP: 0.9,
      topK: 50,
      rng,
    })) {
      frames.push(frame);
      setStatus(`Generating... ${frames.length} frames`);
    }
    const genMs = performance.now() - genT0;

    if (frames.length === 0) throw new Error('model produced no codec frames');

    setStatus('Decoding audio...');
    const decodeT0 = performance.now();
    const codes = framesToCodes(frames, state.manifest.num_codebooks);
    const pcm = await decodeCodes(ort, state.decoderSession, codes);
    const decodeMs = performance.now() - decodeT0;

    const audioCtx = state.audioCtx || (state.audioCtx = new (window.AudioContext || window.webkitAudioContext)());
    const buffer = pcmToAudioBuffer(audioCtx, pcm, state.manifest.sample_rate);
    const wavBlob = pcmToWavBlob(pcm, state.manifest.sample_rate);
    const url = URL.createObjectURL(wavBlob);
    els.player.src = url;
    els.player.hidden = false;
    els.downloadLink.href = url;
    els.downloadLink.hidden = false;

    const totalMs = performance.now() - t0;
    const audioSeconds = pcm.length / state.manifest.sample_rate;
    const rtf = (totalMs / 1000) / audioSeconds;
    els.timingBox.hidden = false;
    els.timingBox.textContent =
      `frames=${frames.length} generation=${genMs.toFixed(0)}ms decode=${decodeMs.toFixed(0)}ms ` +
      `total=${totalMs.toFixed(0)}ms audio=${audioSeconds.toFixed(2)}s RTF=${rtf.toFixed(2)}`;

    setStatus('Done.');
  } catch (err) {
    console.error(err);
    setError(`Synthesis failed: ${err.message}`);
    setStatus('Idle.');
  } finally {
    els.synthesizeButton.disabled = false;
  }
}

els.loadButton.addEventListener('click', () => {
  loadModel(els.epSelect.value).catch((err) => {
    console.error(err);
    setError(`Model load failed: ${err.message}`);
    setStatus('Idle.');
    els.loadButton.disabled = false;
  });
});
els.synthesizeButton.addEventListener('click', () => synthesize());
els.registerButton.addEventListener('click', () => registerVoice());

setStatus('Idle. Click "Load model" to begin.');
