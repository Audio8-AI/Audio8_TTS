// Wires codec_decoder_fp16.onnx into the browser session set and decodes generated
// codec codes to a playable waveform. Port of ArkTtsRuntime.decode_codes plus
// WAV/PCM assembly for Web Audio API playback.

/**
 * @param {Array<number[]>} frames - array of per-step codebook arrays, each length numCodebooks
 * @param {number} numCodebooks
 * @returns {number[][]} codes[codebook][frame] - transposed to match Python's np.stack(frames, axis=1)
 */
export function framesToCodes(frames, numCodebooks) {
  const T = frames.length;
  const codes = [];
  for (let c = 0; c < numCodebooks; c++) {
    const row = new Array(T);
    for (let t = 0; t < T; t++) row[t] = frames[t][c];
    codes.push(row);
  }
  return codes;
}

/**
 * Decode codec codes to a Float32Array PCM waveform.
 * @param {import('onnxruntime-web')} ort
 * @param {import('onnxruntime-web').InferenceSession} decoderSession
 * @param {number[][]} codes - [numCodebooks][T]
 * @returns {Promise<Float32Array>}
 */
export async function decodeCodes(ort, decoderSession, codes) {
  const numCodebooks = codes.length;
  const T = codes[0].length;
  const flat = new BigInt64Array(numCodebooks * T);
  for (let c = 0; c < numCodebooks; c++) {
    for (let t = 0; t < T; t++) {
      flat[c * T + t] = BigInt(codes[c][t]);
    }
  }
  const codesTensor = new ort.Tensor('int64', flat, [1, numCodebooks, T]);
  const outputs = await decoderSession.run({ codes: codesTensor });
  const audioTensor = outputs[decoderSession.outputNames[0]];
  // audio output dims [1, 1, samples] float32
  return Float32Array.from(audioTensor.data);
}

/**
 * Build a playable AudioBuffer from decoded PCM samples.
 * @param {AudioContext} audioCtx
 * @param {Float32Array} pcm
 * @param {number} sampleRate
 * @returns {AudioBuffer}
 */
export function pcmToAudioBuffer(audioCtx, pcm, sampleRate) {
  const buffer = audioCtx.createBuffer(1, pcm.length, sampleRate);
  buffer.copyToChannel(pcm, 0);
  return buffer;
}

/**
 * Encode Float32 PCM samples as a 16-bit PCM WAV Blob, for download or <audio src>.
 * @param {Float32Array} pcm
 * @param {number} sampleRate
 * @returns {Blob}
 */
export function pcmToWavBlob(pcm, sampleRate) {
  const numSamples = pcm.length;
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample; // mono
  const byteRate = sampleRate * blockAlign;
  const dataSize = numSamples * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM fmt chunk size
  view.setUint16(20, 1, true); // audio format = PCM
  view.setUint16(22, 1, true); // channels = mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}
