// Port of onnx_runtime/arktts_runtime/prompt.py's PromptBuilder.
// Verified bit-exact against the Python implementation's build() output
// (shape, text-token sequence, semantic-token offset, codebook interleaving).
import { cleanText, formatReferenceText } from './promptText.js';

export class PromptBuilder {
  constructor(tokenizer, semanticBeginId, numCodebooks) {
    this.tokenizer = tokenizer;
    this.semanticBeginId = Number(semanticBeginId);
    this.numCodebooks = Number(numCodebooks);
  }

  encodeText(text) {
    const ids = this.tokenizer.encode(text, { add_special_tokens: false });
    return Array.isArray(ids) ? ids : Array.from(ids);
  }

  /**
   * @param {string} targetText
   * @param {string} referenceText
   * @param {number[][]} referenceCodes - [numCodebooks][T]
   * @returns {{shape: number[], values: number[][]}} values[row][col], batch dim implicit (shape[0]===1)
   */
  build(targetText, referenceText, referenceCodes) {
    const numCodebooks = this.numCodebooks;
    const T = referenceCodes[0] ? referenceCodes[0].length : 0;
    if (
      referenceCodes.length !== numCodebooks ||
      referenceCodes.some((row) => row.length !== T) ||
      T === 0
    ) {
      throw new Error(
        `reference codes must have shape [${numCodebooks}, T>0], got [${referenceCodes.length}, ${T}]`
      );
    }

    const prefixParts = [
      '<|im_start|>system\n',
      'convert the provided text to speech reference to the following:\n\nText:\n',
      formatReferenceText(referenceText),
      '\n\nSpeech:\n',
    ];
    const suffixParts = [
      '<|im_end|>\n',
      '<|im_start|>user\n',
      cleanText(targetText),
      '<|im_end|>\n',
      '<|im_start|>assistant\n<|voice|>',
    ];

    const prefix = prefixParts.flatMap((part) => this.encodeText(part));
    const suffix = suffixParts.flatMap((part) => this.encodeText(part));

    const semanticIds = referenceCodes[0].map((v) => Number(v) + this.semanticBeginId);
    const row0 = [...prefix, ...semanticIds, ...suffix];
    const totalLen = row0.length;

    const values = [];
    for (let r = 0; r <= numCodebooks; r++) values.push(new Array(totalLen).fill(0));
    values[0] = row0;

    const begin = prefix.length;
    for (let r = 1; r <= numCodebooks; r++) {
      const srcRow = referenceCodes[r - 1];
      for (let t = 0; t < T; t++) {
        values[r][begin + t] = Number(srcRow[t]);
      }
    }

    return { shape: [1, numCodebooks + 1, totalLen], values };
  }
}
