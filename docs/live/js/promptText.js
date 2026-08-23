// Port of onnx_runtime/arktts_runtime/prompt.py's clean_text/format_reference_text.
// Verified bit-exact against the Python implementation for English, Chinese, Japanese,
// French, Korean, mixed-script, and whitespace/CJK-newline edge cases.

const CJK_RANGES = [
  [0x1100, 0x11ff], [0x2e80, 0x2fdf], [0x3000, 0x303f], [0x3040, 0x30ff],
  [0x3100, 0x31ff], [0x3400, 0x4dbf], [0x4e00, 0x9fff], [0xa960, 0xa97f],
  [0xac00, 0xd7a3], [0xd7b0, 0xd7ff], [0xf900, 0xfaff],
  [0xfe30, 0xfe4f], [0xff01, 0xff9f], [0x20000, 0x2fa1f],
];

function isCjkCodepoint(cp) {
  for (const [lo, hi] of CJK_RANGES) {
    if (cp >= lo && cp <= hi) return true;
  }
  return false;
}

// Matches Python's _LINE_BREAK_RE: [\r\n\v\f\x1c-\x1e\x85]
const LINE_BREAK_CODEPOINTS = new Set([
  0x0d, 0x0a, 0x0b, 0x0c, 0x1c, 0x1d, 0x1e, 0x85, 0x2028, 0x2029,
]);

function hasLineBreak(s) {
  for (const ch of s) {
    if (LINE_BREAK_CODEPOINTS.has(ch.codePointAt(0))) return true;
  }
  return false;
}

// Python's unicodedata.category(char).startswith("C"): Cc, Cf, Cs, Co, Cn.
const CONTROL_CATEGORY_RE = /\p{Cc}|\p{Cf}|\p{Co}|\p{Cs}|\p{Cn}/u;

function isControlChar(ch) {
  return CONTROL_CATEGORY_RE.test(ch);
}

function isWhitespaceChar(ch) {
  return /\s/u.test(ch);
}

function normalizeWhitespace(text) {
  // Re-implements Python's re.sub(r"\s+", replace, text).strip():
  // a whitespace run touching a line break AND flanked on both sides by a
  // single CJK codepoint collapses to nothing; every other run becomes one space.
  const codepoints = Array.from(text);
  let result = '';
  let i = 0;
  while (i < codepoints.length) {
    const ch = codepoints[i];
    if (isWhitespaceChar(ch)) {
      let j = i;
      while (j < codepoints.length && isWhitespaceChar(codepoints[j])) j++;
      const run = codepoints.slice(i, j).join('');
      const left = i > 0 ? codepoints[i - 1] : '';
      const right = j < codepoints.length ? codepoints[j] : '';
      const leftIsCjk = left && isCjkCodepoint(left.codePointAt(0));
      const rightIsCjk = right && isCjkCodepoint(right.codePointAt(0));
      if (!(hasLineBreak(run) && leftIsCjk && rightIsCjk)) {
        result += ' ';
      }
      i = j;
    } else {
      result += ch;
      i++;
    }
  }
  return result.trim();
}

export function cleanText(text) {
  const codepoints = Array.from(String(text));
  let value = '';
  for (const ch of codepoints) {
    if (isWhitespaceChar(ch)) {
      value += ch;
    } else if (!isControlChar(ch)) {
      value += ch;
    }
  }
  return normalizeWhitespace(value);
}

const SPEAKER_TAG_RE = /<\|speaker:\d+\|>/;

export function formatReferenceText(text) {
  const cleaned = cleanText(text);
  return SPEAKER_TAG_RE.test(cleaned) ? cleaned : `<|speaker:0|>${cleaned}`;
}
