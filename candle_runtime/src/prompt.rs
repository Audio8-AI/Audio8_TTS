use std::path::Path;

use tokenizers::Tokenizer;

fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x1100..=0x11FF | 0x2E80..=0x2FDF | 0x3000..=0x303F | 0x3040..=0x30FF |
        0x3100..=0x31FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xA960..=0xA97F |
        0xAC00..=0xD7A3 | 0xD7B0..=0xD7FF | 0xF900..=0xFAFF |
        0xFE30..=0xFE4F | 0xFF01..=0xFF9F | 0x20000..=0x2FA1F
    )
}

fn is_line_break(c: char) -> bool {
    matches!(c, '\r' | '\n' | '\x0b' | '\x0c' | '\u{1c}'..='\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}')
}

/// Mirrors onnx_runtime/arktts_runtime/prompt.py's clean_text: strips Unicode
/// control/format/etc categories, then collapses whitespace runs - a run that
/// contains a line break AND sits between two CJK characters is deleted
/// entirely (CJK text has no word-space convention), otherwise collapsed to " ".
pub fn clean_text(text: &str) -> String {
    let filtered: String = text
        .chars()
        .filter(|c| {
            if c.is_whitespace() {
                return true;
            }
            !is_control_or_format(*c)
        })
        .collect();

    normalize_whitespace(&filtered)
}

fn is_control_or_format(c: char) -> bool {
    // Unicode general category C* (Cc, Cf, Cs, Co, Cn). char::is_control()
    // only covers Cc; approximate the rest via known non-printable ranges
    // used by unicodedata.category(...).startswith("C") in Python.
    c.is_control()
        || matches!(c as u32,
            0x00AD | 0x200B..=0x200F | 0x202A..=0x202E | 0x2060..=0x2064 |
            0xD800..=0xDFFF | 0xE000..=0xF8FF | 0xFDD0..=0xFDEF | 0xFFF9..=0xFFFB
        )
}

fn normalize_whitespace(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_whitespace() {
            let start = i;
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            let run = &chars[start..i];
            let has_break = run.iter().any(|c| is_line_break(*c));
            let left = if start > 0 { Some(chars[start - 1]) } else { None };
            let right = if i < chars.len() { Some(chars[i]) } else { None };
            let both_cjk = left.map(is_cjk).unwrap_or(false) && right.map(is_cjk).unwrap_or(false);
            if has_break && both_cjk {
                // drop entirely
            } else {
                result.push(' ');
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }
    result.trim().to_string()
}

fn format_reference_text(text: &str) -> String {
    let cleaned = clean_text(text);
    if cleaned.contains("<|speaker:") {
        cleaned
    } else {
        format!("<|speaker:0|>{}", cleaned)
    }
}

pub struct PromptBuilder {
    tokenizer: Tokenizer,
    semantic_begin_id: i64,
    num_codebooks: usize,
}

impl PromptBuilder {
    pub fn new(tokenizer_dir: &Path, semantic_begin_id: i64, num_codebooks: usize) -> anyhow::Result<Self> {
        let tokenizer_path = tokenizer_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer at {:?}: {e}", tokenizer_path))?;
        Ok(Self { tokenizer, semantic_begin_id, num_codebooks })
    }

    pub fn encode_text(&self, text: &str) -> anyhow::Result<Vec<i64>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
    }

    /// Builds the [1, num_codebooks+1, T] prompt tensor exactly as
    /// PromptBuilder.build in prompt.py: row 0 = text token ids with the
    /// reference's semantic codes spliced in (offset by semantic_begin_id),
    /// rows 1..=num_codebooks = the reference's raw codec codes placed at the
    /// same column offset, everything else zero.
    pub fn build(
        &self,
        target_text: &str,
        reference_text: &str,
        reference_codes: &ndarray::Array2<i64>,
    ) -> anyhow::Result<ndarray::Array3<i64>> {
        if reference_codes.shape()[0] != self.num_codebooks || reference_codes.shape()[1] == 0 {
            anyhow::bail!(
                "reference codes must have shape [{}, T>0], got {:?}",
                self.num_codebooks,
                reference_codes.shape()
            );
        }

        let mut prefix = Vec::new();
        prefix.extend(self.encode_text("<|im_start|>system\n")?);
        prefix.extend(self.encode_text("convert the provided text to speech reference to the following:\n\nText:\n")?);
        prefix.extend(self.encode_text(&format_reference_text(reference_text))?);
        prefix.extend(self.encode_text("\n\nSpeech:\n")?);

        let mut suffix = Vec::new();
        suffix.extend(self.encode_text("<|im_end|>\n")?);
        suffix.extend(self.encode_text("<|im_start|>user\n")?);
        suffix.extend(self.encode_text(&clean_text(target_text))?);
        suffix.extend(self.encode_text("<|im_end|>\n")?);
        suffix.extend(self.encode_text("<|im_start|>assistant\n<|voice|>")?);

        let ref_len = reference_codes.shape()[1];
        let semantic_ids: Vec<i64> = (0..ref_len)
            .map(|t| reference_codes[[0, t]] + self.semantic_begin_id)
            .collect();

        let mut row0 = Vec::with_capacity(prefix.len() + semantic_ids.len() + suffix.len());
        row0.extend(&prefix);
        row0.extend(&semantic_ids);
        row0.extend(&suffix);

        let total_len = row0.len();
        let mut values = ndarray::Array3::<i64>::zeros((1, self.num_codebooks + 1, total_len));
        for (t, &id) in row0.iter().enumerate() {
            values[[0, 0, t]] = id;
        }
        let begin = prefix.len();
        for cb in 0..self.num_codebooks {
            for t in 0..ref_len {
                values[[0, cb + 1, begin + t]] = reference_codes[[cb, t]];
            }
        }

        Ok(values)
    }
}
