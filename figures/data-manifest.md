# Evaluation Figure Data Manifest

All values are real evaluation results transcribed from the Evaluation tables
in this repository's `README.md`. Similarity (SIM) values are intentionally
excluded. The Seed-TTS export contains only the core EN WER and ZH CER tracks;
the Hard ZH track is intentionally omitted from that CSV and figure.

| Figure | Data file | Real/mock | Source | Script | Outputs |
|---|---|---|---|---|---|
| Seed-TTS error rates | `data/seed_tts_error_rates.csv` | Real | `README.md` Seed-TTS table | `evaluation/fig1_seed_tts_error_rates.py` | `evaluation/fig1_seed_tts_error_rates.png`, `evaluation/fig1_seed_tts_error_rates.svg` |
| CV3 multilingual error rates | `data/cv3_error_rates.csv` | Real | `README.md` CV3 table | `evaluation/fig2_cv3_error_rates.py` | `evaluation/fig2_cv3_error_rates.png`, `evaluation/fig2_cv3_error_rates.svg` |

Metric mapping for CV3 follows the benchmark language convention: CER is used
for Chinese, Japanese, and Korean; WER is used for English and the remaining
alphabetic languages. Empty cells represent results that were not reported.
