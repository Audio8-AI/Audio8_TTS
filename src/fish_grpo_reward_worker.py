#!/usr/bin/env python3
import argparse
import contextlib
import importlib.util
import json
import math
import os
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

try:
    import zhconv
except ImportError:
    zhconv = None


SUPPORTED_ASR_LANGS = {
    "cs", "da", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it",
    "ja", "ko", "lt", "nl", "no", "pl", "pt", "ro", "sk", "sl", "sv", "zh",
}
CHAR_METRIC_LANGS = {"ja", "ko", "zh"}
WHISPER_LANGUAGE_NAMES = {
    "cs": "czech",
    "da": "danish",
    "de": "german",
    "en": "english",
    "es": "spanish",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "hr": "croatian",
    "hu": "hungarian",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "lt": "lithuanian",
    "nl": "dutch",
    "no": "norwegian",
    "pl": "polish",
    "pt": "portuguese",
    "ro": "romanian",
    "sk": "slovak",
    "sl": "slovenian",
    "sv": "swedish",
    "zh": "chinese",
}


def normalize_lang(value: str) -> str:
    lang = str(value or "en").strip().lower().replace("_", "-").split("-", 1)[0]
    aliases = {name: code for code, name in WHISPER_LANGUAGE_NAMES.items()}
    aliases.update({"chinese": "zh", "cn": "zh"})
    lang = aliases.get(lang, lang)
    if lang not in SUPPORTED_ASR_LANGS:
        raise ValueError(f"unsupported ASR reward language: {value}")
    return lang


def apply_extra_pythonpath(paths_text: str):
    for item in str(paths_text or "").split(":"):
        item = item.strip()
        if item and item not in sys.path:
            sys.path.insert(0, item)


def apply_hf_modules_cache(cache_dir: pathlib.Path):
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_MODULES_CACHE", str(cache_dir))


def disable_transformers_librosa():
    # Audio is loaded with soundfile/scipy below. Avoid Transformers importing
    # the optional librosa/numba stack, which is incompatible with NumPy 2.3.
    import transformers.utils.import_utils as transformers_imports

    transformers_imports._librosa_available = False


def import_eval_module(seedtts_root: pathlib.Path):
    path = seedtts_root / "scripts" / "eval_seedtts_metrics.py"
    spec = importlib.util.spec_from_file_location("seedtts_eval_metrics_for_grpo", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RewardEngine:
    def __init__(self, args):
        self.args = args
        self.mod = import_eval_module(args.seedtts_root)
        self.en_asr = None
        self.zh_asr_model = None
        self.zh_asr_processor = None
        self.zh_asr_tokenizer = None
        self.zh_bad_words_ids = None
        self.sim_model = None
        self.sim_feature_extractor = None
        self.sim_prompt_cache = {}

    def _sim_reward_from_value(self, sim: float, req: dict) -> float:
        floor = float(req.get("sim_floor", self.args.sim_floor))
        ceil = float(req.get("sim_ceil", self.args.sim_ceil))
        normalized = (float(sim) - floor) / max(ceil - floor, 1e-6)
        shape = str(req.get("sim_reward_shape", self.args.sim_reward_shape)).lower()
        if shape == "linear":
            return max(0.0, min(1.0, normalized))
        beta = float(req.get("sim_reward_beta", self.args.sim_reward_beta))
        beta = max(beta, 1e-6)
        # Non-saturating around the old floor/ceil range: ceil maps below 1.0,
        # and values above ceil can still improve the relative GRPO signal.
        return 1.0 / (1.0 + math.exp(-beta * (normalized - 0.5)))

    def _ensure_sim_model(self, device: str):
        backend = str(self.args.sim_backend).lower()
        if backend == "cv3_eres2net":
            return self._ensure_cv3_sim_model(device)
        if backend == "omnivoice":
            if self.sim_model is None:
                with contextlib.redirect_stdout(sys.stderr):
                    self.sim_model = self.mod.build_omnivoice_sim_model(
                        self.args.omnivoice_repo,
                        self.args.omnivoice_model_dir,
                        self.args.wavlm_checkpoint,
                        device,
                    )
            return self.sim_model
        if self.sim_model is None:
            with contextlib.redirect_stdout(sys.stderr):
                self.sim_model = self.mod.build_offline_sim_model(
                    self.args.seedtts_root / "thirdparty/UniSpeech/downstreams/speaker_verification",
                    self.args.wavlm_checkpoint,
                    device,
                )
        return self.sim_model

    def _ensure_cv3_sim_model(self, device: str):
        if self.sim_model is not None:
            return self.sim_model
        speakerlab_root = str(self.args.cv3_speakerlab_root)
        if speakerlab_root and speakerlab_root not in sys.path:
            sys.path.insert(0, speakerlab_root)
        import torch
        from speakerlab.models.eres2net.ERes2Net import ERes2Net
        from speakerlab.process.processor import FBank

        checkpoint = pathlib.Path(self.args.cv3_sim_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"CV3 ERes2Net checkpoint not found: {checkpoint}")
        with contextlib.redirect_stdout(sys.stderr):
            model = ERes2Net(feat_dim=80, embedding_size=192)
            state = torch.load(str(checkpoint), map_location="cpu")
            model.load_state_dict(state)
            model.to(device)
            model.eval()
        self.sim_model = model
        self.sim_feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
        return self.sim_model

    def _compute_cv3_embedding(self, wav: pathlib.Path, device: str):
        import torch
        import torchaudio

        model = self._ensure_cv3_sim_model(device)
        waveform, sample_rate = torchaudio.load(str(wav))
        if waveform.shape[0] > 1:
            waveform = waveform[0, :].unsqueeze(0)
        if int(sample_rate) != 16000:
            waveform = torchaudio.functional.resample(waveform, int(sample_rate), 16000)
        assert self.sim_feature_extractor is not None
        feature = self.sim_feature_extractor(waveform).unsqueeze(0).to(device)
        with torch.inference_mode():
            return model(feature)

    def _compute_sim_embedding(self, wav: pathlib.Path, device: str):
        backend = str(self.args.sim_backend).lower()
        if backend == "cv3_eres2net":
            return self._compute_cv3_embedding(wav, device)
        model = self._ensure_sim_model(device)
        if backend == "omnivoice":
            return model([self.mod.load_omnivoice_audio(wav, device)])
        return model(self.mod.load_audio_16k(wav, device).unsqueeze(0))

    def _load_sim_audio(self, wav: pathlib.Path, device: str):
        if str(self.args.sim_backend).lower() == "omnivoice":
            return self.mod.load_omnivoice_audio(wav, device)
        return self.mod.load_audio_16k(wav, device)

    def _build_bad_words_ids(self, tokenizer):
        eos_ids = tokenizer.eos_token_id
        keep_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        bad_ids = set(tokenizer.all_special_ids) - keep_ids
        bad_ids.update(
            token_id
            for token, token_id in tokenizer.get_added_vocab().items()
            if token.startswith("<") and token.endswith(">") and token_id not in keep_ids
        )
        return [[token_id] for token_id in sorted(bad_ids)]

    def _load_ark_zh_asr(self, device: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        torch_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        model_path = str(self.args.ark_asr_path)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Decoder-only batched generation must continue after real prompt tokens,
        # not after right-padding tokens from shorter audio prompts.
        processor.tokenizer.padding_side = "left"
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
        ).to(device)
        model.eval()
        self.zh_asr_processor = processor
        self.zh_asr_tokenizer = tokenizer
        self.zh_asr_model = model
        self.zh_bad_words_ids = self._build_bad_words_ids(tokenizer)

    def _transcribe_ark_zh(self, wav: pathlib.Path, device: str) -> str:
        import torch

        if self.zh_asr_model is None:
            self._load_ark_zh_asr(device)
        processor = self.zh_asr_processor
        tokenizer = self.zh_asr_tokenizer
        model = self.zh_asr_model
        assert processor is not None and tokenizer is not None and model is not None
        torch_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": str(wav)},
                    {"type": "text", "text": "Please transcribe this audio."},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            sampling_rate=16000,
            audio_padding="longest",
            text_kwargs={"padding": "longest"},
            audio_max_length=int(self.args.ark_asr_audio_max_seconds * 16000),
        )
        inputs = inputs.to(device)
        if "audios" in inputs:
            inputs["audios"] = inputs["audios"].to(dtype=torch_dtype)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(self.args.ark_asr_max_new_tokens),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bad_words_ids=self.zh_bad_words_ids,
            )
        decoded = tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        text = decoded[0].strip() if decoded else ""
        if zhconv is not None:
            text = zhconv.convert(text, "zh-cn")
        return text

    def _transcribe_ark_zh_batch(self, wavs: list[pathlib.Path], device: str) -> list[str]:
        import torch

        if not wavs:
            return []
        if self.zh_asr_model is None:
            self._load_ark_zh_asr(device)
        processor = self.zh_asr_processor
        tokenizer = self.zh_asr_tokenizer
        model = self.zh_asr_model
        assert processor is not None and tokenizer is not None and model is not None
        torch_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        batch_size = max(1, int(self.args.ark_asr_batch_size))
        texts = []
        for start in range(0, len(wavs), batch_size):
            chunk = wavs[start : start + batch_size]
            conversations = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "path": str(wav)},
                            {"type": "text", "text": "Please transcribe this audio."},
                        ],
                    }
                ]
                for wav in chunk
            ]
            inputs = processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                return_tensors="pt",
                sampling_rate=16000,
                audio_padding="longest",
                text_kwargs={"padding": "longest"},
                audio_max_length=int(self.args.ark_asr_audio_max_seconds * 16000),
            )
            if int(inputs.input_ids.size(0)) != len(chunk):
                raise RuntimeError(
                    f"ARK-ASR processor batch mismatch: got {inputs.input_ids.size(0)} "
                    f"expected {len(chunk)}"
                )
            inputs = inputs.to(device)
            if "audios" in inputs:
                inputs["audios"] = inputs["audios"].to(dtype=torch_dtype)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=int(self.args.ark_asr_max_new_tokens),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bad_words_ids=self.zh_bad_words_ids,
                )
            decoded = tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            if len(decoded) != len(chunk):
                raise RuntimeError(
                    f"ARK-ASR decode batch mismatch: got {len(decoded)} expected {len(chunk)}"
                )
            for text in decoded:
                text = text.strip()
                if zhconv is not None:
                    text = zhconv.convert(text, "zh-cn")
                texts.append(text)
        return texts

    def _transcribe_whisper_batch(self, wavs: list[pathlib.Path], lang: str, device: str) -> list[str]:
        import scipy.signal
        import soundfile as sf
        import torch

        if not wavs:
            return []
        if self.en_asr is None:
            self.en_asr = self.mod.load_en_asr(self.args.whisper_path, device)
        processor, model = self.en_asr
        arrays = []
        for wav_path in wavs:
            wav, sr = sf.read(str(wav_path))
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
            arrays.append(wav)
        input_features = processor(arrays, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=WHISPER_LANGUAGE_NAMES[lang],
            task="transcribe",
        )
        with torch.inference_mode():
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)

    def _score_sim(self, wav: pathlib.Path, req: dict, device: str):
        prompt_wav = req.get("prompt_wav") or str(self.args.prompt_wav or "")
        if not prompt_wav:
            raise ValueError("prompt_wav is required for sim reward")
        prompt_key = str(pathlib.Path(prompt_wav))
        import torch
        import torch.nn.functional as F

        with torch.inference_mode():
            if prompt_key not in self.sim_prompt_cache:
                self.sim_prompt_cache[prompt_key] = self._compute_sim_embedding(
                    pathlib.Path(prompt_wav), device
                )
            emb = self._compute_sim_embedding(wav, device)
            sim = float(F.cosine_similarity(emb, self.sim_prompt_cache[prompt_key]).cpu().item())
        reward = self._sim_reward_from_value(sim, req)
        return {"score": reward, "sim": sim}

    def score_sim_batch(self, items: list[dict]):
        import torch
        import torch.nn.functional as F

        if not items:
            return []
        device = items[0].get("device") or self.args.device
        if str(self.args.sim_backend).lower() == "cv3_eres2net":
            results = []
            for req in items:
                try:
                    result = self._score_sim(pathlib.Path(req["wav"]), req, device)
                    results.append({"ok": True, "result": result})
                except Exception as exc:
                    results.append({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return results
        sim_model = self._ensure_sim_model(device)
        results = [None for _ in items]
        valid = []
        prompt_keys = []
        with torch.inference_mode():
            for idx, req in enumerate(items):
                try:
                    prompt_wav = req.get("prompt_wav") or str(self.args.prompt_wav or "")
                    if not prompt_wav:
                        raise ValueError("prompt_wav is required for sim reward")
                    prompt_key = str(pathlib.Path(prompt_wav))
                    if prompt_key not in self.sim_prompt_cache:
                        x_prompt = self._load_sim_audio(pathlib.Path(prompt_wav), device)
                        if str(self.args.sim_backend).lower() == "omnivoice":
                            self.sim_prompt_cache[prompt_key] = sim_model([x_prompt])
                        else:
                            self.sim_prompt_cache[prompt_key] = sim_model(x_prompt.unsqueeze(0))
                    valid.append((idx, req, pathlib.Path(req["wav"])))
                    prompt_keys.append(prompt_key)
                except Exception as exc:
                    results[idx] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

            if valid:
                wavs = [self._load_sim_audio(wav, device) for _idx, _req, wav in valid]
                embs = sim_model(wavs)
                prompt_embs = torch.cat([self.sim_prompt_cache[key] for key in prompt_keys], dim=0)
                sims = F.cosine_similarity(embs, prompt_embs, dim=-1).detach().cpu().tolist()
                for (idx, req, _wav), sim in zip(valid, sims):
                    reward = self._sim_reward_from_value(float(sim), req)
                    results[idx] = {"ok": True, "result": {"score": reward, "sim": float(sim)}}
        return results

    def _wer_stats(self, hypo: str, truth: str, lang: str) -> dict:
        punctuation = self.mod.PUNCTUATION_ALL
        if lang == "en":
            truth = self.mod.normalize_en_contractions(truth)
            hypo = self.mod.normalize_en_contractions(hypo)
        for char in punctuation:
            if char == "'":
                continue
            truth = truth.replace(char, "")
            hypo = hypo.replace(char, "")
        truth = truth.replace("  ", " ")
        hypo = hypo.replace("  ", " ")
        if lang in CHAR_METRIC_LANGS:
            truth = " ".join(truth)
            hypo = " ".join(hypo)
        else:
            truth = truth.lower()
            hypo = hypo.lower()
        if self.mod.compute_measures is not None:
            measures = self.mod.compute_measures(truth, hypo)
            error_rate = measures["wer"]
        else:
            error_rate = self.mod.process_words(truth, hypo).wer
        return {"wer": float(error_rate)}

    def _score_asr(self, wav: pathlib.Path, req: dict, device: str):
        lang = normalize_lang(req.get("lang") or self.args.lang)
        text = req.get("text") or self.args.text
        if lang == "zh":
            hypo = self._transcribe_ark_zh(wav, device)
        else:
            hypo = self._transcribe_whisper_batch([wav], lang, device)[0]
        stats = self._wer_stats(hypo, text, lang)
        err = float(stats["wer"])
        alpha = float(req.get("alpha", self.args.alpha))
        reward = max(0.0, 1.0 - max(0.0, err) / max(alpha, 1e-6))
        reward = max(0.0, min(1.0, reward))
        return {
            "score": reward,
            "error_rate": err,
            "wer": None if lang in CHAR_METRIC_LANGS else err,
            "cer": err if lang in CHAR_METRIC_LANGS else None,
            "hypothesis": hypo,
        }

    def _score_asr_from_hypothesis(self, hypo: str, req: dict) -> dict:
        lang = normalize_lang(req.get("lang") or self.args.lang)
        text = req.get("text") or self.args.text
        stats = self._wer_stats(hypo, text, lang)
        err = float(stats["wer"])
        alpha = float(req.get("alpha", self.args.alpha))
        reward = max(0.0, 1.0 - max(0.0, err) / max(alpha, 1e-6))
        reward = max(0.0, min(1.0, reward))
        return {
            "score": reward,
            "error_rate": err,
            "wer": None if lang in CHAR_METRIC_LANGS else err,
            "cer": err if lang in CHAR_METRIC_LANGS else None,
            "hypothesis": hypo,
        }

    def score_asr_batch(self, items: list[dict]):
        if not items:
            return []
        device = items[0].get("device") or self.args.device
        results = [None for _ in items]
        grouped = {lang: [] for lang in sorted(SUPPORTED_ASR_LANGS)}
        for idx, req in enumerate(items):
            lang = normalize_lang(req.get("lang") or self.args.lang)
            grouped[lang].append((idx, req, pathlib.Path(req["wav"])))

        for lang, group in grouped.items():
            if not group:
                continue
            try:
                wavs = [wav for _idx, _req, wav in group]
                if lang == "zh":
                    hypos = self._transcribe_ark_zh_batch(wavs, device)
                else:
                    hypos = self._transcribe_whisper_batch(wavs, lang, device)
                if len(hypos) != len(group):
                    raise RuntimeError(f"{lang} ASR batch size mismatch: got {len(hypos)} expected {len(group)}")
                for (idx, req, _wav), hypo in zip(group, hypos):
                    results[idx] = {"ok": True, "result": self._score_asr_from_hypothesis(hypo, req)}
            except Exception as exc:
                for idx, _req, _wav in group:
                    results[idx] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return results

    def _combine_asr_sim(self, asr_result: dict, sim_result: dict, req: dict):
        asr_score = float(asr_result.get("score", 0.0))
        sim_score = float(sim_result.get("score", 0.0))
        asr_weight = float(req.get("asr_weight", self.args.asr_weight))
        sim_weight = float(req.get("sim_weight", self.args.sim_weight))
        total = max(asr_weight + sim_weight, 1e-6)
        score = (asr_weight * asr_score + sim_weight * sim_score) / total
        out = dict(asr_result)
        out.update({
            "score": max(0.0, min(1.0, score)),
            "asr_score": asr_score,
            "sim_score": sim_score,
            "sim": sim_result.get("sim"),
        })
        return out

    def score(self, req: dict):
        reward_type = req.get("reward_type") or self.args.reward_type
        wav = pathlib.Path(req["wav"])
        device = req.get("device") or self.args.device
        if reward_type == "sim":
            return self._score_sim(wav, req, device)
        if reward_type == "asr_sim":
            sim_result = self._score_sim(wav, req, device)
            asr_result = self._score_asr(wav, req, device)
            return self._combine_asr_sim(asr_result, sim_result, req)
        return self._score_asr(wav, req, device)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--seedtts-root", type=pathlib.Path, required=True)
    p.add_argument("--reward-type", choices=["asr", "sim", "asr_sim"], default="asr")
    p.add_argument("--wav", type=pathlib.Path, default=None)
    p.add_argument("--text", default="")
    p.add_argument("--lang", choices=sorted(SUPPORTED_ASR_LANGS), default="en")
    p.add_argument("--prompt-wav", type=pathlib.Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--whisper-path",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("WHISPER_PATH", "whisper-large-v3")),
    )
    p.add_argument(
        "--ark-asr-path",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("ARK_ASR_PATH", "ark_asr_v1.1")),
    )
    p.add_argument("--ark-asr-max-new-tokens", type=int, default=256)
    p.add_argument("--ark-asr-audio-max-seconds", type=float, default=30.0)
    p.add_argument("--ark-asr-batch-size", type=int, default=32)
    p.add_argument("--hf-modules-cache", type=pathlib.Path, default=pathlib.Path("/dev/shm/fish_grpo_hf_modules"))
    p.add_argument(
        "--wavlm-checkpoint",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("WAVLM_CHECKPOINT", "wavlm_large_finetune.pth")),
    )
    p.add_argument(
        "--sim-backend",
        choices=["omnivoice", "seedtts_wavlm", "cv3_eres2net"],
        default="seedtts_wavlm",
    )
    p.add_argument(
        "--omnivoice-repo",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("OMNIVOICE_REPO", "OmniVoice")),
    )
    p.add_argument(
        "--omnivoice-model-dir",
        type=pathlib.Path,
        default=pathlib.Path(
            os.environ.get("OMNIVOICE_MODEL_DIR", "OmniVoice/download/tts_eval_models")
        ),
    )
    p.add_argument(
        "--cv3-root",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("CV3_ROOT", "CV3-Eval-main")),
    )
    p.add_argument(
        "--cv3-speakerlab-root",
        type=pathlib.Path,
        default=pathlib.Path(
            os.environ.get("CV3_SPEAKERLAB_ROOT", "CV3-Eval-main/utils/3D-Speaker")
        ),
    )
    p.add_argument(
        "--cv3-sim-checkpoint",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("CV3_SIM_CHECKPOINT", "pretrained_eres2net.ckpt")),
    )
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--asr-weight", type=float, default=0.5)
    p.add_argument("--sim-weight", type=float, default=0.5)
    p.add_argument("--sim-floor", type=float, default=0.35)
    p.add_argument("--sim-ceil", type=float, default=0.75)
    p.add_argument("--sim-reward-shape", choices=["linear", "logistic"], default="logistic")
    p.add_argument("--sim-reward-beta", type=float, default=5.0)
    p.add_argument(
        "--extra-pythonpath",
        default="",
        help="Optional extra source/dependency path prepended by reward worker.",
    )
    p.add_argument("--server", action="store_true")
    return p


def score_batch(engine: RewardEngine, items: list[dict]):
    if items and all((item.get("reward_type") or engine.args.reward_type) == "sim" for item in items):
        try:
            return engine.score_sim_batch(items)
        except Exception as exc:
            return [{"ok": False, "error": f"{type(exc).__name__}: {exc}"} for _ in items]
    if items and all((item.get("reward_type") or engine.args.reward_type) == "asr_sim" for item in items):
        device = items[0].get("device") or engine.args.device
        if any((normalize_lang(item.get("lang") or engine.args.lang) == "zh") for item in items) and engine.zh_asr_model is None:
            engine._load_ark_zh_asr(device)
        if any((normalize_lang(item.get("lang") or engine.args.lang) != "zh") for item in items) and engine.en_asr is None:
            engine.en_asr = engine.mod.load_en_asr(engine.args.whisper_path, device)
        engine._ensure_sim_model(device)
        def run_sim():
            try:
                return engine.score_sim_batch(items)
            except Exception as exc:
                return [{"ok": False, "error": f"{type(exc).__name__}: {exc}"} for _ in items]
        def run_asr():
            try:
                return engine.score_asr_batch(items)
            except Exception as exc:
                return [{"ok": False, "error": f"{type(exc).__name__}: {exc}"} for _ in items]
        with ThreadPoolExecutor(max_workers=2) as executor:
            sim_future = executor.submit(run_sim)
            asr_future = executor.submit(run_asr)
            sim_results = sim_future.result()
            asr_results = asr_future.result()
        results = []
        for item, sim_item, asr_item in zip(items, sim_results, asr_results):
            if not sim_item.get("ok"):
                results.append(sim_item)
                continue
            if not asr_item.get("ok"):
                results.append(asr_item)
                continue
            combined = engine._combine_asr_sim(asr_item["result"], sim_item["result"], item)
            results.append({"ok": True, "result": combined})
        return results
    results = []
    for item in items:
        try:
            results.append({"ok": True, "result": engine.score(item)})
        except Exception as exc:
            results.append({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return results


def main():
    args = build_parser().parse_args()
    apply_extra_pythonpath(args.extra_pythonpath)
    apply_hf_modules_cache(args.hf_modules_cache)
    disable_transformers_librosa()
    protocol_stdout = sys.stdout
    if args.server:
        sys.stdout = sys.stderr
    engine = RewardEngine(args)
    if args.server:
        print(json.dumps({"ready": True}), file=protocol_stdout, flush=True)
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                req = json.loads(line)
                if req.get("cmd") == "stop":
                    print(json.dumps({"ok": True}), file=protocol_stdout, flush=True)
                    return
                if req.get("cmd") == "score_batch":
                    res = score_batch(engine, list(req.get("items") or []))
                    print(json.dumps({"ok": True, "results": res}, ensure_ascii=False), file=protocol_stdout, flush=True)
                    continue
                res = engine.score(req)
                print(json.dumps({"ok": True, "result": res}, ensure_ascii=False), file=protocol_stdout, flush=True)
            except Exception as exc:
                print(json.dumps({
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }, ensure_ascii=False), file=protocol_stdout, flush=True)
        return

    if args.wav is None:
        raise ValueError("--wav is required unless --server is used")
    req = {
        "reward_type": args.reward_type,
        "wav": str(args.wav),
        "text": args.text,
        "lang": args.lang,
        "prompt_wav": str(args.prompt_wav or ""),
        "device": args.device,
        "alpha": args.alpha,
        "asr_weight": args.asr_weight,
        "sim_weight": args.sim_weight,
        "sim_floor": args.sim_floor,
        "sim_ceil": args.sim_ceil,
        "sim_reward_shape": args.sim_reward_shape,
        "sim_reward_beta": args.sim_reward_beta,
    }
    print(json.dumps(engine.score(req), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
