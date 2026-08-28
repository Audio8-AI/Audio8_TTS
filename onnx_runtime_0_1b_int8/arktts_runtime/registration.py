from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from math import gcd
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

MAX_AUDIO_BYTES = 50 * 1024 * 1024
MAX_AUDIO_SECONDS = 30.0
MIN_AUDIO_SECONDS = 0.5


class VoiceRegistration:
    """Optional reference-audio encoder used to create a voice profile."""

    def __init__(self, registration_dir: Path, voices_root: Path, model_fingerprint: str):
        self.root = Path(registration_dir).resolve()
        self.voices_root = Path(voices_root).resolve()
        self.encoder_path = self.root / "codec_encoder_fp16.onnx"
        self.manifest_path = self.root / "registration_manifest.json"
        self.expected_fingerprint = model_fingerprint

    def status(self) -> dict[str, str | bool | None]:
        available = self.encoder_path.is_file() and self.manifest_path.is_file()
        reason: str | None = None
        if available:
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if manifest.get("model_fingerprint") != self.expected_fingerprint:
                    available = False
                    reason = "registration model fingerprint does not match online model"
            except (OSError, json.JSONDecodeError):
                available = False
                reason = "registration manifest is invalid"
        else:
            reason = "registration package was not found"
        return {"available": available, "reason": reason}

    @staticmethod
    def _validate_name(name: str) -> str:
        value = name.strip()
        if not value or len(value) > 64 or value in {".", ".."} or Path(value).name != value:
            raise ValueError("voice name must be one path component with at most 64 characters")
        return value

    @staticmethod
    def _decode_audio(data: bytes, target_rate: int) -> tuple[np.ndarray, int]:
        if not data or len(data) > MAX_AUDIO_BYTES:
            raise ValueError("audio file must be between 1 byte and 50 MiB")
        try:
            audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        except (RuntimeError, sf.LibsndfileError) as exc:
            raise ValueError(f"unsupported or invalid audio file: {exc}") from exc
        audio = audio.mean(axis=1)
        source_rate = int(sample_rate)
        duration = audio.size / max(1, source_rate)
        if duration < MIN_AUDIO_SECONDS or duration > MAX_AUDIO_SECONDS:
            raise ValueError("reference audio duration must be between 0.5 and 30 seconds")
        if not np.isfinite(audio).all():
            raise ValueError("reference audio contains non-finite samples")
        if source_rate != int(target_rate):
            from scipy.signal import resample_poly

            factor = gcd(source_rate, int(target_rate))
            audio = resample_poly(
                audio,
                int(target_rate) // factor,
                source_rate // factor,
            ).astype(np.float32)
        padding = (-audio.size) % 2048
        if padding:
            audio = np.pad(audio, (0, padding))
        return np.ascontiguousarray(audio.reshape(1, 1, -1)), source_rate

    def register(
        self,
        data: bytes,
        filename: str,
        text: str,
        name: str,
        overwrite: bool = False,
    ) -> dict:
        state = self.status()
        if not state["available"]:
            raise RuntimeError(str(state["reason"]))
        voice_name = self._validate_name(name)
        reference_text = " ".join(text.strip().split())
        if not reference_text:
            raise ValueError("reference text must not be empty")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        audio, source_rate = self._decode_audio(data, int(manifest["sample_rate"]))
        target = self.voices_root / voice_name
        if target.exists() and not overwrite:
            raise FileExistsError(f"voice already exists: {voice_name}")

        options = ort.SessionOptions()
        options.log_severity_level = 3
        session = None
        try:
            session = ort.InferenceSession(
                str(self.encoder_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            input_info = session.get_inputs()[0]
            input_type = np.float16 if input_info.type == "tensor(float16)" else np.float32
            codes = np.asarray(session.run(None, {input_info.name: audio.astype(input_type)})[0])
        finally:
            del session
            gc.collect()

        if codes.ndim == 3:
            codes = codes[0]
        expected_codebooks = int(manifest["num_codebooks"])
        codebook_size = int(manifest.get("codebook_size", 4096))
        if (
            codes.ndim != 2
            or codes.shape[0] != expected_codebooks
            or codes.shape[1] == 0
            or int(codes.min()) < 0
            or int(codes.max()) >= codebook_size
        ):
            raise RuntimeError(f"encoder returned invalid codes: {codes.shape}")

        self.voices_root.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=f".{voice_name}.", dir=self.voices_root))
        try:
            np.save(temp / "codes.npy", codes.astype(np.uint16))
            meta = {
                "name": voice_name,
                "reference_text": reference_text,
                "shape": list(codes.shape),
                "dtype": "uint16",
                "sample_rate": int(manifest["sample_rate"]),
                "source_audio": Path(filename or "reference_audio").name,
                "source_sample_rate": source_rate,
                "source_sha256": hashlib.sha256(data).hexdigest(),
                "model_fingerprint": manifest["model_fingerprint"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_kind": "onnx_runtime_registration",
            }
            (temp / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if target.exists():
                shutil.rmtree(target)
            os.replace(temp, target)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
        return meta
