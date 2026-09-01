from __future__ import annotations

import base64
import gc
import io
import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path

try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None  # type: ignore[assignment]

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .registration import MAX_AUDIO_BYTES, VoiceRegistration
from .runtime import ArkTtsRuntime
from .web_ui import TEST_PAGE

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("ARKTTS_MODEL_DIR", str(ROOT / "model"))).expanduser()
VOICES_DIR = Path(os.getenv("ARKTTS_VOICES_DIR", str(ROOT / "voices"))).expanduser()
REGISTRATION_DIR = Path(
    os.getenv("ARKTTS_REGISTRATION_DIR", str(MODEL_DIR / "registration"))
).expanduser()
PRECISION = os.getenv("ARKTTS_PRECISION") or None
CODEC_PRECISION = os.getenv("ARKTTS_CODEC_PRECISION") or None
THREADS = int(os.getenv("ARKTTS_THREADS", "5"))

app = FastAPI(title="Audio8 0.1B INT8 ONNX Runtime")
runtime: ArkTtsRuntime | None = None
registration: VoiceRegistration | None = None
request_lock = threading.RLock()
active_stop: threading.Event | None = None
service_started_at = time.monotonic()


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    voice_name: str = "default"
    max_new_tokens: int = Field(1024, ge=16, le=2048)
    temperature: float = Field(0.7, gt=0.0, le=2.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    top_k: int = Field(50, ge=1, le=4096)
    seed: int = 42


class OpenAiSpeechRequest(BaseModel):
    model: str = "arktts-0.1b"
    input: str = Field(..., min_length=1, max_length=1000)
    voice: str = "default"
    response_format: str = "wav"


@app.get("/", response_class=HTMLResponse)
def test_page() -> str:
    return TEST_PAGE


def _load_runtime() -> ArkTtsRuntime:
    global runtime, registration
    runtime = ArkTtsRuntime(MODEL_DIR, VOICES_DIR, PRECISION, CODEC_PRECISION, THREADS)
    registration = VoiceRegistration(
        REGISTRATION_DIR,
        runtime.voices.root,
        str(runtime.manifest["model_fingerprint"]),
    )
    return runtime


@app.on_event("startup")
def startup() -> None:
    _load_runtime()


def require_runtime() -> ArkTtsRuntime:
    if runtime is None:
        raise HTTPException(503, "runtime is not loaded")
    return runtime


def _wav_response(audio: np.ndarray, sample_rate: int) -> Response:
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(audio, dtype=np.float32), int(sample_rate), format="WAV")
    return Response(buffer.getvalue(), media_type="audio/wav")


@app.get("/api/health")
def health() -> dict:
    obj = require_runtime()
    return {
        "ok": True,
        "model": obj.manifest.get("model_id"),
        "precision": obj.precision,
        "codec_precision": obj.codec_precision,
        "providers": {
            "slow": obj.slow.get_providers(),
            "fast": obj.fast.get_providers(),
            "decoder": obj.decoder.get_providers(),
        },
    }


@app.get("/api/voices")
def voices() -> dict:
    return {"voices": require_runtime().voices.list()}


@app.get("/api/registration/status")
def registration_status() -> dict:
    if registration is None:
        raise HTTPException(503, "registration is not initialized")
    return registration.status()


def _linux_memory_mb() -> tuple[float | None, float | None]:
    try:
        values: dict[str, float] = {}
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition(":")
            if separator and name in {"VmRSS", "VmHWM"}:
                values[name] = float(value.strip().split()[0]) / 1024
        return values.get("VmRSS"), values.get("VmHWM")
    except (OSError, ValueError, IndexError):
        return None, None


def _mac_current_memory_mb() -> float | None:
    try:
        result = subprocess.run(
            ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
        return float(result.stdout.strip()) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


@app.get("/api/system")
def system_stats() -> dict:
    current_mb, peak_mb = _linux_memory_mb()
    system = platform.system()
    if current_mb is None and system == "Darwin":
        current_mb = _mac_current_memory_mb()
    if peak_mb is None and resource is not None:
        usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak_mb = usage / (1024 * 1024) if system == "Darwin" else usage / 1024
    return {
        "memory": {"current_mb": current_mb, "peak_mb": peak_mb},
        "uptime_seconds": time.monotonic() - service_started_at,
        "refresh_interval_ms": 2000,
    }


@app.post("/api/voices/register")
def register_voice(
    audio: UploadFile = File(...),
    text: str = Form(...),
    name: str = Form(...),
    overwrite: bool = Form(False),
) -> dict:
    global runtime
    if registration is None:
        raise HTTPException(503, "registration is not initialized")
    data = audio.file.read(MAX_AUDIO_BYTES + 1)
    with request_lock:
        current = require_runtime()
        precision, codec_precision = current.precision, current.codec_precision
        runtime = None
        del current
        gc.collect()
        try:
            meta = registration.register(
                data,
                audio.filename or "reference_audio",
                text,
                name,
                overwrite,
            )
        except FileExistsError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        finally:
            runtime = ArkTtsRuntime(MODEL_DIR, VOICES_DIR, precision, codec_precision, THREADS)
            gc.collect()
    return {"ok": True, "voice": meta}


@app.post("/api/tts")
def tts(request: TtsRequest) -> Response:
    with request_lock:
        obj = require_runtime()
        audio, _ = obj.synthesize(
            text=request.text,
            voice=request.voice_name,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            seed=request.seed,
        )
        return _wav_response(audio, int(obj.manifest["sample_rate"]))


@app.post("/v1/audio/speech")
def openai_speech(request: OpenAiSpeechRequest) -> Response:
    if request.model not in {"arktts", "arktts-0.1b", "tts-1"}:
        raise HTTPException(400, f"unsupported model: {request.model}")
    if request.response_format not in {"wav", "pcm"}:
        raise HTTPException(400, "response_format must be wav or pcm")
    with request_lock:
        obj = require_runtime()
        audio, _ = obj.synthesize(text=request.input, voice=request.voice)
        if request.response_format == "pcm":
            pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            return Response(pcm, media_type="application/octet-stream")
        return _wav_response(audio, int(obj.manifest["sample_rate"]))


@app.post("/api/tts/stream")
def stream_tts(request: TtsRequest) -> StreamingResponse:
    def events():
        global active_stop
        stop = threading.Event()
        active_stop = stop
        try:
            with request_lock:
                obj = require_runtime()
                yield json.dumps(
                    {
                        "event": "start",
                        "sample_rate": int(obj.manifest["sample_rate"]),
                        "channels": 1,
                        "sample_format": "s16le",
                    }
                ) + "\n"
                for event in obj.stream(
                    text=request.text,
                    voice=request.voice_name,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    seed=request.seed,
                    stop_event=stop,
                ):
                    if stop.is_set():
                        yield json.dumps({"event": "cancelled"}) + "\n"
                        return
                    if event["type"] == "audio_chunk":
                        pcm = (np.clip(event["audio"], -1.0, 1.0) * 32767.0).astype("<i2")
                        yield json.dumps(
                            {
                                "event": "audio_chunk",
                                "seq": int(event["seq"]),
                                "frame_count": int(event["frame_count"]),
                                "pcm_b64": base64.b64encode(pcm.tobytes()).decode("ascii"),
                            }
                        ) + "\n"
                    else:
                        yield json.dumps(
                            {"event": "complete", "frame_count": int(event["codes"].shape[1])}
                        ) + "\n"
        finally:
            if active_stop is stop:
                active_stop = None

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/api/tts/cancel")
def cancel() -> dict[str, bool]:
    if active_stop is None:
        return {"ok": True, "cancelled": False}
    active_stop.set()
    return {"ok": True, "cancelled": True}


@app.post("/api/runtime/reload")
def reload_runtime() -> dict:
    global active_stop, runtime, registration
    if active_stop is not None:
        active_stop.set()
    started = time.monotonic()
    with request_lock:
        runtime = None
        registration = None
        gc.collect()
        obj = _load_runtime()
    return {
        "ok": True,
        "precision": obj.precision,
        "codec_precision": obj.codec_precision,
        "reload_seconds": time.monotonic() - started,
    }
