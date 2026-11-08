"""Speech to text with word-level timestamps using faster-whisper (runs locally, free)."""
import os
import sys
import threading
import traceback
import wave
from pathlib import Path

import numpy as np

from .config import MODELS_DIR, WHISPER_MODELS
from .utils import run

_models: dict = {}
_lock = threading.Lock()
_gpu_broken = False   # set once the GPU fails, so we stop trying it
active_device = None  # "cuda" or "cpu", shown on the dashboard


def _add_nvidia_dll_dirs():
    """If the optional nvidia-cublas/cudnn pip packages are installed, let Windows find their DLLs."""
    if sys.platform != "win32":
        return
    try:
        import nvidia  # namespace package from nvidia-*-cu12 wheels
    except ImportError:
        return
    for base in getattr(nvidia, "__path__", []):
        for sub in Path(base).glob("*/bin"):
            try:
                os.add_dll_directory(str(sub))
                os.environ["PATH"] = str(sub) + os.pathsep + os.environ.get("PATH", "")
            except OSError:
                pass


_add_nvidia_dll_dirs()


def gpu_available() -> bool:
    if _gpu_broken or os.getenv("FORCE_CPU", "").strip() in ("1", "true", "yes"):
        return False
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _new_model(name: str, device: str):
    from faster_whisper import WhisperModel
    return WhisperModel(name, device=device,
                        compute_type="float16" if device == "cuda" else "int8",
                        download_root=str(MODELS_DIR / "whisper"))


def _gpu_works(model) -> bool:
    """A GPU model can load fine and then crash on first use when CUDA DLLs
    (cuBLAS/cuDNN) are missing, so run a tiny test transcription first."""
    try:
        segs, _ = model.transcribe(np.zeros(16000 * 2, dtype=np.float32), language="en", beam_size=1)
        list(segs)
        return True
    except Exception:
        traceback.print_exc()
        return False


def _load(name: str):
    global _gpu_broken, active_device
    with _lock:
        if name in _models:
            return _models[name]
        m = None
        if gpu_available():
            try:
                m = _new_model(name, "cuda")
                if not _gpu_works(m):
                    m = None
            except Exception:
                traceback.print_exc()
                m = None
            if m is None:
                _gpu_broken = True
                print("\n[AutoClipper] Your NVIDIA GPU couldn't be used (CUDA libraries missing). "
                      "Using the CPU instead. This works fine, just slower.\n", flush=True)
        if m is None:
            m = _new_model(name, "cpu")
            active_device = "cpu"
        else:
            active_device = "cuda"
        _models[name] = m
        return m


def _run(model, audio, duration, on_progress):
    segments, _info = model.transcribe(
        audio, word_timestamps=True, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        beam_size=5, condition_on_previous_text=False,
    )
    out = []
    for seg in segments:
        words = [{"start": float(w.start), "end": float(w.end), "word": w.word.strip()}
                 for w in (seg.words or []) if w.word.strip()]
        if not words:
            continue
        out.append({"start": words[0]["start"], "end": words[-1]["end"],
                    "text": seg.text.strip(), "words": words})
        if on_progress and duration:
            on_progress(min(1.0, seg.end / duration))
    return out


def transcribe(video: Path, work: Path, quality: str, duration: float, on_progress=None) -> list[dict]:
    """Returns a list of segments: {start, end, text, words:[{start,end,word}]}"""
    global _gpu_broken, active_device
    wav = work / "audio.wav"
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])

    # Load the audio ourselves instead of letting faster-whisper decode the file:
    # newer PyAV versions break faster-whisper's own decoder.
    with wave.open(str(wav)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0

    model_name = WHISPER_MODELS.get(quality, WHISPER_MODELS["balanced"])
    try:
        model = _load(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Couldn't load the speech model '{model_name}'. The first run downloads it from "
            f"huggingface.co, so you need an internet connection.\n\nDetails: {e}") from None

    try:
        return _run(model, audio, duration, on_progress)
    except Exception:
        if active_device != "cuda":
            raise
        # GPU failed partway through: switch to CPU for good and redo this video
        traceback.print_exc()
        print("\n[AutoClipper] GPU failed during transcription, retrying on the CPU.\n", flush=True)
        with _lock:
            _gpu_broken = True
            _models.clear()
        model = _load(model_name)
        return _run(model, audio, duration, on_progress)
