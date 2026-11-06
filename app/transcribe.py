"""Speech to text with word-level timestamps using faster-whisper (runs locally, free)."""
import threading
from pathlib import Path

from .config import MODELS_DIR, WHISPER_MODELS
from .utils import run

_models: dict = {}
_lock = threading.Lock()


def _device():
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _load(name: str):
    from faster_whisper import WhisperModel
    with _lock:
        if name in _models:
            return _models[name]
        device, compute = _device()
        try:
            m = WhisperModel(name, device=device, compute_type=compute,
                             download_root=str(MODELS_DIR / "whisper"))
        except Exception:
            if device == "cpu":
                raise
            # GPU present but CUDA libraries missing: fall back to CPU
            m = WhisperModel(name, device="cpu", compute_type="int8",
                             download_root=str(MODELS_DIR / "whisper"))
        _models[name] = m
        return m


def transcribe(video: Path, work: Path, quality: str, duration: float, on_progress=None) -> list[dict]:
    """Returns a list of segments: {start, end, text, words:[{start,end,word}]}"""
    wav = work / "audio.wav"
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)])

    model_name = WHISPER_MODELS.get(quality, WHISPER_MODELS["balanced"])
    try:
        model = _load(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Couldn't load the speech model '{model_name}'. The first run downloads it from "
            f"huggingface.co, so you need an internet connection.\n\nDetails: {e}") from None
    segments, _info = model.transcribe(
        str(wav), word_timestamps=True, vad_filter=True,
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
