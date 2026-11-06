"""Runs a clipping job from start to finish in a background thread."""
import json
import queue
import shutil
import threading
import traceback
from pathlib import Path

from . import db
from .captions import build_ass
from .config import DATA_DIR
from .download import download
from .highlights import find_highlights
from .reframe import analyze, choose_layout, crop_path
from .render import render_clip
from .transcribe import transcribe
from .utils import probe

_queue: "queue.Queue[str]" = queue.Queue()


def job_dir(jid: str) -> Path:
    d = DATA_DIR / "jobs" / jid
    d.mkdir(parents=True, exist_ok=True)
    return d


def enqueue(jid: str):
    _queue.put(jid)


def start_worker():
    threading.Thread(target=_worker, daemon=True).start()


def _worker():
    while True:
        jid = _queue.get()
        try:
            _process(jid)
        except Exception as e:
            traceback.print_exc()
            db.update_job(jid, status="error", error=str(e), stage="Failed")


def _process(jid: str):
    job = db.get_job(jid)
    if not job:
        return
    s = job["settings"]
    work = job_dir(jid)

    def stage(name, lo, hi):
        db.update_job(jid, stage=name, progress=lo)
        return lambda f: db.update_job(jid, progress=round(lo + (hi - lo) * f, 3))

    db.update_job(jid, status="running")

    # 1. Get the video
    if s.get("upload"):
        src = work / s["upload"]
        title = Path(s.get("filename") or src.name).stem
    else:
        src, title = download(job["source"], work, stage("Downloading video", 0.0, 0.10))
    db.update_job(jid, title=title)
    info = probe(src)
    db.update_job(jid, duration=info["duration"])
    if not info["has_audio"]:
        raise RuntimeError("This video has no audio, so there's no speech to find clips from.")

    # 2. Transcribe
    on_p = stage("Transcribing speech (first run downloads the speech model)", 0.10, 0.45)
    segments = transcribe(src, work, s.get("quality", "balanced"), info["duration"], on_p)
    (work / "transcript.json").write_text(json.dumps(segments), encoding="utf-8")

    # 3. Find the best moments
    stage("Finding the best moments", 0.45, 0.52)
    clips, used_ai, ai_error = find_highlights(
        segments, title, int(s.get("count", 5)), float(s.get("min_len", 20)),
        float(s.get("max_len", 60)), info["duration"], work / "audio.wav")
    db.update_job(jid, ai_used=1 if used_ai else 0)
    if ai_error:
        db.update_job(jid, error=f"Claude wasn't reachable, used the offline picker instead: {ai_error[:300]}")
    if not clips:
        raise RuntimeError("Couldn't find any good clips in this video. Try a shorter minimum length.")

    # 4. Reframe + caption + render each clip
    words = [w for seg in segments for w in seg["words"]]
    src_aspect = info["width"] / info["height"]
    n = len(clips)
    for i, c in enumerate(clips):
        lo, hi = 0.52 + 0.48 * i / n, 0.52 + 0.48 * (i + 1) / n
        on_p = stage(f"Editing clip {i + 1} of {n}", lo, hi)
        samples = analyze(src, c["start"], c["end"], info["fps"])
        on_p(0.25)
        layout = choose_layout(samples, info["width"], info["height"], s.get("layout", "auto"))
        segs = crop_path(samples, info["width"], info["height"], c["end"] - c["start"]) if layout == "crop" else None
        ass = build_ass(words, c["start"], c["end"], c["hook"] if s.get("hook", True) else "",
                        s.get("captions", "pop"), "crop" if layout == "crop" else "fit", src_aspect)
        out, thumb = render_clip(src, work, f"clip_{i + 1}", c["start"], c["end"], layout, segs, ass, info)
        db.add_clip(jid, idx=i + 1, start=c["start"], end=c["end"], title=c["title"], hook=c["hook"],
                    score=c["score"], reason=c["reason"], description=c["description"],
                    file=out.name, thumb=thumb.name)
        on_p(1.0)

    if not s.get("keep_source", False):
        src.unlink(missing_ok=True)
    (work / "audio.wav").unlink(missing_ok=True)
    db.update_job(jid, status="done", stage="Done", progress=1.0)


def remove_job_files(jid: str):
    shutil.rmtree(DATA_DIR / "jobs" / jid, ignore_errors=True)
