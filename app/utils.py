import json
import shutil
import subprocess
from pathlib import Path


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a command and raise a readable error if it fails."""
    proc = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"{Path(cmd[0]).name} failed:\n{tail}")


def probe(path: Path) -> dict:
    """Return duration, width, height and fps of a video file."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
         "-show_format", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"Could not read video file: {out.stderr.strip()[-300:]}")
    info = json.loads(out.stdout)
    video = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError("That file has no video track.")
    has_audio = any(s.get("codec_type") == "audio" for s in info["streams"])
    w, h = int(video["width"]), int(video["height"])
    # Respect phone-video rotation metadata
    rot = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = abs(int(sd["rotation"]))
    if rot in (90, 270):
        w, h = h, w
    num, _, den = video.get("avg_frame_rate", "30/1").partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 30.0
    duration = float(info["format"].get("duration") or video.get("duration") or 0)
    return {"duration": duration, "width": w, "height": h,
            "fps": fps or 30.0, "has_audio": has_audio}


def fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
