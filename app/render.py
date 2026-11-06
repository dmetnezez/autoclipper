"""Render one vertical clip with FFmpeg."""
import os
from pathlib import Path

from .config import FONTS_DIR, OUT_H, OUT_W, VIDEO_CRF, VIDEO_PRESET
from .reframe import crop_expr
from .utils import run


def render_clip(src: Path, out_dir: Path, name: str, start: float, end: float,
                layout: str, segs, ass_text: str, info: dict) -> tuple[Path, Path]:
    ass_file = out_dir / f"{name}.ass"
    ass_file.write_text(ass_text, encoding="utf-8")
    out = out_dir / f"{name}.mp4"
    thumb = out_dir / f"{name}.jpg"

    # Relative paths avoid Windows drive-letter escaping problems in the ass filter
    subs = f"ass=filename={ass_file.name}"
    try:
        rel_fonts = os.path.relpath(FONTS_DIR, out_dir).replace("\\", "/")
        if any(FONTS_DIR.glob("*.[ot]tf")):
            subs += f":fontsdir={rel_fonts}"
    except ValueError:
        pass

    if layout == "crop":
        cw = int(info["height"] * 9 / 16) // 2 * 2
        vf = (f"[0:v]crop=w={cw}:h=ih:x='{crop_expr(segs)}':y=0,"
              f"scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1,fps=30,{subs}[v]")
    else:
        vf = (f"[0:v]fps=30,split=2[a][b];"
              f"[a]scale={OUT_W // 4}:{OUT_H // 4}:force_original_aspect_ratio=increase,"
              f"crop={OUT_W // 4}:{OUT_H // 4},boxblur=12:3,eq=brightness=-0.12:saturation=1.2,"
              f"scale={OUT_W}:{OUT_H}:flags=bicubic[bg];"
              f"[b]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,{subs}[v]")

    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{end - start:.3f}",
           "-filter_complex", vf, "-map", "[v]"]
    if info.get("has_audio"):
        cmd += ["-map", "0:a:0", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000",
                "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-preset", VIDEO_PRESET, "-crf", str(VIDEO_CRF),
            "-profile:v", "high", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            out.name]
    run(cmd, cwd=out_dir)

    run(["ffmpeg", "-y", "-ss", "1.0", "-i", out.name, "-frames:v", "1",
         "-vf", "scale=360:-2", "-q:v", "4", thumb.name], cwd=out_dir)
    return out, thumb
