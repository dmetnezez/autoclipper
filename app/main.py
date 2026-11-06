"""AutoClipper web server. Run with: python -m uvicorn app.main:app"""
import re
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import db, pipeline
from .config import ANTHROPIC_API_KEY, DATA_DIR
from .utils import have_ffmpeg

app = FastAPI(title="AutoClipper")
STATIC = Path(__file__).parent / "static"
ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


@app.on_event("startup")
def _startup():
    db.init()
    pipeline.start_worker()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    try:
        import ctranslate2
        gpu = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        gpu = False
    return {"ffmpeg": have_ffmpeg(), "ai": bool(ANTHROPIC_API_KEY), "gpu": gpu}


@app.post("/api/jobs")
async def create_job(
    url: str = Form(""),
    file: UploadFile | None = File(None),
    count: int = Form(5),
    min_len: int = Form(20),
    max_len: int = Form(60),
    captions: str = Form("pop"),
    layout: str = Form("auto"),
    quality: str = Form("balanced"),
    hook: bool = Form(True),
):
    if not have_ffmpeg():
        raise HTTPException(400, "FFmpeg isn't installed. Run: winget install Gyan.FFmpeg, then restart.")
    url = url.strip()
    if not url and (file is None or not file.filename):
        raise HTTPException(400, "Paste a link or choose a video file.")
    if url and not re.match(r"^https?://", url):
        raise HTTPException(400, "That doesn't look like a link. It should start with https://")
    min_len, max_len = max(5, min_len), max(min_len + 5, min(max_len, 180))
    settings = {
        "count": max(1, min(count, 20)), "min_len": min_len, "max_len": max_len,
        "captions": captions if captions in ("pop", "green", "clean", "none") else "pop",
        "layout": layout if layout in ("auto", "crop", "fit") else "auto",
        "quality": quality if quality in ("fast", "balanced", "best") else "balanced",
        "hook": hook,
    }
    if file is not None and file.filename and not url:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(400, f"Unsupported file type {ext}. Use mp4, mov, mkv or webm.")
        settings.update(upload=f"source{ext}", filename=file.filename)
        jid = db.create_job(file.filename, settings)
        with open(pipeline.job_dir(jid) / f"source{ext}", "wb") as f:
            shutil.copyfileobj(file.file, f, length=1024 * 1024)
    else:
        jid = db.create_job(url, settings)
    pipeline.enqueue(jid)
    return {"id": jid}


@app.get("/api/jobs")
def jobs():
    return db.list_jobs()


@app.get("/api/jobs/{jid}")
def job(jid: str):
    j = db.get_job(jid)
    if not j:
        raise HTTPException(404, "Not found")
    return j


@app.delete("/api/jobs/{jid}")
def delete(jid: str):
    j = db.get_job(jid)
    if j and j["status"] in ("running", "queued"):
        raise HTTPException(400, "Wait for this job to finish before deleting it.")
    db.delete_job(jid)
    pipeline.remove_job_files(jid)
    return {"ok": True}


@app.get("/media/{jid}/{name}")
def media(jid: str, name: str, download: int = 0):
    if not re.fullmatch(r"[a-f0-9]{10}", jid) or not re.fullmatch(r"clip_\d+\.(mp4|jpg)", name):
        raise HTTPException(404)
    path = DATA_DIR / "jobs" / jid / name
    if not path.exists():
        raise HTTPException(404)
    j = db.get_job(jid)
    fname = name
    if download and j:
        clip = next((c for c in j["clips"] if c["file"] == name), None)
        if clip:
            safe = re.sub(r"[^\w\- ]", "", clip["title"]).strip()[:60] or name
            fname = f"{safe}.mp4"
    return FileResponse(path, filename=fname if download else None)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
