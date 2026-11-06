"""Get the source video: from a link (yt-dlp) or an uploaded file."""
from pathlib import Path

import yt_dlp


def download(url: str, dest_dir: Path, on_progress=None) -> tuple[Path, str]:
    """Download a video link. Returns (file path, video title)."""
    def hook(d):
        if on_progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                on_progress(d.get("downloaded_bytes", 0) / total)

    opts = {
        # Best quality up to 1080p, merged to mp4
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(dest_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "retries": 5,
        "fragment_retries": 5,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as e:
        msg = str(e).replace("ERROR: ", "")
        raise RuntimeError(
            "Couldn't download that link. Check that it's a public video, or try uploading "
            "the file instead. If YouTube links keep failing, update yt-dlp by restarting "
            f"start.bat.\n\nDetails: {msg[:400]}") from None

    files = sorted(dest_dir.glob("source.*"), key=lambda p: p.stat().st_size, reverse=True)
    files = [f for f in files if f.suffix not in (".part", ".ytdl")]
    if not files:
        raise RuntimeError("Download finished but no video file was found.")
    return files[0], (info or {}).get("title") or url
