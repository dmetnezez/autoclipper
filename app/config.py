"""Central settings. Values can be overridden in the .env file."""
import os
import platform
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data")).resolve()
MODELS_DIR = ROOT / "models"
FONTS_DIR = ROOT / "fonts"
DB_PATH = DATA_DIR / "autoclipper.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FONTS_DIR.mkdir(parents=True, exist_ok=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")

_default_font = "Arial Black" if platform.system() == "Windows" else "DejaVu Sans"
CAPTION_FONT = os.getenv("CAPTION_FONT", _default_font)

# Output video
OUT_W, OUT_H = 1080, 1920
VIDEO_CRF = int(os.getenv("VIDEO_CRF", "18"))  # lower = higher quality, bigger file
VIDEO_PRESET = os.getenv("VIDEO_PRESET", "medium")

WHISPER_MODELS = {
    "fast": "base",
    "balanced": "small",
    "best": "large-v3-turbo",
}
