"""Smart vertical reframing: follow the speaker's face, cut with the camera.

Samples frames from the clip, finds faces and camera cuts, then builds a crop
path that holds steady, eases to follow people when they move, and jumps on
camera cuts. Falls back to a blurred-background layout when there are no faces
(gameplay, slides, b-roll).
"""
import statistics
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .config import MODELS_DIR

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SAMPLES_PER_SEC = 4
DETECT_W = 480


class FaceFinder:
    def __init__(self):
        self.yunet = None
        self.haar = None
        if not YUNET_PATH.exists():
            try:
                urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
            except Exception:
                YUNET_PATH.unlink(missing_ok=True)
        if YUNET_PATH.exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.yunet = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.75, 0.3, 50)
            except Exception:
                self.yunet = None
        if self.yunet is None and hasattr(cv2, "CascadeClassifier"):
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def find(self, small):
        """Return faces as (cx, cy, w, h) in the small frame's pixels."""
        h, w = small.shape[:2]
        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(small)
            if faces is None:
                return []
            return [(f[0] + f[2] / 2, f[1] + f[3] / 2, f[2], f[3]) for f in faces]
        if self.haar is not None:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            found = self.haar.detectMultiScale(gray, 1.1, 6, minSize=(int(w * 0.04),) * 2)
            return [(x + fw / 2, y + fh / 2, fw, fh) for x, y, fw, fh in found]
        return []


_finder = None


def _get_finder():
    global _finder
    if _finder is None:
        _finder = FaceFinder()
    return _finder


def analyze(video: Path, start: float, end: float, fps: float):
    """Sample the clip; return list of {t, faces, cut} with t relative to clip start."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    step = max(1, round(fps / SAMPLES_PER_SEC))
    finder = _get_finder()
    samples, prev_hist, i = [], None, 0
    total = int((end - start) * fps)
    while i <= total:
        if i % step == 0:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - start
            h, w = frame.shape[:2]
            scale = DETECT_W / w
            small = cv2.resize(frame, (DETECT_W, int(h * scale)))
            faces = [(cx / DETECT_W, cy / small.shape[0], fw / DETECT_W, fh / small.shape[0])
                     for cx, cy, fw, fh in finder.find(small)]
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            cut = prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > 0.45
            prev_hist = hist
            samples.append({"t": max(0.0, t), "faces": faces, "cut": cut})
        else:
            if not cap.grab():
                break
        i += 1
    cap.release()
    return samples


def choose_layout(samples, src_w, src_h, preference="auto"):
    if src_h >= src_w * 0.9:
        return "vertical"            # already vertical/square: fit with blur pad
    if preference in ("crop", "fit"):
        return preference
    if not samples:
        return "fit"
    with_faces = sum(1 for s in samples if s["faces"]) / len(samples)
    return "crop" if with_faces >= 0.4 else "fit"


def crop_path(samples, src_w, src_h, duration):
    """Build a list of (t0, t1, x0, x1) segments giving the crop's left edge in pixels."""
    cw = src_h * 9 / 16
    max_x = max(0.0, src_w - cw)
    cwf = cw / src_w  # crop width as fraction of source width

    def to_px(cx):
        return float(np.clip(cx * src_w - cw / 2, 0, max_x))

    # Split samples into shots at camera cuts
    shots, cur = [], []
    for s in samples:
        if s["cut"] and cur:
            shots.append(cur)
            cur = []
        cur.append(s)
    if cur:
        shots.append(cur)

    keys = []  # (t, x, mode) where mode is "jump" or "ease"
    last_x = to_px(0.5)
    for shot in shots:
        track = []
        for s in shot:
            if s["faces"]:
                main = max(s["faces"], key=lambda f: f[2] * f[3])  # biggest face
                track.append((s["t"], main[0]))
        if not track:
            keys.append((shot[0]["t"], last_x, "jump"))
            continue
        first = statistics.median(c for t, c in track[:4])
        held = first
        keys.append((shot[0]["t"], to_px(held), "jump"))
        window = []
        for t, c in track:
            window.append(c)
            window = window[-5:]
            med = statistics.median(window)
            # Only move when the face drifts out of the middle of the frame
            if abs(med - held) > 0.22 * cwf:
                held = med
                keys.append((t, to_px(held), "ease"))
        last_x = to_px(held)

    if not keys:
        keys = [(0.0, to_px(0.5), "jump")]
    keys[0] = (0.0, keys[0][1], "jump")

    segs = []
    ease = 0.6
    for i, (t, x, mode) in enumerate(keys):
        t_next = keys[i + 1][0] if i + 1 < len(keys) else duration + 1
        if mode == "ease" and segs:
            prev_x = segs[-1][3]
            t_end = min(t + ease, t_next)
            segs.append((t, t_end, prev_x, x))
            if t_end < t_next:
                segs.append((t_end, t_next, x, x))
        else:
            segs.append((t, t_next, x, x))
    # Merge adjacent constant segments
    merged = []
    for s in segs:
        if merged and merged[-1][2] == merged[-1][3] == s[2] == s[3]:
            merged[-1] = (merged[-1][0], s[1], s[2], s[3])
        else:
            merged.append(s)
    return merged


def crop_expr(segs) -> str:
    """Turn crop path segments into an ffmpeg expression of t."""
    def piece(t0, t1, x0, x1):
        if abs(x1 - x0) < 0.5:
            return f"{x0:.1f}"
        p = f"((t-{t0:.3f})/{max(t1 - t0, 0.001):.3f})"
        return f"({x0:.1f}+({x1 - x0:.1f})*(3*{p}*{p}-2*{p}*{p}*{p}))"

    expr = piece(*segs[-1])
    for t0, t1, x0, x1 in reversed(segs[:-1]):
        expr = f"if(lt(t,{t1:.3f}),{piece(t0, t1, x0, x1)},{expr})"
    return expr
