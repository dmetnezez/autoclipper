"""Pick the best short-form moments from a transcript.

Uses Claude when an API key is set (much better picks); otherwise falls back to
a simple scoring heuristic so the app still works offline.
"""
import json
import re
import wave
from pathlib import Path

import numpy as np

from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL

END_PUNCT = re.compile(r"[.!?…]['\")\]]*$")

SYSTEM_PROMPT = """You are an elite short-form video editor who has grown multiple TikTok, YouTube Shorts and Instagram Reels accounts to millions of followers by clipping long-form videos.

You are given a timestamped transcript of a long video. Your job is to find the moments that will perform best as standalone vertical short-form clips.

What makes a great clip:
- A strong hook in the first 1-3 seconds: a bold claim, a surprising fact, a question, conflict, emotion, or a line that creates curiosity. The first sentence must make a scroller stop.
- It is self-contained: a viewer with zero context understands it. It does not start mid-thought or depend on something said earlier ("like I said", "that thing", "so yeah").
- It has a payoff: a punchline, insight, story resolution, reveal, or a satisfying conclusion. It ends on a complete sentence, ideally on the strongest line.
- Dense: no long tangents, setup or filler. Every sentence earns its place.
- Emotional or useful: funny, controversial, inspiring, surprising, relatable, or genuinely valuable advice.

Rules:
- Use the transcript's timestamps. Start each clip exactly at the start time of a line, and end it exactly at the end time of a line.
- Clips must not overlap.
- Score honestly from 0 to 100 for viral potential. Do not give everything a high score.
- "title": a short, curiosity-driven title for the clip (max 60 characters).
- "hook": on-screen hook text shown at the top for the first seconds of the clip. 2 to 7 words, punchy, no hashtags, no emojis. It must tease the clip without spoiling the payoff.
- "description": a ready-to-post caption, 1-2 sentences plus 3-5 relevant hashtags.
- "reason": one sentence on why this moment works."""


def _clip_schema():
    return {
        "type": "object",
        "properties": {
            "clips": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "title": {"type": "string"},
                        "hook": {"type": "string"},
                        "score": {"type": "integer"},
                        "reason": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["start", "end", "title", "hook", "score", "reason", "description"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["clips"],
        "additionalProperties": False,
    }


def _ask_claude(segments, video_title, n, min_len, max_len):
    import anthropic

    lines = [f"[{s['start']:.2f} - {s['end']:.2f}] {s['text']}" for s in segments]
    user = (
        f"Video title: {video_title}\n\n"
        f"Find the {n} best clips. Each clip must be between {min_len} and {max_len} seconds long. "
        f"If the video doesn't have {n} genuinely good moments, return fewer.\n\n"
        "Transcript:\n" + "\n".join(lines)
    )
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    with client.beta.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=32000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
        output_config={
            "effort": "medium",
            "format": {"type": "json_schema", "schema": _clip_schema()},
        },
    ) as stream:
        msg = stream.get_final_message()

    if msg.stop_reason == "refusal":
        raise RuntimeError("Claude declined to analyze this transcript.")
    text = next((b.text for b in msg.content if b.type == "text"), "")
    return json.loads(text)["clips"]


# ---------------------------------------------------------------- snapping

def _flatten(segments):
    words = []
    for s in segments:
        for i, w in enumerate(s["words"]):
            words.append({**w, "seg_start": i == 0, "seg_end": i == len(s["words"]) - 1})
    return words


def _boundaries(words):
    """Word indices where a sentence starts / ends."""
    starts, ends = set(), set()
    for i, w in enumerate(words):
        prev = words[i - 1] if i else None
        nxt = words[i + 1] if i + 1 < len(words) else None
        if prev is None or END_PUNCT.search(prev["word"]) or w["start"] - prev["end"] > 0.7:
            starts.add(i)
        if nxt is None or END_PUNCT.search(w["word"]) or nxt["start"] - w["end"] > 0.7:
            ends.add(i)
    return sorted(starts), sorted(ends)


def snap_clip(start, end, words, starts, ends, min_len, max_len, duration):
    """Move start/end to clean sentence boundaries and enforce length limits."""
    def nearest(cands, t, key, window):
        best = None
        for i in cands:
            d = abs(words[i][key] - t)
            if d <= window and (best is None or d < best[0]):
                best = (d, i)
        return best[1] if best else None

    si = nearest(starts, start, "start", 4.0)
    if si is None:
        si = min(range(len(words)), key=lambda i: abs(words[i]["start"] - start))
    s = words[si]["start"]

    valid_ends = [i for i in ends if min_len <= words[i]["end"] - s <= max_len]
    ei = nearest(valid_ends, end, "end", 8.0)
    if ei is None:
        # No sentence end fits the length: take the last word that fits
        fits = [i for i in range(si, len(words)) if words[i]["end"] - s <= max_len]
        ei = fits[-1] if fits else si
    e = words[ei]["end"]

    # Small padding so words aren't clipped
    s = max(0.0, s - 0.12)
    e = min(duration, e + 0.45)
    return s, e


# --------------------------------------------------------------- heuristic

HOOK_WORDS = {"you", "your", "why", "how", "never", "secret", "biggest", "mistake", "crazy",
              "actually", "truth", "worst", "best", "nobody", "everyone", "stop", "insane",
              "money", "million", "dead", "love", "hate", "wrong", "real", "honestly", "literally"}


def _loudness(wav_path: Path):
    """Per-second RMS loudness, normalized 0..1."""
    try:
        with wave.open(str(wav_path)) as w:
            sr = w.getframerate()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        secs = len(data) // sr
        if secs == 0:
            return None
        rms = np.sqrt((data[: secs * sr].reshape(secs, sr) ** 2).mean(axis=1))
        lo, hi = np.percentile(rms, 5), np.percentile(rms, 98)
        return np.clip((rms - lo) / max(hi - lo, 1e-6), 0, 1)
    except Exception:
        return None


def _heuristic(words, starts, ends, n, min_len, max_len, wav_path):
    loud = _loudness(wav_path)
    target = (min_len + max_len) / 2
    cands = []
    end_set = set(ends)
    for si in starts:
        s = words[si]["start"]
        best_ei = None
        for ei in range(si, len(words)):
            length = words[ei]["end"] - s
            if length > max_len:
                break
            if ei in end_set and length >= min_len:
                best_ei = ei
                if length >= target:
                    break
        if best_ei is None:
            continue
        seg = words[si: best_ei + 1]
        e = words[best_ei]["end"]
        text = " ".join(w["word"] for w in seg)
        tokens = [re.sub(r"[^a-z']", "", w["word"].lower()) for w in seg]
        wps = len(seg) / max(e - s, 1)
        first = tokens[:12]
        score = min(wps / 3.2, 1.0) * 35
        score += min(sum(t in HOOK_WORDS for t in first), 3) * 8
        score += min(text.count("?") + text.count("!"), 3) * 4
        if loud is not None:
            a, b = int(s), max(int(s) + 1, int(e))
            score += float(loud[a:b].mean()) * 25 if a < len(loud) else 0
        if first and first[0] in {"and", "but", "so", "because", "or", "like", "yeah", "um", "uh"}:
            score -= 12
        cands.append({"start": s, "end": e, "score": score, "text": text})

    cands.sort(key=lambda c: c["score"], reverse=True)
    picked = []
    for c in cands:
        if all(c["end"] <= p["start"] or c["start"] >= p["end"] for p in picked):
            picked.append(c)
        if len(picked) >= n:
            break
    top = max((c["score"] for c in picked), default=1) or 1
    out = []
    for c in picked:
        first_sentence = re.split(r"(?<=[.!?])\s", c["text"])[0]
        title = first_sentence[:57] + ("..." if len(first_sentence) > 57 else "")
        hook_words = first_sentence.rstrip(".!?,").split()[:6]
        out.append({
            "start": c["start"], "end": c["end"], "title": title,
            "hook": " ".join(hook_words),
            "score": int(40 + 50 * c["score"] / top),
            "reason": "Picked by the offline scorer (fast speech, energy and hook words). Add a Claude API key for smarter picks.",
            "description": title,
        })
    return out


def find_highlights(segments, video_title, n, min_len, max_len, duration, wav_path):
    """Returns (clips, used_ai). Clips have clean boundaries and no overlaps."""
    words = _flatten(segments)
    if len(words) < 10:
        raise RuntimeError("Not enough speech was found in this video to make clips.")
    starts, ends = _boundaries(words)

    used_ai = False
    raw = None
    ai_error = None
    if ANTHROPIC_API_KEY:
        try:
            raw = _ask_claude(segments, video_title, n, min_len, max_len)
            used_ai = True
        except Exception as e:  # network/key problems: fall back rather than fail
            ai_error = str(e)
    if raw is None:
        raw = _heuristic(words, starts, ends, n, min_len, max_len, wav_path)

    clips = []
    for c in sorted(raw, key=lambda c: c.get("score", 0), reverse=True):
        s, e = snap_clip(float(c["start"]), float(c["end"]), words, starts, ends,
                         min_len, max_len, duration)
        if e - s < 5:
            continue
        if any(not (e <= o["start"] + 0.5 or s >= o["end"] - 0.5) for o in clips):
            continue
        clips.append({**c, "start": s, "end": e,
                      "score": max(0, min(100, int(c.get("score", 50))))})
        if len(clips) >= n:
            break
    return clips, used_ai, ai_error
