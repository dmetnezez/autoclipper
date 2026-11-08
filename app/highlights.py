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


# ------------------------------------------------------- offline picker
# Works without any API key. Scores every possible clip on hook strength,
# energy, pace, emotional language and clean start/end, then picks the best.

STOPWORDS = set("""a an the and or but so because if then than that this these those it its it's is are was were be been
being am do does did doing have has had having i me my mine we us our you your yours he him his she her they them their
what which who whom whose when where why how all any both each few more most other some such no nor not only own same
too very can will just should now of at by for with about against between into through during before after above below
to from up down in out on off over under again further once here there also yeah like um uh okay ok really know mean
think thing things going gonna got get gets would could said says say one two well right kind sort lot lots actually
basically literally maybe something someone anything everything people""".split())

STRONG_WORDS = {
    "never", "always", "secret", "secrets", "biggest", "mistake", "mistakes", "crazy", "insane", "truth", "worst",
    "best", "nobody", "everyone", "everybody", "stop", "wrong", "dead", "died", "die", "love", "hate", "money",
    "million", "millions", "billion", "rich", "broke", "fired", "quit", "failed", "fail", "scared", "afraid",
    "shocking", "unbelievable", "incredible", "honestly", "seriously", "literally", "impossible", "dangerous",
    "illegal", "lie", "lied", "lying", "fake", "real", "problem", "important", "hardest", "easiest", "only",
    "first", "last", "huge", "massive", "terrible", "amazing", "perfect", "favorite", "hated", "loved", "regret",
    "warning", "trick", "hack", "rule", "rules", "lesson", "advice", "changed", "change", "life", "everything",
}
QUESTION_STARTS = {"why", "how", "what", "who", "when", "where", "would", "have", "did", "do", "is", "are", "can"}
DEPENDENT_STARTS = {"and", "but", "so", "because", "or", "then", "also", "which", "yeah", "like", "um", "uh",
                    "well", "right", "okay", "ok", "anyway", "plus", "cause", "'cause"}
REFERENCE_STARTS = {"that", "it", "this", "they", "he", "she", "those", "these", "there"}
FILLER_LEAD = {"so", "and", "but", "um", "uh", "like", "yeah", "well", "okay", "ok", "right", "anyway", "i mean"}


def _tok(word):
    return re.sub(r"[^a-z0-9$%']", "", word.lower())


def _sentences(words, starts, ends):
    end_set = set(ends)
    out, cur_start = [], None
    start_set = set(starts)
    for i in range(len(words)):
        if i in start_set or cur_start is None:
            cur_start = i if cur_start is None else cur_start
        if i in end_set:
            seg = words[cur_start: i + 1]
            text = " ".join(w["word"] for w in seg)
            out.append({"si": cur_start, "ei": i, "start": seg[0]["start"], "end": seg[-1]["end"],
                        "text": text, "tokens": [_tok(w["word"]) for w in seg]})
            cur_start = None
    return out


def _hookiness(sent):
    toks = [t for t in sent["tokens"] if t]
    if not toks:
        return 0.0, []
    score, why = 0.0, []
    if sent["text"].rstrip().endswith("?") or toks[0] in QUESTION_STARTS:
        score += 0.35
        why.append("opens with a question")
    strong = sum(t in STRONG_WORDS for t in toks)
    if strong:
        score += min(strong, 3) * 0.15
        why.append("bold opening line")
    if "you" in toks or "your" in toks:
        score += 0.12
    if any(re.search(r"\d|\$|%", t) for t in toks):
        score += 0.15
        why.append("specific numbers")
    if any(t.endswith("est") and len(t) > 5 for t in toks):
        score += 0.08
    if 4 <= len(toks) <= 16:
        score += 0.1  # short punchy first line
    if sent["text"].rstrip().endswith("!"):
        score += 0.1
    return min(score, 1.0), why


def _clean_lead(text):
    words = text.split()
    while words and _tok(words[0]) in FILLER_LEAD:
        words = words[1:]
    text = " ".join(words)
    return text[:1].upper() + text[1:] if text else text


def _shorten(text, max_chars):
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "..."


def _heuristic(words, starts, ends, n, min_len, max_len, wav_path):
    loud = _loudness(wav_path)
    sents = _sentences(words, starts, ends)
    if not sents:
        return []
    hook = [_hookiness(s) for s in sents]
    target = (min_len + max_len) / 2

    cands = []
    for i, first in enumerate(sents):
        # Try every ending sentence that keeps the clip within the length limits
        best = None
        for j in range(i, len(sents)):
            length = sents[j]["end"] - first["start"]
            if length > max_len:
                break
            if length < min_len:
                continue
            group = sents[i: j + 1]
            toks = [t for s in group for t in s["tokens"] if t]
            nwords = len(toks)
            score, why = 0.0, list(hook[i][1])

            score += hook[i][0] * 30 + (hook[i + 1][0] * 8 if i + 1 <= j else 0)

            wps = nwords / max(length, 1)
            pace = max(0.0, 1 - abs(wps - 3.0) / 1.8)
            score += pace * 14
            if wps > 3.0:
                why.append("fast-paced")

            if loud is not None:
                a, b = int(first["start"]), max(int(first["start"]) + 1, int(sents[j]["end"]))
                chunk = loud[a:b]
                if len(chunk):
                    energy = float(chunk.mean())
                    score += energy * 14 + min(float(chunk.std()) * 2, 1) * 6
                    if energy > 0.55:
                        why.append("high energy")

            strong = sum(t in STRONG_WORDS for t in toks) / max(nwords, 1)
            score += min(strong * 12, 1) * 10

            last = group[-1]["text"].rstrip()
            if last.endswith(("!", "?")) or hook[j][0] > 0.3:
                score += 6
                why.append("ends on a strong line")

            gaps = sum(1 for k in range(first["si"] + 1, sents[j]["ei"] + 1)
                       if words[k]["start"] - words[k - 1]["end"] > 1.5)
            score -= gaps * 4

            first_tok = next((t for t in first["tokens"] if t), "")
            if first_tok in DEPENDENT_STARTS:
                score -= 16
            elif first_tok in REFERENCE_STARTS:
                score -= 7

            score += max(0.0, 1 - abs(length - target) / max(target, 1)) * 5
            if best is None or score > best["score"]:
                best = {"start": first["start"], "end": sents[j]["end"], "score": score,
                        "i": i, "j": j, "why": why}
        if best:
            cands.append(best)

    cands.sort(key=lambda c: c["score"], reverse=True)
    picked = []
    for c in cands:
        if all(c["end"] + 3 <= p["start"] or c["start"] >= p["end"] + 3 for p in picked):
            picked.append(c)
        if len(picked) >= n:
            break

    out = []
    for c in picked:
        group = sents[c["i"]: c["j"] + 1]
        # Title: the most hook-like of the first three sentences
        lead = max(range(c["i"], min(c["j"], c["i"] + 2) + 1), key=lambda k: hook[k][0])
        title_src = _clean_lead(sents[lead]["text"])
        title = _shorten(title_src, 72)
        hook_words = title_src.rstrip(".!,").split()
        hook_text = " ".join(hook_words) if len(hook_words) <= 8 else " ".join(hook_words[:6]) + "..."

        counts = {}
        for s in group:
            for t in s["tokens"]:
                if len(t) >= 5 and t not in STOPWORDS and not t.isdigit() and "'" not in t:
                    counts[t] = counts.get(t, 0) + 1
        tags = [f"#{t}" for t, k in sorted(counts.items(), key=lambda x: -x[1]) if k >= 2][:3]
        description = f"{title_src.rstrip('.')} {' '.join(tags + ['#shorts'])}".strip()

        why = list(dict.fromkeys(c["why"]))[:3] or ["clean, self-contained moment"]
        reason = (", ".join(why)).capitalize() + "."
        out.append({
            "start": c["start"], "end": c["end"], "title": title, "hook": hook_text,
            "score": int(max(20, min(97, 25 + c["score"]))), "reason": reason,
            "description": description,
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
