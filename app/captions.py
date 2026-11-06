"""Animated word-by-word captions and hook titles as an ASS subtitle file."""
import re

from .config import CAPTION_FONT, OUT_H, OUT_W

FILLERS = {"um", "uh", "umm", "uhh", "erm", "hmm", "mm"}

# ASS colors are &HAABBGGRR
STYLES = {
    "pop":   {"active": "&H0000E5FF", "upper": True,  "size": 82, "box": False},  # yellow highlight
    "green": {"active": "&H0045F04A", "upper": True,  "size": 82, "box": False},
    "clean": {"active": None,         "upper": False, "size": 70, "box": True},
}


def _ts(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _esc(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def _chunks(words, max_words=3, max_chars=16):
    """Group words into short on-screen phrases."""
    chunks, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        text_len = sum(len(x["word"]) + 1 for x in cur)
        nxt = words[i + 1] if i + 1 < len(words) else None
        brk = (
            len(cur) >= max_words or text_len >= max_chars
            or re.search(r"[.!?,;:]$", w["word"])
            or (nxt is not None and nxt["start"] - w["end"] > 0.5)
        )
        if brk:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def build_ass(words, clip_start, clip_end, hook, style_name, layout, src_aspect) -> str:
    style = STYLES.get(style_name)
    dur = clip_end - clip_start

    if layout == "crop":
        cap_y, hook_y = int(OUT_H * 0.70), int(OUT_H * 0.17)
    else:  # video sits in the middle band; put text above and below it
        box_h = OUT_W / max(src_aspect, 0.1)
        cap_y = int(min(OUT_H * 0.80, OUT_H / 2 + box_h / 2 + 150))
        hook_y = int(max(OUT_H * 0.14, OUT_H / 2 - box_h / 2 - 140))

    size = style["size"] if style else 80
    outline = 14 if (style and style["box"]) else 7
    border_style = 3 if (style and style["box"]) else 1
    # With BorderStyle 3 the box is drawn in the outline colour
    outline_col = "&H55000000" if (style and style["box"]) else "&H00000000"

    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{CAPTION_FONT},{size},&H00FFFFFF,&H00FFFFFF,{outline_col},&H80000000,-1,0,0,0,100,100,1,0,{border_style},{outline},2,5,60,60,0,1
Style: Hook,{CAPTION_FONT},66,&H00000000,&H00FFFFFF,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,18,0,5,90,90,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []

    if hook and hook.strip():
        hook_text = _esc(hook.strip())
        hook_end = min(dur, 3.2)
        lines.append(f"Dialogue: 1,{_ts(0)},{_ts(hook_end)},Hook,,0,0,0,,"
                     f"{{\\pos({OUT_W // 2},{hook_y})\\fad(120,250)}}{hook_text}")

    if style:
        local = []
        for w in words:
            if w["end"] <= clip_start or w["start"] >= clip_end:
                continue
            bare = re.sub(r"[^\w']", "", w["word"].lower())
            if bare in FILLERS:
                continue
            text = w["word"].upper() if style["upper"] else w["word"]
            if style["upper"]:
                text = text.rstrip(",.")
            local.append({"start": w["start"] - clip_start, "end": w["end"] - clip_start,
                          "word": _esc(text)})

        chunks = _chunks(local, 3 if style["upper"] else 5, 15 if style["upper"] else 28)
        for ci, ch in enumerate(chunks):
            # Keep each phrase on screen until the next one starts (no flicker)
            nxt_start = chunks[ci + 1][0]["start"] if ci + 1 < len(chunks) else ch[-1]["end"] + 0.4
            chunk_end = min(nxt_start, ch[-1]["end"] + 0.6, dur)
            pos = f"\\pos({OUT_W // 2},{cap_y})"
            if style["active"] is None:
                text = " ".join(w["word"] for w in ch)
                lines.append(f"Dialogue: 0,{_ts(ch[0]['start'])},{_ts(chunk_end)},Cap,,0,0,0,,{{{pos}}}{text}")
                continue
            for wi, w in enumerate(ch):
                w_start = w["start"] if wi else ch[0]["start"]
                w_end = ch[wi + 1]["start"] if wi + 1 < len(ch) else chunk_end
                if w_end <= w_start:
                    continue
                parts = []
                for wj, other in enumerate(ch):
                    if wj == wi:
                        parts.append(f"{{\\c{style['active']}\\fscx112\\fscy112}}{other['word']}{{\\c&H00FFFFFF&\\fscx100\\fscy100}}")
                    else:
                        parts.append(other["word"])
                intro = "\\fscx80\\fscy80\\t(0,90,\\fscx100\\fscy100)" if wi == 0 else ""
                lines.append(f"Dialogue: 0,{_ts(w_start)},{_ts(w_end)},Cap,,0,0,0,,"
                             f"{{{pos}{intro}}}" + " ".join(parts))

    return head + "\n".join(lines) + "\n"
