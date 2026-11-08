# AutoClipper

Turn long videos into ready-to-post vertical shorts for TikTok, Reels and YouTube Shorts. AutoClipper runs on your own computer.

Paste a YouTube link or upload a video. AutoClipper then:

1. **Transcribes** the speech with word-level timing. It uses Whisper, which runs locally for free.
2. **Finds the best moments** by scoring every possible clip for a strong hook, energy, pace and a clean start and end. Optionally, Claude can pick instead. Each clip gets a viral score, a title and a ready-to-post caption with hashtags.
3. **Reframes to 9:16** by tracking faces. The crop holds steady, eases to follow people who move, and cuts when the camera cuts. Gameplay, slides and shots with no faces get a full-frame layout with a blurred background.
4. **Adds animated captions.** Words appear in bold, a few at a time, and the spoken word is highlighted. Filler words like "um" and "uh" are removed.
5. **Adds a hook title** at the top for the first 3 seconds.
6. **Exports** 1080×1920, 30fps H.264 video with loudness-normalized audio.

## Setup (Windows)

You only do this once.

1. **Install Python 3.12** from https://www.python.org/downloads/. During install, tick **"Add python.exe to PATH"**.
2. **Install FFmpeg** by opening PowerShell and running:
   ```
   winget install Gyan.FFmpeg
   ```
3. **(Recommended)** Install Deno so YouTube links download reliably:
   ```
   winget install DenoLand.Deno
   ```
4. **Download this project.** Use the green **Code → Download ZIP** button on GitHub and unzip it, or run `git clone`.
5. **Double-click `start.bat`.** The first launch installs everything, which takes a few minutes. Your browser then opens to http://localhost:8000.

### No API key needed

AutoClipper works out of the box with its built-in clip picker, which runs on your PC for free. It scores every possible clip on hook strength, energy, pace and emotional language, and on whether it starts and ends cleanly.

*Optional:* if you ever want Claude to pick the clips instead, put `ANTHROPIC_API_KEY=sk-ant-...` in the `.env` file and restart. A key from https://console.anthropic.com costs a few cents per video.

## Using it

- **Paste link** or **Upload file**, then click **Make clips**.
- **Clips**: how many clips you want.
- **Clip length**: 20–60 seconds suits most platforms.
- **Captions**: bold with a yellow or green highlighted word, clean subtitles, or no captions.
- **Framing**: *Auto* follows faces when there are any and otherwise uses the blurred layout. You can force either one.
- **Transcription**: *Balanced* is fine for most videos. *Best* is more accurate but slow without an NVIDIA GPU.

Each clip shows its score, why it was chosen, and a caption you can copy. Click **Download** to save the MP4.

The first video takes longer because the speech model downloads once (about 250 MB for Balanced).

## Speed

| Setup | One hour of video |
|---|---|
| NVIDIA GPU | ~3–6 min |
| Modern CPU (Balanced) | ~10–20 min |

AutoClipper uses an NVIDIA GPU automatically if CUDA is available, and falls back to the CPU otherwise.

## Customizing

- **Caption font**: put a `.ttf` file in `fonts/` and set `CAPTION_FONT=Font Name` in `.env`. Montserrat Black and Anton are popular choices and free on Google Fonts. The default is Arial Black.
- **Video quality**: set `VIDEO_CRF` in `.env`. The default is 18. Lower means sharper video and bigger files.
- **Highlight prompt**: edit `SYSTEM_PROMPT` in `app/highlights.py` to change what counts as a good clip.

## A note on YouTube links

Downloading videos is against YouTube's Terms of Service unless the video is yours or you have permission. Use AutoClipper on your own content or on content you're allowed to clip. Many creators pay clippers to do exactly this. If a link fails, restart `start.bat` (it updates the downloader) or upload the file instead.

## Troubleshooting

- **"FFmpeg isn't installed"**: run `winget install Gyan.FFmpeg`, then close and reopen `start.bat`.
- **YouTube download fails**: install Deno, restart `start.bat`, or upload the file instead.
- **A job failed halfway**: the error is shown on the project. Delete the project and try again.
- **Start over completely**: delete the `.venv` folder and the `.installed` file, then run `start.bat` again.

## How it works

```
app/
  main.py         web server and API (FastAPI)
  pipeline.py     runs each job: download → transcribe → pick → reframe → render
  download.py     YouTube and other links via yt-dlp
  transcribe.py   faster-whisper, with word timestamps
  highlights.py   picks the clips (built-in scorer, or Claude if a key is set) and snaps cuts to sentence boundaries
  reframe.py      face tracking, camera-cut detection, smooth crop path
  captions.py     animated word-by-word captions and hook title (ASS subtitles)
  render.py       FFmpeg render to 1080×1920
  static/         the dashboard
```

All projects and clips are stored in `data/` on your computer.
