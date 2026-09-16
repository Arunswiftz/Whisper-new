# Local Whisper

Browser-only speech-to-text using Whisper and Transformers.js.

## Features

- Local browser inference; audio is not uploaded to a transcription server.
- English variants: US, Australia, and UK.
- Whisper Base with word-level timestamps.
- SRT export with timestamps and speaker labels.
- TXT export with speaker-labelled paragraphs.
- Automatic speaker segmentation using `onnx-community/pyannote-segmentation-3.0`.
- Speaker labels are shown as `Speaker 1`, `Speaker 2`, etc.
- Speech capture sensitivity controls for quieter speech vs. background-noise rejection.
- WebGPU is used for Whisper when available; WASM fallback is used otherwise.
- Audio is normalised to mono 16 kHz before inference.

## Speaker detection

The speaker detector identifies changes between voice tracks and associates Whisper word timestamps with the detected speaker regions. It does not know a person's real name and should be treated as automatic speaker diarization rather than biometric identity recognition.

The current browser implementation follows the same Transformers.js + Whisper timestamp + PyAnnote segmentation architecture demonstrated by the Xenova browser diarization example.

## SRT format

Generated subtitles use the following structure:

```text
1
00:00:01,240 --> 00:00:04,860
Speaker 1
Hello, how are you?

2
00:00:05,020 --> 00:00:07,910
Speaker 2
I'm doing well, thank you.
```

## Speech capture sensitivity

- **High**: lower no-speech threshold; more likely to retain quiet speech, but may capture more background noise.
- **Balanced**: recommended default.
- **Strict**: more conservative about weak/uncertain speech and background noise.

These thresholds affect Whisper's long-form decoding behaviour; they cannot recover speech that is completely absent from the audio.

## Models

- ASR: `onnx-community/whisper-base_timestamped`
- Speaker segmentation: `onnx-community/pyannote-segmentation-3.0`

Whisper's word timestamp support is provided through `return_timestamps: "word"`. Transformers.js exposes the returned word chunks and timestamps for downstream subtitle generation.

## Important limitations

- US/Australian/UK English are transcription-style preferences, not separate accent-specific Whisper acoustic models.
- Speaker diarization can make mistakes with overlapping speech, strong background noise, very short turns, or voices that sound similar.
- The speaker labels are anonymous (`Speaker 1`, `Speaker 2`, etc.).
- Larger audio files require more browser memory and processing time.

## Run

This project is a static browser application. Serve the repository with any static web server or GitHub Pages and open `index.html` through the server.


## Online audio/video links

The web UI now has a **Paste an online audio/video link** option.

There are two paths:

1. **Direct media URL** — a URL that points directly to an MP3, WAV, MP4, WebM, etc. can be fetched by the browser when the host permits CORS.
2. **Media-page URL** — YouTube and many other supported sites need the included Node.js media bridge. The bridge uses `yt-dlp` + FFmpeg to resolve the page URL, extract/convert the audio to WAV, and return that file to the browser. Whisper then processes the returned file exactly like an uploaded file.

### Run locally with online-link support

Requirements:
- Node.js 20+
- FFmpeg
- Python 3 + yt-dlp

Then:

    npm install
    yt-dlp --version
    ffmpeg -version
    npm start

Open `http://localhost:3000`.

The included `Dockerfile` installs Node, FFmpeg and yt-dlp automatically for a container deployment.

### Important architecture note

The Whisper inference remains **in the browser**. The media bridge is only used to retrieve/convert online media. A static GitHub Pages deployment cannot itself run `server.js`, so YouTube/media-page URLs will require the Node service to be deployed separately (or the whole application deployed with the included Dockerfile).

The phrase "any link" means any public HTTP/HTTPS URL that either:
- is a browser-accessible direct media file, or
- is supported by the installed version of yt-dlp.

Private/authenticated media, DRM-protected streams, links requiring a login, and sites that block automated retrieval may not work.

### Security

The media endpoint accepts only HTTP/HTTPS URLs and rejects obvious localhost/private metadata targets. If this service is exposed publicly, add authentication/rate limiting and stronger outbound-network/SSRF controls before production use.
