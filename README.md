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
