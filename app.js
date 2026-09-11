import {
    pipeline,
    AutoProcessor,
    AutoModelForAudioFrameClassification
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const ASR_MODEL = "onnx-community/whisper-base_timestamped";
const SEGMENTATION_MODEL = "onnx-community/pyannote-segmentation-3.0";
const MAX_BLOCK_SECONDS = 7;
const SPEAKER_MERGE_GAP = 1.25;
const SPEAKER_CONFIDENCE = 0.20;
const MAX_AUDIO_DURATION_SECONDS = 12 * 60;
const TRANSCRIPTION_CHUNK_SECONDS = 29; // ~30s; 29 avoids timestamp bug in whisper-base_timestamped

const SENSITIVITY = {
    high: { noSpeechThreshold: 0.15, label: "High" },
    balanced: { noSpeechThreshold: 0.25, label: "Balanced" },
    strict: { noSpeechThreshold: 0.40, label: "Strict" }
};

const ENGLISH_VARIANTS = {
    "en-US": {
        label: "English US",
        prompt: "Transcribe in American English. Use US English spelling, vocabulary, punctuation, and conventions."
    },
    "en-AU": {
        label: "English Australia",
        prompt: "Transcribe in Australian English. Use Australian English spelling, vocabulary, punctuation, and conventions."
    },
    "en-GB": {
        label: "English UK",
        prompt: "Transcribe in British English. Use UK English spelling, vocabulary, punctuation, and conventions."
    }
};

const $ = id => document.getElementById(id);
const audioFile = $("audioFile");
const fileInfo = $("fileInfo");
const fileName = $("fileName");
const fileSize = $("fileSize");
const removeFile = $("removeFile");
const dropZone = $("dropZone");
const transcribeButton = $("transcribeButton");
const progressArea = $("progressArea");
const progressBar = $("progressBar");
const status = $("status");
const resultSection = $("resultSection");
const resultStatus = $("resultStatus");
const transcription = $("transcription");
const speakerTranscript = $("speakerTranscript");
const downloadTxt = $("downloadTxt");
const downloadSrt = $("downloadSrt");
const copyTranscript = $("copyTranscript");
const languageSelect = $("language");
const sensitivitySelect = $("sensitivity");
const mediaPreview = $("mediaPreview");
const audioPreview = $("audioPreview");
const videoPreview = $("videoPreview");

let selectedFile = null;
let mediaObjectUrl = null;
let transcriber = null;
let segmentationProcessor = null;
let segmentationModel = null;
let modelsLoaded = false;
let transcriptionData = [];

function setProgress(value, message) {
    progressBar.style.width = `${value}%`;
    if (message) status.textContent = message;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatTime(seconds) {
    const safe = Math.max(0, Number(seconds) || 0);
    const h = Math.floor(safe / 3600);
    const m = Math.floor((safe % 3600) / 60);
    const s = Math.floor(safe % 60);
    return h > 0
        ? `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
        : `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function formatSrtTime(seconds) {
    const safe = Math.max(0, Number(seconds) || 0);
    let totalMs = Math.round(safe * 1000);
    const h = Math.floor(totalMs / 3600000);
    totalMs %= 3600000;
    const m = Math.floor(totalMs / 60000);
    totalMs %= 60000;
    const s = Math.floor(totalMs / 1000);
    const ms = totalMs % 1000;
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")},${String(ms).padStart(3, "0")}`;
}

function cleanText(text) {
    return text
        .replace(/\s+([,.!?;:])/g, "$1")
        .replace(/\s+(['’])/g, "$1")
        .replace(/\s+/g, " ")
        .trim();
}

function selectFile(file) {
    selectedFile = file;
    transcriptionData = [];
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.classList.remove("hidden");
    resultSection.classList.add("hidden");
    transcribeButton.disabled = false;
    setupMediaPreview(file);
}

function setupMediaPreview(file) {
    if (mediaObjectUrl) URL.revokeObjectURL(mediaObjectUrl);
    mediaObjectUrl = URL.createObjectURL(file);
    const isVideo = file.type.startsWith("video/") || /\.(mp4|webm|mov|mkv)$/i.test(file.name);

    audioPreview.pause();
    videoPreview.pause();
    audioPreview.removeAttribute("src");
    videoPreview.removeAttribute("src");

    if (isVideo) {
        videoPreview.src = mediaObjectUrl;
        videoPreview.classList.remove("hidden");
        audioPreview.classList.add("hidden");
    } else {
        audioPreview.src = mediaObjectUrl;
        audioPreview.classList.remove("hidden");
        videoPreview.classList.add("hidden");
    }
    mediaPreview.classList.remove("hidden");
}

function removeSelectedFile() {
    if (mediaObjectUrl) URL.revokeObjectURL(mediaObjectUrl);
    mediaObjectUrl = null;
    selectedFile = null;
    transcriptionData = [];
    audioFile.value = "";
    fileInfo.classList.add("hidden");
    mediaPreview.classList.add("hidden");
    audioPreview.removeAttribute("src");
    videoPreview.removeAttribute("src");
    resultSection.classList.add("hidden");
    transcribeButton.disabled = true;
}

audioFile.addEventListener("change", e => {
    if (e.target.files[0]) selectFile(e.target.files[0]);
});
removeFile.addEventListener("click", removeSelectedFile);

dropZone.addEventListener("dragover", e => {
    e.preventDefault();
    dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("dragging");
    if (e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]);
});

async function loadModels() {
    if (modelsLoaded) return;

    setProgress(5, "Checking browser acceleration...");
    let device = "wasm";

    if ("gpu" in navigator) {
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (adapter) device = "webgpu";
        } catch (error) {
            console.warn("WebGPU unavailable; using WASM.", error);
        }
    }

    const whisperOptions = device === "webgpu"
        ? { device: "webgpu", dtype: { encoder_model: "fp32", decoder_model_merged: "q4" } }
        : { device: "wasm", dtype: "q8" };

    setProgress(10, `Loading Whisper Base (${device.toUpperCase()})...`);
    transcriber = await pipeline("automatic-speech-recognition", ASR_MODEL, whisperOptions);
    // Speaker diarization is optional and is loaded only after Whisper succeeds.
    // A diarization failure must never prevent the transcript from appearing.
    modelsLoaded = true;
    setProgress(55, "Whisper model ready.");
}

async function decodeAudio(file) {
    setProgress(58, "Reading and normalising audio to 16 kHz mono...");
    const buffer = await file.arrayBuffer();
    const context = new AudioContext();
    const sourceBuffer = await context.decodeAudioData(buffer);
    const duration = sourceBuffer.duration;
    const sampleRate = 16000;
    const samplesCount = Math.ceil(duration * sampleRate);
    const offline = new OfflineAudioContext(1, samplesCount, sampleRate);
    const source = offline.createBufferSource();
    source.buffer = sourceBuffer;
    source.connect(offline.destination);
    source.start();
    const rendered = await offline.startRendering();
    const samples = rendered.getChannelData(0);
    await context.close();
    return { samples, duration };
}

async function transcribeAudio(samples, duration, variant, sensitivity) {
    const SAMPLE_RATE = 16000;
    const chunkSamples = TRANSCRIPTION_CHUNK_SECONDS * SAMPLE_RATE;
    const allChunks = [];
    const totalChunks = Math.ceil(samples.length / chunkSamples);

    // Whisper itself accepts only ~30 seconds per inference window. We process
    // the recording one window at a time and merge the timestamped results
    // onto the original timeline. 29 seconds is used instead of exactly 30
    // because whisper-base_timestamped has a known 30s timestamp issue.
    for (let offset = 0, chunkIndex = 0; offset < samples.length; offset += chunkSamples, chunkIndex++) {
        const end = Math.min(offset + chunkSamples, samples.length);
        const chunk = samples.slice(offset, end);
        const chunkStart = offset / SAMPLE_RATE;
        const chunkEnd = end / SAMPLE_RATE;

        setProgress(
            60 + Math.round((chunkIndex / totalChunks) * 18),
            "Transcribing chunk " + (chunkIndex + 1) + " of " + totalChunks +
            " • " + formatTime(chunkStart) + " → " + formatTime(Math.min(chunkEnd, duration))
        );

        const result = await transcriber(chunk, {
            language: "english",
            task: "transcribe",
            initial_prompt: variant.prompt,
            return_timestamps: "word",
            no_speech_threshold: sensitivity.noSpeechThreshold,
            logprob_threshold: -1.0,
            compression_ratio_threshold: 2.4
        });

        for (const word of result?.chunks || []) {
            const timestamp = word.timestamp || [];
            const localStart = Number(timestamp[0]);
            const localEnd = Number(timestamp[1]);

            if (!Number.isFinite(localStart)) continue;

            const start = chunkStart + Math.max(0, localStart);
            const endTime = chunkStart + (
                Number.isFinite(localEnd) && localEnd > localStart
                    ? localEnd
                    : localStart + 0.1
            );

            if (start >= duration) continue;

            allChunks.push({
                text: word.text,
                timestamp: [
                    Math.min(start, duration),
                    Math.min(Math.max(start + 0.05, endTime), duration)
                ]
            });
        }
    }

    // Keep the original word order and remove duplicate/empty timestamped
    // words at chunk boundaries.
    const chunks = [];
    for (const chunk of allChunks) {
        const text = cleanText(chunk.text || "");
        if (!text) continue;

        const previous = chunks.at(-1);
        const previousText = previous ? cleanText(previous.text).toLowerCase() : "";
        const currentText = text.toLowerCase();

        if (
            previous &&
            previousText === currentText &&
            Math.abs(chunk.timestamp[0] - previous.timestamp[0]) < 0.75
        ) {
            previous.timestamp[1] = Math.max(previous.timestamp[1], chunk.timestamp[1]);
            continue;
        }

        chunks.push({
            text,
            timestamp: chunk.timestamp
        });
    }

    setProgress(78, "All chunks transcribed. Preparing transcript...");
    return { chunks };
}

async function detectSpeakers(samples) {
    setProgress(82, "Loading speaker detection...");

    if (!segmentationProcessor || !segmentationModel) {
        segmentationProcessor = await AutoProcessor.from_pretrained(SEGMENTATION_MODEL);
        segmentationModel = await AutoModelForAudioFrameClassification.from_pretrained(
            SEGMENTATION_MODEL,
            { device: "wasm", dtype: "fp32" }
        );
    }

    const SAMPLE_RATE = 16000;
    const WINDOW_SECONDS = 30;
    const windowSamples = WINDOW_SECONDS * SAMPLE_RATE;
    const allSegments = [];

    for (let offset = 0; offset < samples.length; offset += windowSamples) {
        const end = Math.min(offset + windowSamples, samples.length);
        const window = samples.slice(offset, end);
        const inputs = await segmentationProcessor(window);
        const output = await segmentationModel(inputs);
        const processed =
            segmentationProcessor.post_process_speaker_diarization(output.logits, window.length)[0] || [];

        for (const segment of processed) {
            const confidence = Number(segment.confidence ?? 1);
            if (!Number.isFinite(segment.start) ||
                !Number.isFinite(segment.end) ||
                segment.end <= segment.start ||
                confidence < SPEAKER_CONFIDENCE) continue;

            allSegments.push({
                ...segment,
                start: segment.start + offset / SAMPLE_RATE,
                end: segment.end + offset / SAMPLE_RATE,
                label: "SPEAKER_" + segment.id
            });
        }

        const percent = 82 + Math.round((end / samples.length) * 13);
        setProgress(percent, "Detecting speaker changes... " +
            Math.round((end / samples.length) * 100) + "%");
    }

    const merged = [];
    for (const segment of allSegments) {
        const previous = merged.at(-1);
        if (previous &&
            previous.label === segment.label &&
            segment.start - previous.end <= SPEAKER_MERGE_GAP) {
            previous.end = Math.max(previous.end, segment.end);
            previous.confidence = Math.max(
                previous.confidence ?? 0,
                segment.confidence ?? 0
            );
        } else {
            merged.push({ ...segment });
        }
    }
    return merged;
}

function speakerForWord(word, segments, previousSpeaker) {
    const [rawStart, rawEnd] = word.timestamp || [];
    const start = Number(rawStart);
    const end = Number.isFinite(Number(rawEnd)) ? Number(rawEnd) : start + 0.05;
    if (!Number.isFinite(start)) return previousSpeaker || "Speaker 1";

    let best = null;
    let bestOverlap = 0;
    for (const segment of segments) {
        const overlap = Math.max(0, Math.min(end, segment.end) - Math.max(start, segment.start));
        const midpointMatch = overlap === 0 && (start + end) / 2 >= segment.start && (start + end) / 2 <= segment.end;
        if (overlap > bestOverlap || (midpointMatch && !best)) {
            best = segment;
            bestOverlap = overlap;
        }
    }
    return best?.label || previousSpeaker || "Speaker 1";
}

function buildTranscript(wordChunks, speakerSegments) {
    const words = [];
    let previousSpeaker = null;

    for (const chunk of wordChunks || []) {
        const text = (chunk.text || "").trim();
        const [rawStart, rawEnd] = chunk.timestamp || [];
        const start = Number(rawStart);
        const end = Number(rawEnd);
        if (!text || !Number.isFinite(start)) continue;

        const rawSpeaker = speakerForWord(chunk, speakerSegments, previousSpeaker);
        previousSpeaker = rawSpeaker;
        words.push({ text, start, end: Number.isFinite(end) ? end : start + 0.1, rawSpeaker });
    }

    const speakerMap = new Map();
    let speakerNumber = 1;
    for (const word of words) {
        if (!speakerMap.has(word.rawSpeaker)) speakerMap.set(word.rawSpeaker, `Speaker ${speakerNumber++}`);
        word.speaker = speakerMap.get(word.rawSpeaker);
    }

    const blocks = [];
    for (const word of words) {
        const previous = blocks.at(-1);
        if (previous && previous.speaker === word.speaker && word.start - previous.end <= SPEAKER_MERGE_GAP) {
            previous.words.push(word);
            previous.end = Math.max(previous.end, word.end);
        } else {
            blocks.push({ speaker: word.speaker, start: word.start, end: word.end, words: [word] });
        }
    }

    const output = [];
    for (const block of blocks) {
        let current = null;
        for (const word of block.words) {
            if (!current) {
                current = { speaker: block.speaker, start: word.start, end: word.end, words: [word] };
                continue;
            }
            if (word.end - current.start > MAX_BLOCK_SECONDS) {
                output.push(current);
                current = { speaker: block.speaker, start: word.start, end: word.end, words: [word] };
            } else {
                current.words.push(word);
                current.end = Math.max(current.end, word.end);
            }
        }
        if (current) output.push(current);
    }

    return output.map(block => ({
        ...block,
        text: cleanText(block.words.map(word => word.text).join(" "))
    }));
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function getMediaElement() {
    return videoPreview.classList.contains("hidden") ? audioPreview : videoPreview;
}

function renderTranscript() {
    speakerTranscript.innerHTML = "";

    if (!transcriptionData.length) {
        speakerTranscript.innerHTML = `<p class="transcript-text">No speech was detected.</p>`;
        return;
    }

    transcriptionData.forEach((block, blockIndex) => {
        const article = document.createElement("article");
        article.className = "transcript-block";
        article.dataset.index = blockIndex;
        article.dataset.start = block.start;
        article.dataset.end = block.end;

        const meta = document.createElement("div");
        meta.className = "transcript-meta";
        meta.innerHTML = `<span class="speaker-label">${escapeHtml(block.speaker)}</span><span class="speaker-time">${formatTime(block.start)} — ${formatTime(block.end)}</span>`;

        const paragraph = document.createElement("p");
        paragraph.className = "transcript-text";
        block.words.forEach((word, wordIndex) => {
            const span = document.createElement("span");
            span.className = "transcript-word";
            span.textContent = word.text + (wordIndex === block.words.length - 1 ? "" : " ");
            span.dataset.start = word.start;
            span.addEventListener("click", event => {
                event.stopPropagation();
                seekTo(word.start);
            });
            paragraph.appendChild(span);
        });

        article.append(meta, paragraph);
        article.addEventListener("click", () => seekTo(block.start));
        speakerTranscript.appendChild(article);
    });
}

function seekTo(seconds) {
    const media = getMediaElement();
    if (!media.src) return;
    media.currentTime = Math.max(0, Number(seconds) || 0);
    media.focus({ preventScroll: true });
}

function updateActiveTranscript() {
    const media = getMediaElement();
    if (!media.src || !transcriptionData.length) return;
    const time = media.currentTime;
    let activeIndex = -1;
    transcriptionData.forEach((block, index) => {
        if (time >= block.start && time <= block.end) activeIndex = index;
    });

    document.querySelectorAll(".transcript-block").forEach((element, index) => {
        element.classList.toggle("active", index === activeIndex);
    });

    if (activeIndex >= 0) {
        const active = speakerTranscript.querySelector(`[data-index="${activeIndex}"]`);
        if (active && !speakerTranscript.matches(":hover")) active.scrollIntoView({ block: "nearest" });
    }
}

audioPreview.addEventListener("timeupdate", updateActiveTranscript);
videoPreview.addEventListener("timeupdate", updateActiveTranscript);

transcribeButton.addEventListener("click", async () => {
    if (!selectedFile) return;

    try {
        transcribeButton.disabled = true;
        resultSection.classList.add("hidden");
        progressArea.classList.remove("hidden");
        transcriptionData = [];
        transcription.value = "";

        const variant =
            ENGLISH_VARIANTS[languageSelect.value] || ENGLISH_VARIANTS["en-US"];
        const sensitivity =
            SENSITIVITY[sensitivitySelect.value] || SENSITIVITY.balanced;

        await loadModels();
        const audio = await decodeAudio(selectedFile);

        if (audio.duration > MAX_AUDIO_DURATION_SECONDS) {
            throw new Error(
                "Audio/video is too long. The current limit is " +
                formatTime(MAX_AUDIO_DURATION_SECONDS) +
                ". Your file is " + formatTime(audio.duration) + "."
            );
        }

        setProgress(
            60,
            "Transcribing with " + variant.label + " • " +
            sensitivity.label + " capture..."
        );

        const result = await transcribeAudio(
            audio.samples,
            audio.duration,
            variant,
            sensitivity
        );

        // Whisper is the primary operation. Show its result immediately.
        const wordChunks = result?.chunks || [];

        if (wordChunks.length) {
            transcriptionData = buildTranscript(wordChunks, []);
        } else if (result?.text?.trim()) {
            const fallbackText = cleanText(result.text);
            transcriptionData = [{
                speaker: "Speaker 1",
                start: 0,
                end: audio.duration,
                words: [{
                    text: fallbackText,
                    start: 0,
                    end: audio.duration
                }],
                text: fallbackText
            }];
        }

        transcription.value = transcriptionData
            .map(block => block.speaker + "\n" + block.text)
            .join("\n\n");

        renderTranscript();
        resultSection.classList.remove("hidden");
        setProgress(78, "Transcript ready. Detecting speakers...");

        // Speaker detection is best-effort. If it fails, keep the transcript.
        try {
            const speakerSegments = await detectSpeakers(audio.samples);

            if (speakerSegments.length && wordChunks.length) {
                transcriptionData = buildTranscript(wordChunks, speakerSegments);
                transcription.value = transcriptionData
                    .map(block => block.speaker + "\n" + block.text)
                    .join("\n\n");
                renderTranscript();
            }
        } catch (speakerError) {
            console.warn(
                "Speaker detection unavailable; keeping transcript.",
                speakerError
            );
        }

        const speakerCount =
            new Set(transcriptionData.map(block => block.speaker)).size;

        setProgress(100, "Transcription complete.");

        resultStatus.textContent =
            "Processed " + formatTime(audio.duration) + " • " +
            (speakerCount || 1) + " speaker(s) • " +
            variant.label + " • " + sensitivity.label + " capture";

        resultSection.classList.remove("hidden");
    } catch (error) {
        console.error("TRANSCRIPTION ERROR:", error);
        setProgress(0, "Transcription failed.");
        resultSection.classList.add("hidden");

        alert(
            "Transcription failed:\n\n" +
            (error?.message || error) +
            "\n\nIf this is the first run, make sure the browser can download the Whisper model."
        );
    } finally {
        transcribeButton.disabled = !selectedFile;
    }
});

function getOutputFilename(extension) {
    if (!selectedFile) return `transcription.${extension}`;
    const dot = selectedFile.name.lastIndexOf(".");
    const base = dot > 0 ? selectedFile.name.slice(0, dot) : selectedFile.name;
    return `${base}_transcription.${extension}`;
}

function downloadBlob(content, filename, type) {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
}

downloadTxt.addEventListener("click", () => {
    if (transcription.value.trim()) downloadBlob(transcription.value, getOutputFilename("txt"), "text/plain;charset=utf-8");
});

downloadSrt.addEventListener("click", () => {
    if (!transcriptionData.length) return;
    const srt = transcriptionData.map((block, index) => {
        const end = Math.max(block.end, block.start + 0.1);
        return `${index + 1}\n${formatSrtTime(block.start)} --> ${formatSrtTime(end)}\n${block.speaker}\n${block.text}`;
    }).join("\n\n");
    downloadBlob(`${srt}\n`, getOutputFilename("srt"), "application/x-subrip;charset=utf-8");
});

copyTranscript.addEventListener("click", async () => {
    if (!transcription.value.trim()) return;
    try {
        await navigator.clipboard.writeText(transcription.value);
        const original = copyTranscript.textContent;
        copyTranscript.textContent = "Copied";
        setTimeout(() => { copyTranscript.textContent = original; }, 1200);
    } catch (error) {
        console.warn("Clipboard copy failed.", error);
    }
});

console.log("Local Whisper ready — minimal UI + clickable transcript viewer.");
