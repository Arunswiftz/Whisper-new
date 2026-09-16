import express from "express";
import cors from "cors";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { spawn } from "node:child_process";

const app = express();
const PORT = Number(process.env.PORT || 3000);
const MAX_DOWNLOAD_BYTES = 250 * 1024 * 1024;
const PUBLIC_DIR = process.cwd();

app.use(cors());
app.use(express.json({ limit: "20kb" }));
app.use(express.static(PUBLIC_DIR));

function isSafeUrl(value) {
    let url;
    try { url = new URL(value); } catch { return false; }
    if (!["http:", "https:"].includes(url.protocol)) return false;
    const host = url.hostname.toLowerCase();
    const blockedHosts = new Set(["localhost","127.0.0.1","0.0.0.0","::1","169.254.169.254","metadata.google.internal"]);
    if (blockedHosts.has(host) || host.endsWith(".local")) return false;
    return true;
}


async function resolveApplePodcastsEpisode(url) {
    const parsed = new URL(url);
    if (!parsed.hostname.toLowerCase().endsWith("podcasts.apple.com")) return null;

    const match = parsed.pathname.match(/\/id(\d+)/i);
    const episodeId = parsed.searchParams.get("i");
    if (!match || !episodeId) {
        throw new Error("Apple Podcasts links must point to a specific episode and include the ?i= episode ID.");
    }

    const showId = match[1];
    const apiUrl = "https://itunes.apple.com/lookup?id=" + encodeURIComponent(showId) +
        "&entity=podcastEpisode&limit=200&country=" + encodeURIComponent((parsed.pathname.match(/^\/([a-z]{2})\//i)?.[1] || "us"));

    const response = await fetch(apiUrl, {
        headers: { "User-Agent": "Whisper-new/1.0" }
    });
    if (!response.ok) throw new Error("Apple's podcast metadata service returned HTTP " + response.status + ".");

    const data = await response.json();
    const episode = (data.results || []).find(item =>
        String(item.trackId || "") === String(episodeId)
    );

    if (!episode) {
        throw new Error("The Apple Podcasts episode was not found in Apple's recent episode metadata.");
    }

    const audioUrl = episode.episodeUrl;
    if (!audioUrl) {
        throw new Error("Apple returned the episode metadata, but no direct audio URL was available.");
    }

    return {
        audioUrl,
        title: episode.trackName || "Apple Podcasts episode",
        duration: Number(episode.trackTimeMillis || 0) / 1000
    };
}

async function downloadDirectAudio(url, outputPath) {
    const response = await fetch(url, {
        headers: {
            "User-Agent": "Mozilla/5.0 (compatible; Whisper-new/1.0)"
        },
        redirect: "follow"
    });
    if (!response.ok) throw new Error("Audio host returned HTTP " + response.status + ".");
    const contentLength = Number(response.headers.get("content-length") || 0);
    if (contentLength > MAX_DOWNLOAD_BYTES) throw new Error("The source audio exceeds the 250 MB server limit.");

    const buffer = Buffer.from(await response.arrayBuffer());
    if (!buffer.length || buffer.length > MAX_DOWNLOAD_BYTES) {
        throw new Error("The downloaded audio is empty or exceeds the 250 MB server limit.");
    }
    fs.writeFileSync(outputPath + ".source", buffer);
    await convertToWav(outputPath + ".source", outputPath);
    fs.rmSync(outputPath + ".source", { force: true });
}

function convertToWav(inputPath, outputPath) {
    return new Promise((resolve, reject) => {
        const child = spawn(process.env.FFMPEG_BIN || "ffmpeg", [
            "-y", "-i", inputPath, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", outputPath
        ], { stdio: ["ignore", "ignore", "pipe"] });
        let stderr = "";
        child.stderr.on("data", data => {
            stderr += data.toString();
            if (stderr.length > 8000) stderr = stderr.slice(-8000);
        });
        child.on("error", () => reject(new Error("FFmpeg could not be started. Install FFmpeg and make sure it is on PATH.")));
        child.on("close", code => code === 0 ? resolve() : reject(new Error(stderr.trim() || "FFmpeg audio conversion failed.")));
    });
}

function runYtDlp(url, outputPath) {
    return new Promise((resolve, reject) => {
        const args = [
            "--no-playlist", "--no-warnings", "--restrict-filenames",
            "--max-filesize", String(MAX_DOWNLOAD_BYTES),
            "--extract-audio", "--audio-format", "wav", "--audio-quality", "5",
            "--output", outputPath, url
        ];
        const child = spawn(process.env.YTDLP_BIN || "yt-dlp", args, { stdio: ["ignore","pipe","pipe"] });
        let stderr = "";
        child.stderr.on("data", data => {
            stderr += data.toString();
            if (stderr.length > 12000) stderr = stderr.slice(-12000);
        });
        child.on("error", () => reject(new Error("yt-dlp could not be started. Install yt-dlp and make sure it is on PATH.")));
        child.on("close", code => {
            if (code === 0 && fs.existsSync(outputPath)) resolve();
            else reject(new Error(stderr.trim() || "Media extraction failed."));
        });
    });
}

app.post("/api/media", async (req, res) => {
    const url = String(req.body?.url || "").trim();
    if (!url || !isSafeUrl(url)) return res.status(400).json({ error: "Please provide a valid public HTTP/HTTPS media URL." });

    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "whisper-media-"));
    const outputPath = path.join(tempDir, crypto.randomUUID() + ".wav");

    try {
        const appleEpisode = await resolveApplePodcastsEpisode(url).catch(() => null);

        if (appleEpisode) {
            await downloadDirectAudio(appleEpisode.audioUrl, outputPath);
        } else {
            await runYtDlp(url, outputPath);
        }

        const stat = fs.statSync(outputPath);
        if (!stat.size || stat.size > MAX_DOWNLOAD_BYTES) throw new Error("Downloaded media is empty or too large.");

        res.setHeader("Content-Type", "audio/wav");
        res.setHeader("Content-Disposition", 'inline; filename="online-media.wav"');
        res.sendFile(outputPath, error => {
            fs.rmSync(tempDir, { recursive: true, force: true });
            if (error && !res.headersSent) res.status(500).json({ error: "Could not send the converted audio." });
        });
    } catch (error) {
        fs.rmSync(tempDir, { recursive: true, force: true });
        console.error("Media extraction failed:", error.message);
        res.status(422).json({
            error: "This link could not be converted. The site may require authentication, block automated downloads, or the URL may not contain accessible media."
        });
    }
});

app.get("/api/health", (_req, res) => res.json({ ok: true, service: "Whisper-new media bridge" }));

app.listen(PORT, () => console.log("Whisper-new server running on http://localhost:" + PORT));