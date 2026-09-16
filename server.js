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
        await runYtDlp(url, outputPath);
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