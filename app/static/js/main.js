// ─── State ─────────────────────────────────────────────
let mobileVideo   = null;
let mobileCanvas  = null;
let mobileCtx     = null;
let overlayCanvas = null;
let overlayCtx    = null;

let isDetecting = false;
let history     = [];

// Timer: chỉ chạy khi detection đang ON
let elapsedMs       = 0;       // ms đã tích luỹ
let timerStart      = null;    // null = đang pause
let elapsedInterval = null;

// ─── Config ────────────────────────────────────────────
const DETECT_INTERVAL_MS = 500;
const SEND_W = 640;
const SEND_H = 480;
const JPEG_QUALITY = 0.6;

// ─── Timer helpers ─────────────────────────────────────
function tickTimer() {
    const ms = (timerStart !== null)
        ? elapsedMs + (Date.now() - timerStart)
        : elapsedMs;
    const sec = Math.floor(ms / 1000);
    const m = String(Math.floor(sec/60)).padStart(2, "0");
    const s = String(sec % 60).padStart(2, "0");
    document.getElementById("elapsed").textContent = m + ":" + s;
}

function startTimer() {
    if (timerStart === null) {
        timerStart = Date.now();
    }
    if (!elapsedInterval) {
        elapsedInterval = setInterval(tickTimer, 1000);
    }
    tickTimer();
}

function pauseTimer() {
    if (timerStart !== null) {
        elapsedMs += Date.now() - timerStart;
        timerStart = null;
    }
    if (elapsedInterval) {
        clearInterval(elapsedInterval);
        elapsedInterval = null;
    }
    tickTimer();
}

function resetTimer() {
    elapsedMs = 0;
    if (timerStart !== null) timerStart = Date.now();
    tickTimer();
}

// ─── Camera ────────────────────────────────────────────
async function startMobileCamera() {
    try {
        mobileVideo   = document.getElementById("mobileVideo");
        overlayCanvas = document.getElementById("overlayCanvas");
        overlayCtx    = overlayCanvas.getContext("2d");

        if (!mobileCanvas) {
            mobileCanvas = document.createElement("canvas");
            mobileCanvas.width  = SEND_W;
            mobileCanvas.height = SEND_H;
            mobileCtx = mobileCanvas.getContext("2d");
        }

        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" },
            audio: false
        });

        mobileVideo.srcObject = stream;
        await mobileVideo.play();

        mobileVideo.style.display       = "block";
        overlayCanvas.style.display     = "block";
        document.getElementById("camOff").style.display = "none";
        document.getElementById("btnToggleCam2").style.display = "block";

        overlayCanvas.width  = mobileVideo.videoWidth;
        overlayCanvas.height = mobileVideo.videoHeight;
    } catch (err) {
        console.error(err);
        alert("Không bật được camera. Safari/iPhone yêu cầu HTTPS.");
    }
}

function stopCam() {
    isDetecting = false;
    pauseTimer();
    elapsedMs = 0;
    document.getElementById("elapsed").textContent = "00:00";

    if (mobileVideo && mobileVideo.srcObject) {
        mobileVideo.srcObject.getTracks().forEach(t => t.stop());
        mobileVideo.srcObject = null;
    }
    if (mobileVideo)   mobileVideo.style.display   = "none";
    if (overlayCanvas) overlayCanvas.style.display = "none";
    document.getElementById("camOff").style.display        = "flex";
    document.getElementById("btnToggleCam2").style.display = "none";
}

// ─── Detection ─────────────────────────────────────────
function startDetection() {
    if (!mobileVideo) {
        alert("Hãy bật camera trước!");
        return;
    }
    if (isDetecting) return;
    isDetecting = true;
    startTimer();
    detectLoop();
    console.log("Detection started");
}

function stopDetection() {
    isDetecting = false;
    pauseTimer();
    if (overlayCtx) {
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
    console.log("Detection stopped");
}

async function detectLoop() {
    while (isDetecting) {
        const t0 = performance.now();
        try {
            await detectOnce();
        } catch (e) {
            console.error("detect error:", e);
        }
        const elapsed = performance.now() - t0;
        const wait = Math.max(0, DETECT_INTERVAL_MS - elapsed);
        await new Promise(r => setTimeout(r, wait));
    }
}

async function detectOnce() {
    if (!mobileVideo || mobileVideo.videoWidth === 0) return;

    mobileCtx.drawImage(mobileVideo, 0, 0, SEND_W, SEND_H);
    const imageData = mobileCanvas.toDataURL("image/jpeg", JPEG_QUALITY);

    const resp = await fetch("/detect_frame", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ image: imageData })
    });
    if (!resp.ok) throw new Error("HTTP " + resp.status);

    const result = await resp.json();
    console.log("server →", result.boxes.length, "boxes, count =", result.count);

    document.getElementById("cnt").textContent = result.count;
    drawOverlay(result.boxes, result.line_y, result.frame_w, result.frame_h);
}

// ─── Vẽ overlay ────────────────────────────────────────
function drawOverlay(boxes, lineY, srcW, srcH) {
    if (overlayCanvas.width !== mobileVideo.videoWidth ||
        overlayCanvas.height !== mobileVideo.videoHeight) {
        overlayCanvas.width  = mobileVideo.videoWidth;
        overlayCanvas.height = mobileVideo.videoHeight;
    }
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    const sx = overlayCanvas.width  / srcW;
    const sy = overlayCanvas.height / srcH;

    const ly = lineY * sy;
    overlayCtx.beginPath();
    overlayCtx.moveTo(0, ly);
    overlayCtx.lineTo(overlayCanvas.width, ly);
    overlayCtx.lineWidth   = 4;
    overlayCtx.strokeStyle = "red";
    overlayCtx.stroke();

    overlayCtx.lineWidth   = 3;
    overlayCtx.strokeStyle = "#22c55e";
    overlayCtx.fillStyle   = "#22c55e";
    overlayCtx.font        = "18px Arial";

    boxes.forEach(b => {
        const x = b.x1 * sx;
        const y = b.y1 * sy;
        const w = (b.x2 - b.x1) * sx;
        const h = (b.y2 - b.y1) * sy;
        overlayCtx.strokeRect(x, y, w, h);
        overlayCtx.fillText("Vehicle", x, y - 8);
    });
}

// ─── Reset / Save / History ────────────────────────────
function doReset() {
    fetch("/reset")
        .then(r => r.json())
        .then(() => {
            resetTimer();
            document.getElementById("cnt").textContent = 0;
            renderHistory();
        })
        .catch(() => { resetTimer(); });
}

function doSave() {
    fetch("/stats")
        .then(r => r.json())
        .then(d => {
            const ms = (timerStart !== null)
                ? elapsedMs + (Date.now() - timerStart)
                : elapsedMs;
            const sec = Math.floor(ms / 1000);
            const m = String(Math.floor(sec/60)).padStart(2, "0");
            const s = String(sec % 60).padStart(2, "0");
            history.unshift({
                count:    d.count,
                duration: m + ":" + s,
                time:     new Date().toLocaleTimeString("vi-VN")
            });
            renderHistory();
        })
        .catch(() => {});
}

function renderHistory() {
    const wrap = document.getElementById("historyWrap");
    if (history.length === 0) {
        wrap.innerHTML = "<div class='empty-history'>Chưa có phiên nào — nhấn Lưu để ghi lại</div>";
        return;
    }
    let html = "<table class='history-table'><thead><tr><th>#</th><th>Thời điểm reset</th><th>Thời gian đếm</th><th>Số xe</th></tr></thead><tbody>";
    history.forEach((h, i) => {
        html += "<tr><td>" + (history.length - i) + "</td><td>" + h.time + "</td><td>" + h.duration + "</td><td class='count-cell'>" + h.count + " xe</td></tr>";
    });
    html += "</tbody></table>";
    wrap.innerHTML = html;
}