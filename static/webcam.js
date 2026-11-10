const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const empty = document.getElementById("empty");
const result = document.getElementById("result");
const start = document.getElementById("start");
const stop = document.getElementById("stop");
const capture = document.createElement("canvas");
let stream;
let session;
let generation = 0;

async function endSession(id) {
  if (id) {
    await fetch("/v1/sessions/" + encodeURIComponent(id), { method: "DELETE", keepalive: true });
  }
}

function stopCamera() {
  generation += 1;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = undefined;
  video.srcObject = null;
  empty.hidden = false;
  overlay.width = 0;
  const old = session;
  session = undefined;
  endSession(old).catch(() => {});
  start.disabled = false;
  stop.disabled = true;
  result.textContent = "Ready";
}

async function nextFrame(active) {
  if (generation !== active) return;
  try {
    if (!video.videoWidth || !video.videoHeight) {
      setTimeout(() => nextFrame(active), 100);
      return;
    }
    const width = Math.min(720, video.videoWidth);
    const height = Math.max(1, Math.round(video.videoHeight * width / video.videoWidth));
    capture.width = width;
    capture.height = height;
    capture.getContext("2d").drawImage(video, 0, 0, width, height);
    const jpeg = await new Promise((resolve) => capture.toBlob(resolve, "image/jpeg", 0.85));
    if (generation !== active) return;
    const response = await fetch("/v1/sessions/" + encodeURIComponent(session) + "/frames", {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: jpeg,
    });
    const prediction = await response.json();
    if (generation !== active) return;
    if (!response.ok) throw new Error(prediction.error || "Frame rejected.");
    overlay.width = width;
    overlay.height = height;
    if (!prediction.box) {
      result.textContent = "Waiting for exactly one face";
    } else {
      const [left, top, right, bottom] = prediction.box;
      const context = overlay.getContext("2d");
      context.strokeStyle = prediction.label === "real" ? "#66f3cf" : "#ff8888";
      context.lineWidth = 3;
      context.strokeRect(left, top, right - left, bottom - top);
      result.textContent = prediction.label.toUpperCase() + " · P(real) " + prediction.p_real.toFixed(2);
    }
    setTimeout(() => nextFrame(active), 180);
  } catch (error) {
    if (generation === active) {
      stopCamera();
      result.textContent = error.message;
    }
  }
}

start.addEventListener("click", async () => {
  const active = ++generation;
  start.disabled = true;
  stop.disabled = false;
  result.textContent = "Starting camera…";
  try {
    const camera = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    if (generation !== active) {
      camera.getTracks().forEach((track) => track.stop());
      return;
    }
    stream = camera;
    video.srcObject = stream;
    await video.play();
    const response = await fetch("/v1/sessions", { method: "POST" });
    const body = await response.json();
    if (generation !== active) {
      await endSession(body.session_id);
      return;
    }
    if (!response.ok) throw new Error(body.error || "Session unavailable.");
    session = body.session_id;
    empty.hidden = true;
    await nextFrame(active);
  } catch (error) {
    if (generation === active) {
      stopCamera();
      result.textContent = error.message;
    }
  }
});

stop.addEventListener("click", stopCamera);
window.addEventListener("pagehide", stopCamera);
