import { YOLO, annotate } from "@ultralytics/yolo";

/**
 * UniformCheck: in-browser uniform attribute detection
 * ----------------------------------------------------
 * Keep the default model in ./models/ so the app works from a GitHub Pages
 * project site (including repositories served below /<repository-name>/).
 *
 * Uniform logic: the model's class names are matched against the keyword lists
 * in CONFIG.attributeClasses. If your custom model uses other class names
 * (for example "dasi_abu" or "topi_sma"), add the words there.
 * Note: the stock COCO model only knows "tie", not hats, so it can never
 * report a complete uniform. Use a model trained on both classes.
 */
const CONFIG = Object.freeze({
  defaultModelPath: "models/yolo26n.onnx",
  defaultConfidence: 0.25,
  defaultLang: "id",
  // Camera frames flicker; an attribute counts as present this long after it was last seen.
  attributeHoldMs: 700,
  // Whole-word matches against the detected class name (split on non-letters).
  attributeClasses: Object.freeze({
    tie: ["tie", "necktie", "dasi"],
    hat: ["hat", "cap", "topi"],
  }),
  // A class such as "no_tie" or "tanpa_topi" must not count as the attribute being worn.
  negationWords: ["no", "non", "without", "tanpa", "missing"],
  annotation: {
    lineWidth: 3,
    font: "600 14px IBM Plex Sans, sans-serif",
    labels: true,
    keypoints: true,
  },
});

/* ---------- Translations ---------- */

const I18N = {
  id: {
    "meta.title": "UniformCheck — Deteksi seragam di browser",
    "meta.desc": "UniformCheck memeriksa kelengkapan atribut seragam SMK (dasi dan topi) langsung di browser, tanpa server inferensi.",
    "chip.runtime": "Berjalan di perangkat Anda",
    "link.docs": "Dokumentasi model",
    "lang.label": "Pilih bahasa",
    "hero.copy": "Periksa kelengkapan seragam SMK, dasi dan topi, langsung lewat kamera atau foto. Deteksi berjalan sepenuhnya di browser, jadi gambar Anda tidak pernah meninggalkan perangkat.",
    "hero.art": "Ilustrasi empat siswa SMK: dua laki-laki dan dua perempuan, salah satunya berhijab",
    "attr.hat": "Topi",
    "attr.tie": "Dasi",
    "attr.found": "Terdeteksi",
    "attr.missing": "Belum terdeteksi",
    "attr.unknown": "—",
    "verdict.idle.title": "Belum ada pemeriksaan",
    "verdict.idle.desc": "Muat model, lalu nyalakan kamera atau pilih foto untuk memeriksa dasi dan topi.",
    "verdict.none.title": "Atribut belum terdeteksi",
    "verdict.none.desc": "Dasi dan topi tidak terlihat. Pastikan siswa tampak jelas di dalam gambar.",
    "verdict.complete.title": "Atribut lengkap",
    "verdict.complete.desc": "Dasi dan topi sudah dipakai.",
    "verdict.missingHat.title": "Atribut kurang lengkap",
    "verdict.missingHat.desc": "Dasi terdeteksi, tetapi topi belum dipakai.",
    "verdict.missingTie.title": "Atribut kurang lengkap",
    "verdict.missingTie.desc": "Topi terdeteksi, tetapi dasi belum dipakai.",
    "live.title": "Pemeriksaan langsung",
    "btn.clear": "Hapus pratinjau",
    "empty.title": "Belum ada yang diperiksa",
    "empty.desc": "Muat model, lalu nyalakan kamera atau pilih foto.",
    "btn.startCamera": "Nyalakan kamera",
    "btn.stopCamera": "Matikan kamera",
    "metric.group": "Metrik deteksi",
    "metric.detections": "Deteksi",
    "metric.detectionsFoot": "Objek pada frame saat ini",
    "metric.inference": "Waktu inferensi",
    "metric.inferenceFoot": "Eksekusi model per frame",
    "metric.fps": "Frame rate",
    "metric.fpsFoot": "Perkiraan kecepatan langsung",
    "model.title": "Model",
    "field.path": "Lokasi model",
    "field.pathNote": "Letakkan file ONNX di folder <code>models/</code>, atau pilih file lokal di bawah. <a class=\"inline-link\" href=\"https://huggingface.co/prithivMLmods/YOLO26-ONNX/resolve/main/yolo26n/yolo26n.onnx\" target=\"_blank\" rel=\"noreferrer\">Unduh YOLO26n ONNX ↗</a>",
    "field.otherModel": "Gunakan model lain",
    "field.optional": "Opsional",
    "field.chooseOnnx": "Pilih file ONNX",
    "field.localFile": "File lokal, .onnx",
    "btn.loadModel": "Muat model",
    "btn.reloadModel": "Muat ulang model",
    "btn.loading": "Memuat model…",
    "settings.title": "Pengaturan",
    "field.confidence": "Ambang kepercayaan",
    "range.low": "Lebih banyak deteksi",
    "range.high": "Lebih yakin",
    "field.camera": "Kamera",
    "cam.back": "Kamera belakang (ponsel)",
    "cam.front": "Kamera depan",
    "photo.title": "Periksa foto",
    "photo.note": "Jalankan satu foto lewat model yang sudah dimuat dan lihat hasilnya.",
    "btn.choosePhoto": "Pilih foto",
    "results.title": "Hasil",
    "footer.note": "Deteksi berjalan lokal di browser Anda. File model dan runtime dimuat sesuai kebutuhan.",
    "common.unknown": "kesalahan tidak diketahui",
    "pill.model.none": "Model belum dimuat",
    "pill.model.loading": "Memuat model",
    "pill.model.ready": "Model siap",
    "pill.model.error": "Model gagal",
    "pill.device": "Perangkat {device}",
    "pill.device.none": "Perangkat —",
    "mode.standby": "Siaga",
    "mode.live": "Kamera langsung",
    "mode.camError": "Kamera bermasalah",
    "mode.photo": "Periksa foto",
    "hint.default": "Media Anda tetap berada di sesi browser ini.",
    "hint.modelLoaded": "Model dimuat. Pilih kamera atau foto untuk memulai.",
    "hint.live": "Frame langsung diproses secara lokal di browser ini.",
    "hint.camOff": "Kamera mati. Model siap untuk input lain.",
    "hint.photo": "Foto lokal: {name}",
    "status.waiting": "Menunggu model dimuat",
    "status.loadingModel": "Memuat model dan menyiapkan runtime browser…",
    "status.modelFailed": "Gagal memuat model: {msg}",
    "status.modelFailedGeneric": "Gagal memuat model. Periksa file model lalu coba lagi.",
    "status.modelReady": "Model siap. Nyalakan kamera atau periksa foto.",
    "status.waitInput": "Model siap. Menunggu input.",
    "status.loadFirst": "Muat model sebelum menyalakan kamera.",
    "status.noCamera": "Akses kamera tidak tersedia. Gunakan HTTPS atau localhost, atau periksa foto.",
    "status.requestPerm": "Meminta izin kamera…",
    "status.cameraActive": "Kamera aktif. Memeriksa atribut…",
    "status.permDenied": "Izin kamera ditolak. Izinkan akses kamera atau periksa foto.",
    "status.cameraFail": "Tidak dapat menyalakan kamera: {msg}",
    "status.cameraStopped": "Kamera dimatikan. Model tetap dimuat.",
    "status.inferStopped": "Deteksi berhenti: {msg}",
    "status.badImage": "Pilih file gambar yang didukung.",
    "status.loadingImage": "Memuat foto…",
    "status.imageDone": "Pemeriksaan foto selesai.",
    "status.imageFail": "Pemeriksaan foto gagal: {msg}",
    "status.imageDecode": "Foto yang dipilih tidak dapat dibaca oleh browser ini.",
    "list.empty": "Hasil deteksi akan muncul di sini.",
    "list.waitFirst": "Menunggu prediksi pertama…",
    "list.analyzing": "Menganalisis foto…",
    "list.imageFailed": "Pemeriksaan foto gagal.",
    "list.noPred": "Tidak ada prediksi di atas ambang kepercayaan.",
    "det.class": "Kelas {cls}",
    "det.classTrack": "Kelas {cls}, jalur {id}",
    "det.top": "Kelas teratas {cls}",
  },
  en: {
    "meta.title": "UniformCheck — Uniform detection in your browser",
    "meta.desc": "UniformCheck checks that a vocational high school (SMK) uniform is complete (tie and hat) right in your browser, with no inference server.",
    "chip.runtime": "Runs on your device",
    "link.docs": "Model docs",
    "lang.label": "Choose language",
    "hero.copy": "Check that a vocational high school (SMK) uniform is complete, tie and hat included, live with your camera or from a photo. Detection runs entirely in this browser, so your images never leave your device.",
    "hero.art": "Illustration of four SMK students: two boys and two girls, one wearing a hijab",
    "attr.hat": "Hat",
    "attr.tie": "Tie",
    "attr.found": "Detected",
    "attr.missing": "Not detected",
    "attr.unknown": "—",
    "verdict.idle.title": "No check yet",
    "verdict.idle.desc": "Load a model, then start your camera or choose a photo to check the tie and hat.",
    "verdict.none.title": "No attributes detected",
    "verdict.none.desc": "Neither a tie nor a hat is visible. Make sure the student is clearly in the frame.",
    "verdict.complete.title": "Attributes complete",
    "verdict.complete.desc": "Both the tie and the hat are worn.",
    "verdict.missingHat.title": "Attributes incomplete",
    "verdict.missingHat.desc": "A tie was detected, but the hat is missing.",
    "verdict.missingTie.title": "Attributes incomplete",
    "verdict.missingTie.desc": "A hat was detected, but the tie is missing.",
    "live.title": "Live check",
    "btn.clear": "Clear preview",
    "empty.title": "Nothing to check yet",
    "empty.desc": "Load a model, then start your camera or choose a photo.",
    "btn.startCamera": "Start camera",
    "btn.stopCamera": "Stop camera",
    "metric.group": "Detection metrics",
    "metric.detections": "Detections",
    "metric.detectionsFoot": "Items in the current frame",
    "metric.inference": "Inference time",
    "metric.inferenceFoot": "Model execution per frame",
    "metric.fps": "Frame rate",
    "metric.fpsFoot": "Approximate live speed",
    "model.title": "Model",
    "field.path": "Model path",
    "field.pathNote": "Put your ONNX file in the <code>models/</code> folder, or pick a local file below. <a class=\"inline-link\" href=\"https://huggingface.co/prithivMLmods/YOLO26-ONNX/resolve/main/yolo26n/yolo26n.onnx\" target=\"_blank\" rel=\"noreferrer\">Download YOLO26n ONNX ↗</a>",
    "field.otherModel": "Use a different model",
    "field.optional": "Optional",
    "field.chooseOnnx": "Choose an ONNX file",
    "field.localFile": "Local file, .onnx",
    "btn.loadModel": "Load model",
    "btn.reloadModel": "Reload model",
    "btn.loading": "Loading model…",
    "settings.title": "Settings",
    "field.confidence": "Confidence threshold",
    "range.low": "More detections",
    "range.high": "Higher certainty",
    "field.camera": "Camera",
    "cam.back": "Back camera (mobile)",
    "cam.front": "Front camera",
    "photo.title": "Check a photo",
    "photo.note": "Run one photo through the loaded model and review what it finds.",
    "btn.choosePhoto": "Choose photo",
    "results.title": "Results",
    "footer.note": "Detection runs locally in your browser. Model and runtime files load as needed.",
    "common.unknown": "unknown error",
    "pill.model.none": "Model not loaded",
    "pill.model.loading": "Loading model",
    "pill.model.ready": "Model ready",
    "pill.model.error": "Model error",
    "pill.device": "Device {device}",
    "pill.device.none": "Device —",
    "mode.standby": "Standby",
    "mode.live": "Live camera",
    "mode.camError": "Camera error",
    "mode.photo": "Photo check",
    "hint.default": "Your media stays in this browser session.",
    "hint.modelLoaded": "Model loaded. Choose a camera or a photo to begin.",
    "hint.live": "Live frames are processed locally in this browser.",
    "hint.camOff": "Camera is off. Your model is ready for another input.",
    "hint.photo": "Local photo: {name}",
    "status.waiting": "Waiting for model initialization",
    "status.loadingModel": "Loading model and initializing browser runtime…",
    "status.modelFailed": "Model load failed: {msg}",
    "status.modelFailedGeneric": "Model load failed. Check the model file and try again.",
    "status.modelReady": "Model ready. Start a camera session or check a photo.",
    "status.waitInput": "Model ready. Waiting for input.",
    "status.loadFirst": "Load a model before starting the camera.",
    "status.noCamera": "Camera access is unavailable. Use HTTPS or localhost, or check a photo instead.",
    "status.requestPerm": "Requesting camera permission…",
    "status.cameraActive": "Camera active. Checking attributes…",
    "status.permDenied": "Camera permission was denied. Allow camera access or check a photo.",
    "status.cameraFail": "Could not start the camera: {msg}",
    "status.cameraStopped": "Camera stopped. Model remains loaded.",
    "status.inferStopped": "Detection stopped: {msg}",
    "status.badImage": "Please select a supported image file.",
    "status.loadingImage": "Loading photo…",
    "status.imageDone": "Photo check complete.",
    "status.imageFail": "Photo check failed: {msg}",
    "status.imageDecode": "The selected photo could not be decoded by this browser.",
    "list.empty": "Predictions will appear here.",
    "list.waitFirst": "Waiting for the first prediction…",
    "list.analyzing": "Analyzing photo…",
    "list.imageFailed": "Photo check failed.",
    "list.noPred": "No predictions above the current confidence threshold.",
    "det.class": "Class {cls}",
    "det.classTrack": "Class {cls}, track {id}",
    "det.top": "Top class {cls}",
  },
};

const LANG_STORAGE_KEY = "uniformcheck-lang";

function loadLang() {
  try {
    const saved = localStorage.getItem(LANG_STORAGE_KEY);
    if (saved === "id" || saved === "en") return saved;
  } catch { /* storage can be unavailable */ }
  return CONFIG.defaultLang;
}

let lang = loadLang();

function t(key, params = {}) {
  const text = I18N[lang]?.[key] ?? I18N.en[key] ?? key;
  return text.replace(/\{(\w+)\}/g, (_, name) => String(params[name] ?? ""));
}

/* ---------- Elements and state ---------- */

const $ = (selector) => document.querySelector(selector);

const elements = {
  video: $("#webcam"),
  image: $("#image-preview"),
  canvas: $("#overlay"),
  stage: $("#preview-stage"),
  emptyState: $("#empty-state"),
  statusIndicator: $("#status-indicator"),
  statusText: $("#status-text"),
  modelState: $("#model-state"),
  deviceState: $("#device-state"),
  previewMode: $("#preview-mode"),
  previewHint: $("#preview-hint-text"),
  modelPath: $("#model-path"),
  modelFile: $("#model-file"),
  modelFileLabel: $("#model-file-label"),
  loadModel: $("#load-model"),
  loadModelLabel: $("#load-model-label"),
  startCamera: $("#start-camera"),
  stopCamera: $("#stop-camera"),
  cameraFacing: $("#camera-facing"),
  chooseImage: $("#choose-image"),
  imageFile: $("#image-file"),
  clearPreview: $("#clear-preview"),
  confidence: $("#confidence"),
  confidenceValue: $("#confidence-value"),
  detectionCount: $("#detection-count"),
  detectionBadge: $("#detection-badge"),
  inferenceTime: $("#inference-time"),
  fpsValue: $("#fps-value"),
  detectionList: $("#detection-list"),
  verdict: $("#verdict"),
  verdictIcon: $("#verdict-icon"),
  verdictTitle: $("#verdict-title"),
  verdictDesc: $("#verdict-desc"),
  attrs: {
    tie: { card: $("#attr-tie"), state: $("#attr-tie-state") },
    hat: { card: $("#attr-hat"), state: $("#attr-hat-state") },
  },
  langButtons: document.querySelectorAll(".lang-btn"),
};

const state = {
  model: null,
  stream: null,
  running: false,
  frameRequest: 0,
  imageUrl: null,
  activeSource: null,
  lastFrameAt: 0,
  isLoading: false,
  // Everything below is re-rendered when the language changes.
  status: { key: "status.waiting", type: "idle", params: {} },
  modelPillKey: "pill.model.none",
  device: null,
  mode: { key: "mode.standby", live: false },
  hint: { key: "hint.default", params: {} },
  loadLabelKey: "btn.loadModel",
  modelFileName: null,
  list: { kind: "message", key: "list.empty" },
  lastResults: null,
  // Uniform check
  seenAt: { tie: -Infinity, hat: -Infinity },
  attrs: { tie: null, hat: null }, // null = not checked yet
  verdict: "idle",
};

/* ---------- Rendering helpers ---------- */

function renderStatus() {
  const { key, type, params } = state.status;
  elements.statusText.textContent = t(key, params);
  elements.statusIndicator.className = "status-indicator";
  if (type === "ready" || type === "running") elements.statusIndicator.classList.add("is-ready");
  if (type === "loading") elements.statusIndicator.classList.add("is-loading");
  if (type === "error") elements.statusIndicator.classList.add("is-error");
}

function setStatus(key, type = "idle", params = {}) {
  state.status = { key, type, params };
  renderStatus();
}

function renderPills() {
  elements.modelState.textContent = t(state.modelPillKey);
  elements.deviceState.textContent = state.device
    ? t("pill.device", { device: state.device })
    : t("pill.device.none");
}

function setModelPill(key) {
  state.modelPillKey = key;
  renderPills();
}

function renderPreviewMode() {
  elements.previewMode.replaceChildren(document.createElement("span"), document.createTextNode(` ${t(state.mode.key)}`));
  elements.previewMode.classList.toggle("is-live", state.mode.live);
}

function setPreviewMode(key, live = false) {
  state.mode = { key, live };
  renderPreviewMode();
}

function setHint(key, params = {}) {
  state.hint = { key, params };
  elements.previewHint.textContent = t(key, params);
}

function renderLoadLabel() {
  elements.loadModelLabel.textContent = t(state.loadLabelKey);
}

function setLoadLabel(key) {
  state.loadLabelKey = key;
  renderLoadLabel();
}

function renderModelFileLabel() {
  elements.modelFileLabel.textContent = state.modelFileName || t("field.chooseOnnx");
}

function setStageAspect(width, height) {
  if (width > 0 && height > 0) {
    elements.stage.style.aspectRatio = `${width} / ${height}`;
  }
}

function showSource(sourceType) {
  state.activeSource = sourceType;
  elements.video.classList.toggle("is-visible", sourceType === "camera");
  elements.image.classList.toggle("is-visible", sourceType === "image");
  elements.canvas.classList.toggle("is-visible", sourceType !== null);
  elements.emptyState.hidden = sourceType !== null;
  if (sourceType === null) {
    elements.canvas.width = 0;
    elements.canvas.height = 0;
    elements.stage.style.aspectRatio = "4 / 3";
  }
}

/* ---------- Detection list ---------- */

function makeEmptyRow(text, withIcon = false) {
  const empty = document.createElement("div");
  empty.className = "list-empty";
  if (withIcon) {
    const icon = document.createElement("span");
    icon.className = "list-empty-icon";
    icon.textContent = "⌁";
    empty.append(icon);
  }
  const label = document.createElement("span");
  label.textContent = text;
  empty.append(label);
  return empty;
}

function makeRow({ name, meta, score, color }) {
  const row = document.createElement("div");
  row.className = "detection-item";

  const swatch = document.createElement("span");
  swatch.className = "detection-swatch";
  if (typeof color === "string") swatch.style.setProperty("--swatch", color);

  const info = document.createElement("div");
  info.className = "detection-info";
  const nameEl = document.createElement("span");
  nameEl.className = "detection-name";
  nameEl.textContent = name;
  const metaEl = document.createElement("span");
  metaEl.className = "detection-meta";
  metaEl.textContent = meta;
  info.append(nameEl, metaEl);

  const scoreEl = document.createElement("span");
  scoreEl.className = "detection-score";
  scoreEl.textContent = score;
  row.append(swatch, info, scoreEl);
  return row;
}

function renderList() {
  elements.detectionList.replaceChildren();

  if (state.list.kind === "message") {
    elements.detectionList.append(makeEmptyRow(t(state.list.key), true));
    return;
  }

  const results = state.lastResults;
  const boxes = Array.isArray(results?.boxes) ? results.boxes : [];
  const classification = results?.probs;

  if (boxes.length) {
    const sorted = [...boxes].sort((a, b) => (b.conf ?? 0) - (a.conf ?? 0));
    for (const box of sorted) {
      const cls = box.cls ?? "—";
      elements.detectionList.append(makeRow({
        name: box.name || t("det.class", { cls: box.cls ?? "?" }),
        meta: box.id !== undefined ? t("det.classTrack", { cls, id: box.id }) : t("det.class", { cls }),
        score: Number.isFinite(box.conf) ? `${(box.conf * 100).toFixed(1)}%` : "—",
        color: box.color,
      }));
    }
    return;
  }

  if (classification?.name) {
    elements.detectionList.append(makeRow({
      name: classification.name,
      meta: t("det.top", { cls: classification.top1 ?? "—" }),
      score: Number.isFinite(classification.top1conf) ? `${(classification.top1conf * 100).toFixed(1)}%` : "—",
    }));
    return;
  }

  elements.detectionList.append(makeEmptyRow(t("list.noPred")));
}

/* ---------- Uniform logic ---------- */

function classifyAttribute(name) {
  if (typeof name !== "string") return null;
  const words = name.toLowerCase().split(/[^a-z]+/).filter(Boolean);
  if (words.some((word) => CONFIG.negationWords.includes(word))) return null;
  for (const [attribute, keywords] of Object.entries(CONFIG.attributeClasses)) {
    if (words.some((word) => keywords.includes(word))) return attribute;
  }
  return null;
}

function findAttributes(results) {
  const found = { tie: false, hat: false };
  const boxes = Array.isArray(results?.boxes) ? results.boxes : [];
  for (const box of boxes) {
    const attribute = classifyAttribute(box.name);
    if (attribute) found[attribute] = true;
  }
  return found;
}

// Branching: both worn -> complete; only one worn -> incomplete; none -> not detected.
function decideVerdict({ tie, hat }) {
  if (tie && hat) return "complete";
  if (tie) return "missingHat";
  if (hat) return "missingTie";
  return "none";
}

function renderVerdict() {
  const verdict = state.verdict;
  const panelState = verdict === "complete" ? "complete"
    : verdict === "missingHat" || verdict === "missingTie" ? "partial"
    : "idle";

  // Text is only rewritten when something changed, so screen readers are not flooded on every frame.
  const nextTitle = t(`verdict.${verdict}.title`);
  const nextDesc = t(`verdict.${verdict}.desc`);
  if (elements.verdictTitle.textContent !== nextTitle) elements.verdictTitle.textContent = nextTitle;
  if (elements.verdictDesc.textContent !== nextDesc) elements.verdictDesc.textContent = nextDesc;

  elements.verdict.dataset.state = panelState;
  elements.verdictIcon.textContent = verdict === "complete" ? "✓" : panelState === "partial" ? "!" : verdict === "none" ? "?" : "…";

  for (const [attribute, ui] of Object.entries(elements.attrs)) {
    const value = state.attrs[attribute];
    ui.card.dataset.found = value === null ? "unknown" : String(value);
    ui.state.textContent = value === null ? t("attr.unknown") : value ? t("attr.found") : t("attr.missing");
  }
}

function resetUniform() {
  state.seenAt = { tie: -Infinity, hat: -Infinity };
  state.attrs = { tie: null, hat: null };
  state.verdict = "idle";
  renderVerdict();
}

function evaluateUniform(results, now) {
  const found = findAttributes(results);
  const hold = state.activeSource === "camera" ? CONFIG.attributeHoldMs : 0;
  const present = {};
  for (const attribute of ["tie", "hat"]) {
    if (found[attribute]) state.seenAt[attribute] = now;
    present[attribute] = now - state.seenAt[attribute] <= hold;
  }
  state.attrs = present;
  state.verdict = decideVerdict(present);
  renderVerdict();
}

/* ---------- Detections ---------- */

function clearDetections(messageKey = "list.empty") {
  elements.detectionCount.textContent = "—";
  elements.detectionBadge.textContent = "0";
  elements.inferenceTime.textContent = "—";
  elements.fpsValue.textContent = "—";
  state.lastResults = null;
  state.list = { kind: "message", key: messageKey };
  renderList();
  resetUniform();
}

function updateDetections(results, frameTime = null) {
  const boxes = Array.isArray(results?.boxes) ? results.boxes : [];
  const classification = results?.probs;
  const count = boxes.length || (classification?.name ? 1 : 0);

  elements.detectionCount.textContent = String(count);
  elements.detectionBadge.textContent = String(count);

  const inference = results?.speed?.inference;
  elements.inferenceTime.textContent = Number.isFinite(inference) ? inference.toFixed(1) : "—";

  if (frameTime !== null && state.lastFrameAt > 0) {
    const elapsed = frameTime - state.lastFrameAt;
    if (elapsed > 0) elements.fpsValue.textContent = (1000 / elapsed).toFixed(1);
  }
  if (frameTime !== null) state.lastFrameAt = frameTime;

  state.lastResults = results;
  state.list = { kind: "results" };
  renderList();
  evaluateUniform(results, performance.now());
}

function currentPredictionOptions() {
  return { conf: Number(elements.confidence.value) };
}

/* ---------- Model ---------- */

async function loadModel() {
  if (state.isLoading) return;

  state.isLoading = true;
  elements.loadModel.disabled = true;
  setLoadLabel("btn.loading");
  setStatus("status.loadingModel", "loading");
  setModelPill("pill.model.loading");

  try {
    const selectedFile = elements.modelFile.files?.[0];
    const typedPath = elements.modelPath.value.trim() || CONFIG.defaultModelPath;
    const modelSource = selectedFile || new URL(typedPath.replace(/^\.\//, ""), import.meta.url).href;

    // Automatic device selection prefers WebGPU when available and falls back to CPU/WASM.
    const loadedModel = await YOLO.load(modelSource, { device: "auto" });
    state.model = loadedModel;
    state.device = String(loadedModel.device || "auto").toUpperCase();

    setModelPill("pill.model.ready");
    elements.startCamera.disabled = false;
    elements.chooseImage.disabled = false;
    setStatus("status.modelReady", "ready");
    setHint("hint.modelLoaded");
  } catch (error) {
    console.error("Model load failed:", error);
    setModelPill("pill.model.error");
    setStatus(error?.message ? "status.modelFailed" : "status.modelFailedGeneric", "error", { msg: error?.message });
  } finally {
    state.isLoading = false;
    elements.loadModel.disabled = false;
    setLoadLabel(state.model ? "btn.reloadModel" : "btn.loadModel");
  }
}

/* ---------- Camera ---------- */

function stopCamera({ keepPreview = false } = {}) {
  state.running = false;
  if (state.frameRequest) cancelAnimationFrame(state.frameRequest);
  state.frameRequest = 0;

  if (state.stream) {
    for (const track of state.stream.getTracks()) track.stop();
    state.stream = null;
  }
  elements.video.pause();
  elements.video.srcObject = null;
  elements.stopCamera.disabled = true;
  elements.startCamera.disabled = !state.model;

  if (!keepPreview && state.activeSource === "camera") {
    showSource(null);
    setPreviewMode("mode.standby");
    resetUniform();
  }
}

async function startCamera() {
  if (!state.model) {
    setStatus("status.loadFirst", "error");
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("status.noCamera", "error");
    return;
  }

  stopCamera({ keepPreview: true });
  if (state.imageUrl) {
    URL.revokeObjectURL(state.imageUrl);
    state.imageUrl = null;
  }

  try {
    setStatus("status.requestPerm", "loading");
    const facingMode = elements.cameraFacing.value;
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });

    elements.video.srcObject = state.stream;
    await elements.video.play();
    setStageAspect(elements.video.videoWidth, elements.video.videoHeight);
    showSource("camera");
    state.running = true;
    state.lastFrameAt = 0;
    elements.stopCamera.disabled = false;
    elements.startCamera.disabled = true;
    setHint("hint.live");
    setPreviewMode("mode.live", true);
    setStatus("status.cameraActive", "running");
    clearDetections("list.waitFirst");
    runCameraLoop();
  } catch (error) {
    console.error("Camera initialization failed:", error);
    stopCamera({ keepPreview: true });
    if (error?.name === "NotAllowedError") {
      setStatus("status.permDenied", "error");
    } else {
      setStatus("status.cameraFail", "error", { msg: error?.message || t("common.unknown") });
    }
    setPreviewMode("mode.camError");
  }
}

async function runCameraLoop() {
  if (!state.running || !state.model) return;

  try {
    const results = await state.model.predict(elements.video, currentPredictionOptions());
    await annotate(elements.canvas, elements.video, results, CONFIG.annotation);
    updateDetections(results, performance.now());
  } catch (error) {
    console.error("Frame inference failed:", error);
    stopCamera();
    setStatus("status.inferStopped", "error", { msg: error?.message || t("common.unknown") });
    return;
  }

  if (state.running) state.frameRequest = requestAnimationFrame(runCameraLoop);
}

function clearPreview() {
  stopCamera({ keepPreview: true });
  if (state.imageUrl) {
    URL.revokeObjectURL(state.imageUrl);
    state.imageUrl = null;
  }
  elements.image.removeAttribute("src");
  showSource(null);
  setPreviewMode("mode.standby");
  setHint(state.model ? "hint.modelLoaded" : "hint.default");
  clearDetections();
  setStatus(state.model ? "status.waitInput" : "status.waiting", state.model ? "ready" : "idle");
}

/* ---------- Photo ---------- */

async function analyzeImage(file) {
  if (!file || !state.model) return;
  if (!file.type.startsWith("image/")) {
    setStatus("status.badImage", "error");
    return;
  }

  stopCamera({ keepPreview: true });
  if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
  state.imageUrl = URL.createObjectURL(file);
  clearDetections("list.analyzing");
  setStatus("status.loadingImage", "loading");
  setPreviewMode("mode.photo");
  setHint("hint.photo", { name: file.name });

  const image = new Image();
  image.decoding = "async";
  image.onload = async () => {
    try {
      elements.image.src = state.imageUrl;
      setStageAspect(image.naturalWidth, image.naturalHeight);
      showSource("image");
      const results = await state.model.predict(image, currentPredictionOptions());
      await annotate(elements.canvas, image, results, CONFIG.annotation);
      updateDetections(results);
      setStatus("status.imageDone", "ready");
    } catch (error) {
      console.error("Image inference failed:", error);
      setStatus("status.imageFail", "error", { msg: error?.message || t("common.unknown") });
      clearDetections("list.imageFailed");
    }
  };
  image.onerror = () => {
    setStatus("status.imageDecode", "error");
  };
  image.src = state.imageUrl;
}

/* ---------- Language ---------- */

function applyLanguage() {
  document.documentElement.lang = lang;
  document.title = t("meta.title");
  document.querySelector('meta[name="description"]')?.setAttribute("content", t("meta.desc"));

  for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
  for (const el of document.querySelectorAll("[data-i18n-html]")) el.innerHTML = t(el.dataset.i18nHtml);
  for (const el of document.querySelectorAll("[data-i18n-title]")) el.title = t(el.dataset.i18nTitle);
  for (const el of document.querySelectorAll("[data-i18n-aria]")) el.setAttribute("aria-label", t(el.dataset.i18nAria));
  for (const button of elements.langButtons) {
    button.setAttribute("aria-pressed", String(button.dataset.lang === lang));
  }

  // Text that JavaScript sets at runtime.
  renderStatus();
  renderPills();
  renderPreviewMode();
  elements.previewHint.textContent = t(state.hint.key, state.hint.params);
  renderLoadLabel();
  renderModelFileLabel();
  renderList();
  renderVerdict();
}

function setLanguage(next) {
  if (next !== "id" && next !== "en") return;
  lang = next;
  try { localStorage.setItem(LANG_STORAGE_KEY, next); } catch { /* ignore */ }
  applyLanguage();
}

/* ---------- UI events ---------- */

elements.loadModel.addEventListener("click", loadModel);
elements.startCamera.addEventListener("click", startCamera);
elements.stopCamera.addEventListener("click", () => {
  stopCamera();
  setStatus("status.cameraStopped", "ready");
  setHint("hint.camOff");
});
elements.clearPreview.addEventListener("click", clearPreview);
elements.chooseImage.addEventListener("click", () => elements.imageFile.click());
elements.imageFile.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) analyzeImage(file);
  event.target.value = "";
});
elements.modelFile.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  state.modelFileName = file ? file.name : null;
  renderModelFileLabel();
});
elements.confidence.addEventListener("input", () => {
  elements.confidenceValue.value = Number(elements.confidence.value).toFixed(2);
  elements.confidenceValue.textContent = Number(elements.confidence.value).toFixed(2);
});
elements.cameraFacing.addEventListener("change", () => {
  if (state.running) startCamera();
});
for (const button of elements.langButtons) {
  button.addEventListener("click", () => setLanguage(button.dataset.lang));
}

// Initialize the confidence readout, language and neutral empty state.
elements.confidence.value = String(CONFIG.defaultConfidence);
elements.confidenceValue.value = CONFIG.defaultConfidence.toFixed(2);
elements.confidenceValue.textContent = CONFIG.defaultConfidence.toFixed(2);
setHint("hint.default");
clearDetections();
applyLanguage();
