const API_BASE = "http://localhost:8000";

const dropzone = document.getElementById("dropzone");
const dropzoneContent = document.getElementById("dropzoneContent");
const dropzoneFile = document.getElementById("dropzoneFile");
const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");
const browseBtn = document.getElementById("browseBtn");
const removeFile = document.getElementById("removeFile");
const submitBtn = document.getElementById("submitBtn");
const uploadSection = document.getElementById("uploadSection");
const status = document.getElementById("status");
const statusText = document.getElementById("statusText");
const spinner = document.getElementById("spinner");
const results = document.getElementById("results");
const resultsGrid = document.getElementById("resultsGrid");
const reportBlock = document.getElementById("reportBlock");
const warningEl = document.getElementById("warning");
const error = document.getElementById("error");
const errorMessage = document.getElementById("errorMessage");

let selectedFile = null;

function show(el, visible = true) {
  el.hidden = !visible;
}

function setStatus(visible, text = "Analyzing…") {
  show(status, visible);
  statusText.textContent = text;
}

function setError(visible, message = "") {
  show(error, visible);
  errorMessage.textContent = message;
}

function showResults(data) {
  show(uploadSection);
  show(status, false);
  setError(false);
  show(results, true);

  resultsGrid.innerHTML = "";
  const cards = [
    { label: "Body part", value: data.body_part, class: "" },
    { label: "Body part confidence", value: (data.body_part_confidence * 100).toFixed(1) + "%", class: "" },
    { label: "Fracture detected", value: data.fracture_detected ? "Yes" : "No", class: data.fracture_detected ? "fracture-yes" : "fracture-no" },
    { label: "Fracture confidence", value: (data.fracture_confidence * 100).toFixed(1) + "%", class: "" },
    { label: "Bone age (months)", value: data.bone_age_months != null ? String(data.bone_age_months) : "—", class: "" },
  ];
  cards.forEach(({ label, value, class: cls }) => {
    const card = document.createElement("div");
    card.className = "result-card" + (cls ? " " + cls : "");
    card.innerHTML = `<div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(String(value))}</div>`;
    resultsGrid.appendChild(card);
  });

  reportBlock.innerHTML = `<div class="label">Report</div>${escapeHtml(data.report || "—")}`;
  warningEl.textContent = data.warning || "";
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function onFileSelect(file) {
  if (!file) {
    selectedFile = null;
    submitBtn.disabled = true;
    show(dropzoneContent, true);
    show(dropzoneFile, false);
    return;
  }
  selectedFile = file;
  fileName.textContent = file.name;
  show(dropzoneContent, false);
  show(dropzoneFile, true);
  submitBtn.disabled = false;
  show(results, false);
  setError(false);
}

browseBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  onFileSelect(file || null);
});

removeFile.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.value = "";
  onFileSelect(null);
});

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) {
    fileInput.files = e.dataTransfer.files;
    onFileSelect(file);
  }
});

submitBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  show(uploadSection);
  show(results, false);
  setError(false);
  setStatus(true, "Analyzing…");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch(`${API_BASE}/v1/analyze/dicom`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errBody = await res.text();
      throw new Error(errBody || `Request failed: ${res.status}`);
    }

    const data = await res.json();
    show(uploadSection);
    setStatus(false);
    showResults(data);
  } catch (err) {
    setStatus(false);
    setError(true, err.message || "Request failed. Is the API running on port 8000?");
  }
});
