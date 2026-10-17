"use strict";
// All model assets are same-origin. The uploaded image is never sent to a server.
let sessionPromise;
let metadata;
let baselinePromise;
let lastBlob;
const modelSelect = document.getElementById("model");
const status = document.getElementById("status");
const controls = [...document.querySelectorAll("button,input,select")];
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("vendor/", window.location.href).href;

async function getSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      status.textContent = "Loading the trained model. First use may take a moment...";
      const response = await fetch("metadata.json");
      if (!response.ok) throw new Error("Could not load model metadata");
      metadata = await response.json();
      return ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    })().catch(error => { sessionPromise = undefined; throw error; });
  }
  return sessionPromise;
}

async function getBaseline() {
  if (!baselinePromise) {
    baselinePromise = fetch("baseline.json").then(response => {
      if (!response.ok) throw new Error("Could not load baseline parameters");
      return response.json();
    }).catch(error => { baselinePromise = undefined; throw error; });
  }
  return baselinePromise;
}

function baselineProbability(pixels, width, height, model) {
  // Pillow-compatible 8-bit RGB to grayscale before antialiased resizing.
  const gray = new Uint8ClampedArray(pixels.length);
  for (let i = 0; i < pixels.length; i += 4) {
    const value = (19595 * pixels[i] + 38470 * pixels[i+1] + 7471 * pixels[i+2] + 32768) >> 16;
    gray[i] = gray[i+1] = gray[i+2] = value;
    gray[i+3] = 255;
  }
  const values = resizeTensor(gray, width, height, model.image_size);
  let logit = model.intercept;
  for (let i = 0; i < model.coef.length; ++i) {
    const centered = Math.fround(values[i] - model.mean[i]);
    const scaled = Math.fround(centered / model.scale[i]);
    logit += scaled * model.coef[i];
  }
  return logit >= 0 ? 1 / (1 + Math.exp(-logit)) : Math.exp(logit) / (1 + Math.exp(logit));
}

// Separable, antialiased bilinear resize following Pillow's pixel-center convention.
// Rounding the intermediate horizontal pass mirrors the 8-bit Python pipeline.
function filterWeights(inputSize, outputSize) {
  const scale = inputSize / outputSize, support = Math.max(1, scale);
  return Array.from({length: outputSize}, (_, out) => {
    const center = (out + 0.5) * scale;
    const start = Math.max(0, Math.floor(center - support + 0.5));
    const end = Math.min(inputSize, Math.floor(center + support + 0.5));
    const weights = [];
    let sum = 0;
    for (let i = start; i < end; ++i) {
      const value = Math.max(0, 1 - Math.abs((i + 0.5 - center) / support));
      weights.push(value); sum += value;
    }
    return {start, weights: weights.map(w => w / sum)};
  });
}

function resizeTensor(pixels, width, height, size) {
  const horizontal = filterWeights(width, size), vertical = filterWeights(height, size);
  const intermediate = new Uint8Array(height * size * 3);
  for (let y = 0; y < height; ++y) for (let x = 0; x < size; ++x) {
    const filter = horizontal[x];
    for (let c = 0; c < 3; ++c) {
      let value = 0;
      for (let k = 0; k < filter.weights.length; ++k)
        value += pixels[(y * width + filter.start + k) * 4 + c] * filter.weights[k];
      intermediate[(y * size + x) * 3 + c] = Math.round(value);
    }
  }
  const tensor = new Float32Array(3 * size * size);
  for (let y = 0; y < size; ++y) for (let x = 0; x < size; ++x) {
    const filter = vertical[y];
    for (let c = 0; c < 3; ++c) {
      let value = 0;
      for (let k = 0; k < filter.weights.length; ++k)
        value += intermediate[((filter.start + k) * size + x) * 3 + c] * filter.weights[k];
      tensor[c * size * size + y * size + x] = Math.round(value) / 255;
    }
  }
  return tensor;
}

async function inspect(blob) {
  if (!blob || !["image/jpeg", "image/png"].includes(blob.type)) throw new Error("Choose a JPEG or PNG image.");
  if (blob.size > 10 * 1024 * 1024) throw new Error("Image exceeds the 10 MiB limit.");
  const modelId = modelSelect.value;
  controls.forEach(control => control.disabled = true);
  window.lastPrediction = null;
  document.getElementById("prediction").textContent = "Inspecting image...";
  document.getElementById("probability").textContent = "";
  document.getElementById("details").textContent = "";
  try {
    const bitmap = await createImageBitmap(blob, {imageOrientation: "from-image", premultiplyAlpha: "none", colorSpaceConversion: "none"});
    if (bitmap.width * bitmap.height > 20000000) { bitmap.close(); throw new Error("Image exceeds the 20 megapixel limit."); }
    const preview = document.getElementById("preview");
    if (preview.dataset.objectUrl) URL.revokeObjectURL(preview.dataset.objectUrl);
    preview.dataset.objectUrl = URL.createObjectURL(blob); preview.src = preview.dataset.objectUrl; preview.hidden = false;
    const canvas = document.createElement("canvas"); canvas.width = bitmap.width; canvas.height = bitmap.height;
    const context = canvas.getContext("2d", {willReadFrequently: true}); context.drawImage(bitmap, 0, 0); bitmap.close();
    lastBlob = blob;
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let probability, threshold, started;
    if (modelId === "pixels_logistic") {
      const model = await getBaseline();
      status.textContent = "Running baseline inference...";
      started = performance.now();
      probability = baselineProbability(pixels, canvas.width, canvas.height, model);
      threshold = model.threshold;
    } else {
      const session = await getSession();
      status.textContent = "Running ResNet inference...";
      const size = metadata.image_size;
      const data = resizeTensor(pixels, canvas.width, canvas.height, size);
      started = performance.now();
      const output = await session.run({image: new ort.Tensor("float32", data, [1, 3, size, size])});
      probability = output.defect_probability.data[0];
      threshold = metadata.threshold;
    }
    const defective = probability >= threshold;
    const title = document.getElementById("prediction");
    title.textContent = defective ? "Defective casting" : "Acceptable casting";
    title.dataset.defective = String(defective);
    document.getElementById("probability").textContent = `Defect score: ${(probability * 100).toFixed(2)}%`;
    document.getElementById("details").textContent = `${modelSelect.options[modelSelect.selectedIndex].text} | Decision threshold: ${threshold.toFixed(2)} | Inference: ${(performance.now() - started).toFixed(0)} ms`;
    window.lastPrediction = {probability, defective, threshold, model: modelId};
    status.textContent = "Prediction complete. Your image remained on this device.";
  } finally { controls.forEach(control => control.disabled = false); }
}
function showError(error) {
  console.error(error);
  window.lastPrediction = null;
  document.getElementById("prediction").textContent = "Prediction unavailable";
  document.getElementById("probability").textContent = "";
  document.getElementById("details").textContent = "";
  status.textContent = `Unable to inspect image: ${error.message}`;
}
document.getElementById("image").addEventListener("change", event => {
  if (event.target.files[0]) inspect(event.target.files[0]).catch(showError);
});
modelSelect.addEventListener("change", () => {
  document.getElementById("model-info").textContent = modelSelect.value === "pixels_logistic"
    ? "Pixel baseline: 85.73% held-out accuracy; decision threshold 25%. More false alarms are expected."
    : "ResNet-18: 99.30% held-out accuracy; decision threshold 50%.";
  if (lastBlob) inspect(lastBlob).catch(showError);
});
for (const label of ["ok", "defective"]) {
  document.getElementById(`example-${label}`).addEventListener("click", async () => {
    try {
      const response = await fetch(`example-${label}.png`);
      if (!response.ok) throw new Error("Could not load example image");
      await inspect(await response.blob());
    } catch (error) { showError(error); }
  });
}
