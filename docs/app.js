"use strict";
// All model assets are same-origin. The uploaded image is never sent to a server.
let sessionPromise;
let metadata;
const status = document.getElementById("status");
const controls = [...document.querySelectorAll("button,input")];
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("vendor/", window.location.href).href;

async function getSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      status.textContent = "Loading the trained model. First use may take a moment?";
      const response = await fetch("metadata.json");
      if (!response.ok) throw new Error("Could not load model metadata");
      metadata = await response.json();
      return ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    })().catch(error => { sessionPromise = undefined; throw error; });
  }
  return sessionPromise;
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
  controls.forEach(control => control.disabled = true);
  try {
    const bitmap = await createImageBitmap(blob, {imageOrientation: "from-image", premultiplyAlpha: "none", colorSpaceConversion: "none"});
    if (bitmap.width * bitmap.height > 20000000) { bitmap.close(); throw new Error("Image exceeds the 20 megapixel limit."); }
    const preview = document.getElementById("preview");
    if (preview.dataset.objectUrl) URL.revokeObjectURL(preview.dataset.objectUrl);
    preview.dataset.objectUrl = URL.createObjectURL(blob); preview.src = preview.dataset.objectUrl; preview.hidden = false;
    const canvas = document.createElement("canvas"); canvas.width = bitmap.width; canvas.height = bitmap.height;
    const context = canvas.getContext("2d", {willReadFrequently: true}); context.drawImage(bitmap, 0, 0); bitmap.close();
    const session = await getSession();
    status.textContent = "Running inference?";
    const size = metadata.image_size;
    const data = resizeTensor(context.getImageData(0, 0, canvas.width, canvas.height).data, canvas.width, canvas.height, size);
    const started = performance.now();
    const output = await session.run({image: new ort.Tensor("float32", data, [1, 3, size, size])});
    const probability = output.defect_probability.data[0];
    const defective = probability >= metadata.threshold;
    const title = document.getElementById("prediction");
    title.textContent = defective ? "Defective casting" : "Acceptable casting";
    title.dataset.defective = String(defective);
    document.getElementById("probability").textContent = `Defect score: ${(probability * 100).toFixed(2)}%`;
    document.getElementById("details").textContent = `Decision threshold: ${metadata.threshold.toFixed(2)} ? Inference: ${(performance.now() - started).toFixed(0)} ms`;
    window.lastPrediction = {probability, defective, threshold: metadata.threshold};
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
document.getElementById("image").addEventListener("change", event => inspect(event.target.files[0]).catch(showError));
for (const label of ["ok", "defective"]) {
  document.getElementById(`example-${label}`).addEventListener("click", async () => {
    try {
      const response = await fetch(`example-${label}.png`);
      if (!response.ok) throw new Error("Could not load example image");
      await inspect(await response.blob());
    } catch (error) { showError(error); }
  });
}
