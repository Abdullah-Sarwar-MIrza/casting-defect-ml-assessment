"""Vendor a fixed ONNX Runtime Web release for self-contained public hosting."""
import urllib.request
from pathlib import Path

VERSION = "1.22.0"
root = Path("docs/vendor")
root.mkdir(parents=True, exist_ok=True)
for name in ["ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]:
    url = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{VERSION}/dist/{name}"
    with urllib.request.urlopen(url, timeout=120) as response:
        (root/name).write_bytes(response.read())
    print(name, (root/name).stat().st_size)
with urllib.request.urlopen(f"https://raw.githubusercontent.com/microsoft/onnxruntime/v{VERSION}/LICENSE", timeout=30) as response:
    (root/"LICENSE").write_bytes(response.read())
(root/"VERSION.txt").write_text(VERSION + "\nSource: https://www.npmjs.com/package/onnxruntime-web\n")

with urllib.request.urlopen(f"https://raw.githubusercontent.com/microsoft/onnxruntime/v{VERSION}/ThirdPartyNotices.txt", timeout=30) as response:
    (root/"ThirdPartyNotices.txt").write_bytes(response.read())
