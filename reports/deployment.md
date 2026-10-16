# Deployment verification

Public URL: https://abdullah-sarwar-mirza.github.io/casting-defect-ml-assessment/

Hosting: GitHub Pages, main branch `/docs`. Inference: real trained ONNX model in the browser using vendored ONNX Runtime Web 1.22.0. The image-upload interface is the publicly deployed inference path; the REST API is an additional containerized deployment option.

Verification is recorded in `live_browser_verification.json`, including the verification time, live URL, predictions for both sample classes, Python/browser score differences, successful file upload, invalid-file rejection, and browser request/error checks. `live_browser.png` captures the working public interface. Hosting should be checked again at actual submission time.

Container build and API evidence are recorded in `container_verification.json`. The live public deployment does not depend on a local server or temporary tunnel.
