# Deployment verification

Public URL: https://abdullah-sarwar-mirza.github.io/casting-defect-ml-assessment/

Hosting: GitHub Pages, main branch `/docs`. Inference: real trained ONNX model in the browser using vendored ONNX Runtime Web 1.22.0. The image-upload interface is the publicly deployed inference path; the REST API is an additional containerized deployment option.

Verification is recorded in `live_browser_verification.json`, including the verification time, live URL, predictions for both sample classes, Python/browser score differences, successful file upload, invalid-file rejection, and browser request/error checks. `live_browser.png` captures the working public interface. Hosting should be checked again at actual submission time.

Container build and API evidence are recorded in `container_verification.json`. The live public deployment does not depend on a local server or temporary tunnel.

## Verified outcome

The public deployment was exercised on 2026-10-16T17:34:38Z. Both example classes were correct, the actual image upload control worked, and an invalid image was rejected. There were zero browser page errors and zero failed asset requests. The largest example score difference from Python was 0.00001711. GitHub Pages deployment and automated tests completed successfully. The Docker build, health route, both class predictions, and non-root UID were independently verified.
