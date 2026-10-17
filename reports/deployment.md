# Deployment verification

Public URL: https://abdullah-sarwar-mirza.github.io/casting-defect-ml-assessment/

Hosting: GitHub Pages, main branch `/docs`. Inference: real trained ONNX model in the browser using vendored ONNX Runtime Web 1.22.0. The image-upload interface is the publicly deployed inference path; the REST API is an additional containerized deployment option.

Verification is recorded in `live_browser_verification.json`, including the verification time, live URL, predictions for both sample classes, Python/browser score differences, successful file upload, invalid-file rejection, and browser request/error checks. `live_browser.png` captures the working public interface. Hosting should be checked again at actual submission time.

Container build and API evidence are recorded in `container_verification.json`. The live public deployment does not depend on a local server or temporary tunnel.

## Verified outcome

Last live browser check: 2026-10-17T07:53:53Z. Both normal and defective examples were correct through the example buttons and actual file uploads. Invalid image data were rejected. There were zero browser page errors, failed requests, or HTTP error responses. The largest example score difference from Python was 0.00001711. The public ONNX file (44,700,088 bytes) has the same SHA-256 as the repository model: `5de664c751ab50eed0fdc1e829f983eaa2a871483daae4c2abf4ea1448daf9d4`.

See `verification.json` for the current review and `evaluation_verification.json` for the unchanged artifacts and full test-set replay. The Docker evidence was recorded on 16 October 2026; its build, health route, both class predictions, and non-root UID were verified then, not rerun in this review. The current public path is browser inference, not the optional REST container.
