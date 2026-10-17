"""Real browser inference and Python/browser parity. May target a live URL."""
import argparse
import functools
import http.server
import json
import threading
import time
from pathlib import Path
from playwright.sync_api import sync_playwright
from casting_ml.inference import Predictor

parser=argparse.ArgumentParser(); parser.add_argument("--url"); parser.add_argument("--live",action="store_true")
args=parser.parse_args()
server=None
if args.url:
    url=args.url
else:
    handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory="docs")
    server=http.server.ThreadingHTTPServer(("127.0.0.1",0),handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    url=f"http://127.0.0.1:{server.server_port}/"
results=[]; uploads=[]; errors=[]; failed=[]; http_errors=[]
with sync_playwright() as play:
    try:
        browser=play.chromium.launch(channel="msedge",headless=True)
    except Exception:
        browser=play.chromium.launch(headless=True)
    page=browser.new_page(viewport={"width":1200,"height":900})
    page.on("pageerror",lambda error:errors.append(str(error)))
    page.on("requestfailed",lambda request:failed.append(dict(url=request.url,error=request.failure)))
    page.on("response",lambda response:http_errors.append(dict(url=response.url,status=response.status)) if response.status >= 400 else None)
    response=page.goto(url,wait_until="networkidle",timeout=120000)
    assert response.status==200
    predictor=Predictor()
    for name,target in [("ok",False),("defective",True)]:
        page.evaluate("window.lastPrediction = null")
        page.click(f"#example-{name}")
        page.wait_for_function("window.lastPrediction !== null && window.lastPrediction !== undefined",timeout=180000)
        result=page.evaluate("window.lastPrediction")
        reference=predictor.predict_bytes(Path(f"docs/example-{name}.png").read_bytes())["defect_probability"]
        result.update(example=name,python_probability=reference,absolute_difference=abs(reference-result["probability"]))
        assert result["defective"] is target,result
        assert result["absolute_difference"]<0.02,result
        results.append(result)
    # Exercise real file uploads for both classes, independently of example buttons.
    for name,target in [("ok",False),("defective",True)]:
        page.evaluate("window.lastPrediction = null")
        page.set_input_files("#image",f"docs/example-{name}.png")
        page.wait_for_function("window.lastPrediction !== null && window.lastPrediction !== undefined",timeout=60000)
        result=page.evaluate("window.lastPrediction")
        reference=predictor.predict_bytes(Path(f"docs/example-{name}.png").read_bytes())["defect_probability"]
        result.update(example=name,python_probability=reference,absolute_difference=abs(reference-result["probability"]))
        assert result["defective"] is target,result
        assert result["absolute_difference"]<0.02,result
        uploads.append(result)
    page.screenshot(path=f"reports/{'live' if args.live else 'local'}_browser.png",full_page=True)
    page.set_input_files("#image",{"name":"bad.png","mimeType":"image/png","buffer":b"bad image"})
    page.wait_for_function("document.getElementById('status').textContent.includes('Unable to inspect')",timeout=30000)
    browser.close()
assert not errors,errors
assert not failed,failed
assert not http_errors,http_errors
report=dict(url=url,examples=results,uploads=uploads,page_errors=errors,failed_requests=failed,http_errors=http_errors,
            upload_test="passed",invalid_upload_test="passed",browser="Microsoft Edge / Chromium",verified_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
Path(f"reports/{'live' if args.live else 'local'}_browser_verification.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
if server:server.shutdown()
