"""Take screenshots of the running demo page with Edge, driven over its debugging port (no extra packages).

    python capture.py <out_dir> [name ...]        names from SHOTS below; none means all

The demo page must be running (streamlit run app.py) with LM Studio serving Qwen3 1.7B and nomic-embed-text. Each shot
opens the page with ?q=<number> (one of the five fixed questions), waits until both answers and the score line are on
screen, then saves a PNG of the content area. The earlier three-section version is in _backup/.
"""

import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import websocket  # websocket-client

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
BASE = "http://localhost:8501/"
PORT = 9333
WIDTH, HEIGHT = 1500, 900

SHOTS = {"frugal_rag_q%d" % (i + 1): "?q=%d&bare=1" % i for i in range(5)}      # run with app.py

# run with app_faq.py (the author's real QCNP notes, so the pictures show real FAQ names: the author accepted that on 2026-10-28)
FAQ_QUESTIONS = (
    "Does a quantum computer try every answer at once?",
    "Can entanglement be used to send a message faster than light?",
    "Will quantum computers break all encryption soon?",
    "What is a surface code?",
    "Why is the no-cloning theorem important?",
)
SHOTS.update({"frugal_faq_q%d" % (i + 1): "?bare=1&q=" + urllib.parse.quote(q) for i, q in enumerate(FAQ_QUESTIONS)})


class Page:
    def __init__(self, port: int):
        deadline = time.time() + 20
        while True:
            try:
                targets = json.load(urllib.request.urlopen("http://localhost:%d/json" % port, timeout=2))
                page = next(t for t in targets if t["type"] == "page")
                break
            except Exception:
                if time.time() > deadline:
                    raise RuntimeError("Edge did not open its debugging port")
                time.sleep(0.5)
        self.ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=60, suppress_origin=True)
        self.n = 0

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                return msg.get("result", {})

    def value(self, expression):
        return self.call("Runtime.evaluate", expression=expression, returnByValue=True).get("result", {}).get("value")


def shoot(page: Page, name: str, query: str, out: Path, timeout: int = 240) -> Path:
    page.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=HEIGHT, deviceScaleFactor=1, mobile=False)  # full width before layout
    page.call("Page.navigate", url=BASE + query)
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(1.5)
        txt = page.value("document.body.innerText") or ""
        if "Running" in txt and "Stop" in txt:
            continue
        if ("Base model + RAG" in txt and "answered" in txt) or ("Where it already appears" in txt and "one run" in txt):
            break
    else:
        raise RuntimeError("%s: the page never showed both answers" % name)
    time.sleep(1.5)
    r = json.loads(page.value("JSON.stringify({b: document.querySelector('.block-container').getBoundingClientRect(), "
                              "q: document.querySelector('.q').getBoundingClientRect(), "
                              "n: document.querySelector('.note').getBoundingClientRect()})"))
    page.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=int(r["n"]["bottom"]) + 80, deviceScaleFactor=1, mobile=False)
    time.sleep(1.0)
    top = max(r["q"]["top"] - 20, 0)
    data = page.call("Page.captureScreenshot", format="png",
                     clip={"x": r["b"]["x"], "y": top, "width": r["b"]["width"], "height": r["n"]["bottom"] - top + 25, "scale": 1})["data"]
    path = out / (name + ".png")
    path.write_bytes(base64.b64decode(data))
    return path


def main(argv):
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    names = argv[1:] or list(SHOTS)
    profile = Path(tempfile.mkdtemp(prefix="edge_demo_"))
    proc = subprocess.Popen([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--remote-debugging-port=%d" % PORT,
                             "--user-data-dir=" + str(profile), "--window-size=%d,%d" % (WIDTH, HEIGHT), "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        page = Page(PORT)
        for name in names:
            p = shoot(page, name, SHOTS[name], out)
            print("saved %s (%d bytes)" % (p, p.stat().st_size), flush=True)
    finally:
        proc.terminate()


if __name__ == "__main__":
    main(sys.argv[1:])
