"""Take screenshots of the running demo page with Edge, driven over its debugging port (no extra packages).

    python capture.py <out_dir> [name ...]        names from SHOTS below; none means all

The demo page must be running (streamlit run app.py) with LM Studio serving the three models. Each shot opens the
page with query parameters that pick the section and the fixed question, waits until the answers are on screen,
then saves a PNG. Marathi shots use hide=1 (questions and measured table only), see TODO.md.
"""

import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websocket  # websocket-client

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
BASE = "http://localhost:8501/"
PORT = 9333
WIDTH, HEIGHT = 1600, 1250

# name -> (query string, text that appears when the answers are done)
SHOTS = {
    "frugal_demo_doc_status": ("?s=0&q=0", "Structure-aware agent", "Measured on 110"),
    "frugal_demo_doc_list": ("?s=0&q=1", "Structure-aware agent", "Measured on 110"),
    "frugal_demo_doc_subtopics": ("?s=0&q=2", "Structure-aware agent", "Measured on 110"),
    "frugal_demo_doc_numbers": ("?s=0&q=3", "Structure-aware agent", "Measured on 110"),
    "frugal_demo_doc_cross": ("?s=0&q=4", "Structure-aware agent", "Measured on 110"),
    "frugal_demo_doc_unanswerable": ("?s=0&q=5", "Structure-aware agent", "Measured on 110"),
    "frugal_demo_marathi_reworded": ("?s=1&m=0&hide=1", "Base model, no help", "Measured on 52"),
    "frugal_demo_marathi_raw": ("?s=1&m=2&hide=1", "Base model, no help", "Measured on 52"),
    "frugal_demo_marathi_control": ("?s=1&m=4&hide=1", "Base model, no help", "Measured on 52"),
    "frugal_demo_router": ("?s=2&r=0", "Intent:", "Measured on 150"),
}


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

    def text(self) -> str:
        r = self.call("Runtime.evaluate", expression="document.body.innerText", returnByValue=True)
        return r.get("result", {}).get("value", "") or ""

    def height(self) -> int:
        r = self.call("Runtime.evaluate", returnByValue=True,
                      expression="Math.max(document.querySelector('.block-container').getBoundingClientRect().bottom, "
                                 "document.querySelector('[data-testid=stSidebarContent]').scrollHeight * 0 + 900)")
        return int(r.get("result", {}).get("value", HEIGHT))


def shoot(page: Page, name: str, query: str, done_text: str, tail_text: str, out: Path, timeout: int = 240) -> Path:
    page.call("Page.navigate", url=BASE + query)
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(1.5)
        txt = page.text()
        if "Running" in txt and "Stop" in txt:
            continue
        if done_text in txt and tail_text in txt:
            break
    else:
        raise RuntimeError("%s: the page never showed %r" % (name, tail_text))
    if "s=0" not in query:                          # the Marathi and router pages end with a results table: wait for it
        for _ in range(20):
            if page.call("Runtime.evaluate", expression="document.querySelectorAll('table').length", returnByValue=True)["result"]["value"]:
                break
            time.sleep(1)
    time.sleep(2.0)
    h = min(page.height() + 40, 2600)
    page.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=h, deviceScaleFactor=1, mobile=False)
    time.sleep(1.0)
    # Content area only (no sidebar, no page title): x 370 to 1560, from y 270 down, so the text stays readable on a slide.
    data = page.call("Page.captureScreenshot", format="png",
                     clip={"x": 370, "y": 270, "width": 1190, "height": h - 270 - 30, "scale": 1})["data"]
    path = out / (name + ".png")
    path.write_bytes(base64.b64decode(data))
    return path


def main(argv):
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    names = argv[1:] or list(SHOTS)
    profile = Path(__import__("tempfile").mkdtemp(prefix="edge_demo_"))
    proc = subprocess.Popen([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--remote-debugging-port=%d" % PORT,
                             "--user-data-dir=" + str(profile), "--window-size=%d,%d" % (WIDTH, HEIGHT), "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        page = Page(PORT)
        for name in names:
            q, done, tail = SHOTS[name]
            p = shoot(page, name, q, done, tail, out)
            print("saved %s (%d bytes)" % (p, p.stat().st_size), flush=True)
    finally:
        proc.terminate()


if __name__ == "__main__":
    main(sys.argv[1:])
