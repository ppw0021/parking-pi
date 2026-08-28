'''
GateWatcher calibration UI.

A small Flask web app (like spot-watcher's cal.py).  Point a browser at
http://<this-machine>:8000/ and you get the live camera view with two
draggable/resizable rectangles: one marks where to OCR text for the
ENTER lane, the other for the EXIT lane.  Tick "rotate 180" for a lane
whose plates face away from the camera.

Save -> writes regions.json next to this script:

    {
      "enter": {"box": [x, y, w, h], "rot180": false},
      "exit":  {"box": [x, y, w, h], "rot180": true}
    }

box is in native capture pixels.  main.py loads this file and, if it is
present, OCRs only those two crops instead of the whole frame.

Run only one of cal.py / main.py at a time - they both open the camera.

    cd gate-watcher/src && uv run cal.py

Env: GATE_CAMERA_INDEX (default 1), GATE_CAL_PORT (default 8000).
'''

import json
import os
import threading

import cv2
from flask import Flask, Response, jsonify, request

# Keep these in sync with main.py.
CAMERA_INDEX = int(os.environ.get("GATE_CAMERA_INDEX", "1"))
CAMERA_RESOLUTION = (1280, 720)
PORT = int(os.environ.get("GATE_CAL_PORT", "8000"))
REGIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regions.json")

app = Flask(__name__)
_cap = None
_cap_lock = threading.Lock()


def get_camera():
    """Open the camera once, trying the configured index then 0."""
    global _cap
    with _cap_lock:
        if _cap is not None and _cap.isOpened():
            return _cap
        for idx in dict.fromkeys([CAMERA_INDEX, 0]):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
                print(f"camera: opened index {idx}")
                _cap = cap
                return _cap
            cap.release()
        raise RuntimeError("cannot open camera (try GATE_CAMERA_INDEX=<n>)")


def read_frame():
    cap = get_camera()
    with _cap_lock:
        ok, frame = cap.read()
    return frame if ok else None


def load_regions():
    try:
        with open(REGIONS_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# --------------------------------------------------------------------------

PAGE = """<!doctype html>
<title>GateWatcher calibration</title>
<style>
  body { margin: 0; background: #1e1e1e; color: #eee;
         font: 14px/1.4 system-ui, sans-serif; }
  #bar { padding: 8px 12px; display: flex; gap: 8px; align-items: center;
         flex-wrap: wrap; background: #262626; border-bottom: 1px solid #333; }
  button { padding: 6px 10px; background: #333; color: #eee; border: 1px solid #555;
           border-radius: 4px; cursor: pointer; }
  button:hover { background: #3d3d3d; }
  button.enter { border-color: #3fb950; }
  button.exit  { border-color: #d29922; }
  label { display: flex; gap: 4px; align-items: center; }
  #save { border-color: #58a6ff; }
  #status { margin-left: auto; opacity: .8; }
  #wrap { position: relative; display: inline-block; margin: 12px; }
  #cam { display: block; max-width: calc(100vw - 24px); height: auto; }
  #ov { position: absolute; left: 0; top: 0; cursor: crosshair; }
</style>
<div id="bar">
  <button class="enter" data-draw="enter">Draw ENTER box</button>
  <button class="enter" data-clear="enter">Clear ENTER</button>
  <label><input type="checkbox" data-rot="enter"> ENTER rot180</label>
  <span style="width:12px"></span>
  <button class="exit" data-draw="exit">Draw EXIT box</button>
  <button class="exit" data-clear="exit">Clear EXIT</button>
  <label><input type="checkbox" data-rot="exit"> EXIT rot180</label>
  <span style="width:12px"></span>
  <button id="save">Save regions.json</button>
  <span id="status"></span>
</div>
<div id="wrap">
  <img id="cam" src="/stream">
  <canvas id="ov"></canvas>
</div>
<script>
const cam = document.getElementById('cam');
const ov = document.getElementById('ov');
const ctx = ov.getContext('2d');
const statusEl = document.getElementById('status');
const COLORS = { enter: '#3fb950', exit: '#d29922' };
let native = { w: 1280, h: 720 };
let regions = { enter: null, exit: null };   // {x,y,w,h,rot180} in native px
let draw = null;                             // side currently being drawn
let rubber = null;                           // {x0,y0,x1,y1} in native px
let drag = null;                             // {side, mode, handle, ...}

function setStatus(t) { statusEl.textContent = t; }

async function boot() {
  native = await (await fetch('/meta')).json();
  const saved = await (await fetch('/regions')).json();
  for (const side of ['enter', 'exit']) {
    const r = saved[side];
    if (r && r.box) {
      regions[side] = { x: r.box[0], y: r.box[1], w: r.box[2], h: r.box[3],
                        rot180: !!r.rot180 };
      document.querySelector(`[data-rot="${side}"]`).checked = !!r.rot180;
    }
  }
  requestAnimationFrame(render);
}

function fit() {
  ov.width = cam.clientWidth;
  ov.height = cam.clientHeight;
}
function sx() { return ov.width / native.w; }
function sy() { return ov.height / native.h; }
function toNative(ev) {
  const b = ov.getBoundingClientRect();
  return { x: (ev.clientX - b.left) / sx(), y: (ev.clientY - b.top) / sy() };
}

function handlesOf(r) {
  return {
    nw: { x: r.x,        y: r.y },
    ne: { x: r.x + r.w,  y: r.y },
    sw: { x: r.x,        y: r.y + r.h },
    se: { x: r.x + r.w,  y: r.y + r.h },
  };
}
function hitHandle(r, p) {
  const tol = 10 / sx();
  for (const [name, h] of Object.entries(handlesOf(r)))
    if (Math.abs(p.x - h.x) < tol && Math.abs(p.y - h.y) < tol) return name;
  return null;
}
function inside(r, p) {
  return p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h;
}
function norm(r) {
  if (r.w < 0) { r.x += r.w; r.w = -r.w; }
  if (r.h < 0) { r.y += r.h; r.h = -r.h; }
  r.x = Math.max(0, Math.round(r.x));
  r.y = Math.max(0, Math.round(r.y));
  r.w = Math.round(r.w);
  r.h = Math.round(r.h);
}

ov.addEventListener('mousedown', (ev) => {
  const p = toNative(ev);
  if (draw) { rubber = { x0: p.x, y0: p.y, x1: p.x, y1: p.y }; return; }
  for (const side of ['exit', 'enter']) {
    const r = regions[side];
    if (!r) continue;
    const h = hitHandle(r, p);
    if (h) { drag = { side, mode: 'resize', handle: h }; return; }
    if (inside(r, p)) {
      drag = { side, mode: 'move', dx: p.x - r.x, dy: p.y - r.y };
      return;
    }
  }
});

window.addEventListener('mousemove', (ev) => {
  const p = toNative(ev);
  if (rubber) { rubber.x1 = p.x; rubber.y1 = p.y; return; }
  if (!drag) return;
  const r = regions[drag.side];
  if (drag.mode === 'move') {
    r.x = p.x - drag.dx;
    r.y = p.y - drag.dy;
  } else {
    if (drag.handle.includes('w')) { r.w += r.x - p.x; r.x = p.x; }
    if (drag.handle.includes('n')) { r.h += r.y - p.y; r.y = p.y; }
    if (drag.handle.includes('e')) { r.w = p.x - r.x; }
    if (drag.handle.includes('s')) { r.h = p.y - r.y; }
  }
});

window.addEventListener('mouseup', () => {
  if (rubber) {
    const r = { x: rubber.x0, y: rubber.y0,
                w: rubber.x1 - rubber.x0, h: rubber.y1 - rubber.y0,
                rot180: document.querySelector(`[data-rot="${draw}"]`).checked };
    norm(r);
    if (r.w > 4 && r.h > 4) regions[draw] = r;
    rubber = null; draw = null;
    ov.style.cursor = 'crosshair';
    setStatus('drawn - remember to Save');
  }
  if (drag) { norm(regions[drag.side]); drag = null; setStatus('edited - remember to Save'); }
});

document.querySelectorAll('[data-draw]').forEach((b) =>
  b.addEventListener('click', () => {
    draw = b.dataset.draw;
    setStatus(`click two opposite corners for the ${draw.toUpperCase()} box`);
  }));
document.querySelectorAll('[data-clear]').forEach((b) =>
  b.addEventListener('click', () => {
    regions[b.dataset.clear] = null;
    setStatus(`${b.dataset.clear.toUpperCase()} cleared - remember to Save`);
  }));
document.querySelectorAll('[data-rot]').forEach((c) =>
  c.addEventListener('change', () => {
    const r = regions[c.dataset.rot];
    if (r) r.rot180 = c.checked;
    setStatus('rotation changed - remember to Save');
  }));

document.getElementById('save').addEventListener('click', async () => {
  const out = {};
  for (const side of ['enter', 'exit']) {
    const r = regions[side];
    if (r) out[side] = { box: [r.x, r.y, r.w, r.h], rot180: !!r.rot180 };
  }
  const res = await fetch('/regions', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(out),
  });
  setStatus(res.ok ? 'saved regions.json' : 'save failed');
});

function drawRect(side) {
  const r = regions[side];
  if (!r) return;
  const x = r.x * sx(), y = r.y * sy(), w = r.w * sx(), h = r.h * sy();
  ctx.strokeStyle = COLORS[side];
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = COLORS[side];
  ctx.font = '13px system-ui, sans-serif';
  ctx.fillText(side.toUpperCase() + (r.rot180 ? '  (rot180)' : ''), x + 4, y - 5);
  for (const h2 of Object.values(handlesOf(r)))
    ctx.fillRect(h2.x * sx() - 4, h2.y * sy() - 4, 8, 8);
}

function render() {
  if (ov.width !== cam.clientWidth || ov.height !== cam.clientHeight) fit();
  ctx.clearRect(0, 0, ov.width, ov.height);
  drawRect('exit');
  drawRect('enter');
  if (rubber) {
    ctx.strokeStyle = COLORS[draw] || '#fff';
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(rubber.x0 * sx(), rubber.y0 * sy(),
                   (rubber.x1 - rubber.x0) * sx(), (rubber.y1 - rubber.y0) * sy());
    ctx.setLineDash([]);
  }
  requestAnimationFrame(render);
}

cam.addEventListener('load', fit);
window.addEventListener('resize', fit);
boot();
</script>
"""


@app.route("/")
def index():
    return PAGE


@app.route("/meta")
def meta():
    frame = read_frame()
    if frame is None:
        return jsonify(w=CAMERA_RESOLUTION[0], h=CAMERA_RESOLUTION[1])
    h, w = frame.shape[:2]
    return jsonify(w=w, h=h)


@app.route("/regions", methods=["GET", "POST"])
def regions():
    if request.method == "GET":
        return jsonify(load_regions())
    data = request.get_json(force=True, silent=True) or {}
    clean = {}
    for side in ("enter", "exit"):
        r = data.get(side)
        if r and isinstance(r.get("box"), list) and len(r["box"]) == 4:
            clean[side] = {"box": [int(v) for v in r["box"]],
                           "rot180": bool(r.get("rot180"))}
    with open(REGIONS_FILE, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"wrote {REGIONS_FILE}: {clean}")
    return jsonify(ok=True, saved=clean)


@app.route("/stream")
def stream():
    def gen():
        while True:
            frame = read_frame()
            if frame is None:
                continue
            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print(f"calibration UI: http://0.0.0.0:{PORT}/  (Ctrl-C to stop)")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
