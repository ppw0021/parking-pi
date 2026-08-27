#!/usr/bin/env python3
"""
cal.py - browser-based calibration for the 16 parking bays.

The spot-watcher Pi is headless, so calibration runs as a small web app:

    uv run cal.py
    # then open  http://<pi-ip>:8000/  from a laptop on the same network

In the browser you get the live camera view with 16 boxes on top (6 top row,
4 middle, 6 bottom - spot ids 0-5, 6-9, 10-15, the order the web server
expects). Drag a box to move it, drag its corner to resize, click to select,
arrow keys nudge. Buttons:

  Refresh view              re-grab the camera frame
  Auto-detect               fit the 6/4/6 layout to the bright lane markers
  Reset grid                drop back to an even grid
  Save layout               write spots.json
  Capture empty references  (lot must be empty) write refs/00.png .. 15.png

main.py then reads spots.json + refs/.
"""
import io
import json
import os
import statistics
import threading

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file

# ---------------- Configuration ----------------
CAMERA_INDEX = 0
RESOLUTION = (1920, 1080)          # (width, height) requested from the camera
HOST, PORT = "0.0.0.0", 8000
SPOTS_PATH = "spots.json"
REFS_DIR = "refs"
PREVIEW_PATH = "cal_preview.jpg"
REFS_PREVIEW_PATH = "cal_preview_refs.jpg"
REF_SIZE = (128, 128)             # every reference crop is stored at this size

# (first spot id, number of spots) for the top, middle and bottom rows
ROWS = [(0, 6), (6, 4), (10, 6)]

# Auto-detection: bays are read from bright, tall-thin vertical lane markers.
MARKER_THRESH = 200               # grayscale level a marker pixel must exceed
MARKER_MIN_H_FRAC = 0.04          # min marker height, fraction of frame height
MARKER_MAX_W_FRAC = 0.05          # max marker width, fraction of frame width
MARKER_MIN_ASPECT = 2.0          # min height / width of a marker blob
# ----------------------------------------------

_cam_lock = threading.Lock()
_cap = None


def _camera():
    global _cap
    if _cap is None or not _cap.isOpened():
        _cap = cv2.VideoCapture(CAMERA_INDEX)
        _cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        _cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        _cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not _cap.isOpened():
        raise RuntimeError(f"could not open camera {CAMERA_INDEX}")
    return _cap


def grab_frame():
    """Newest available frame, with a few throwaways to clear the buffer."""
    with _cam_lock:
        cap = _camera()
        frame = None
        for _ in range(6):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
        if frame is None:
            raise RuntimeError("camera returned no frames")
        return frame


# ---------------- Geometry helpers ----------------
def clamp_box(box, width, height):
    x, y, w, h = box
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    w = max(1, min(int(w), width - x))
    h = max(1, min(int(h), height - y))
    return [x, y, w, h]


def split_row(start_id, count, x, y, w, h, gap_frac=0.15):
    """Divide an outer rectangle into `count` evenly spaced boxes."""
    cell = w / count
    gap = cell * gap_frac
    boxes = {}
    for i in range(count):
        boxes[str(start_id + i)] = [
            int(round(x + i * cell + gap / 2)),
            int(round(y)),
            int(round(cell - gap)),
            int(round(h)),
        ]
    return boxes


def default_grid(width, height):
    """An even 6/4/6 grid spread across the frame."""
    bands = [(0.12, 0.36, 0.05, 0.95),
             (0.42, 0.58, 0.20, 0.80),
             (0.64, 0.88, 0.05, 0.95)]
    spots = {}
    for (start_id, count), (y0, y1, x0, x1) in zip(ROWS, bands):
        spots.update(split_row(start_id, count, x0 * width, y0 * height,
                               (x1 - x0) * width, (y1 - y0) * height))
    return spots


# ---------------- Auto detection ----------------
def find_markers(frame):
    """Bright, tall-thin vertical blobs -> list of (x, y, w, h) bounding rects."""
    h_img, w_img = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, MARKER_THRESH, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(5, int(h_img * 0.03))))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    markers = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (h >= MARKER_MIN_H_FRAC * h_img
                and w <= MARKER_MAX_W_FRAC * w_img
                and h / max(w, 1) >= MARKER_MIN_ASPECT):
            markers.append((x, y, w, h))
    return markers


def group_rows(markers):
    """Cluster markers into horizontal rows by vertical centre."""
    if not markers:
        return []
    med_h = statistics.median(m[3] for m in markers)
    tol = max(med_h * 0.7, 25)
    rows = []
    for m in sorted(markers, key=lambda m: m[1]):
        cy = m[1] + m[3] / 2
        for row in rows:
            avg = sum(r[1] + r[3] / 2 for r in row) / len(row)
            if abs(cy - avg) < tol:
                row.append(m)
                break
        else:
            rows.append([m])
    return rows


def auto_detect(frame):
    """Detect lane markers and fit the 6/4/6 layout.

    Returns (spots_dict_or_None, markers). When a row shows exactly `count + 1`
    markers the bay edges come from them; otherwise the row is split evenly
    between its outermost markers.
    """
    h_img, w_img = frame.shape[:2]
    markers = find_markers(frame)
    rows = [r for r in group_rows(markers) if len(r) >= 2]
    if len(rows) < 3:
        return None, markers
    rows = sorted(rows, key=len, reverse=True)[:3]
    rows.sort(key=lambda r: sum(m[1] for m in r) / len(r))

    spots = {}
    for (start_id, count), row in zip(ROWS, rows):
        row = sorted(row, key=lambda m: m[0])
        y0 = min(m[1] for m in row)
        y1 = max(m[1] + m[3] for m in row)
        if len(row) == count + 1:
            edges = [m[0] + m[2] / 2 for m in row]
            for i in range(count):
                lx, rx = edges[i], edges[i + 1]
                gap = (rx - lx) * 0.12
                spots[str(start_id + i)] = clamp_box(
                    [lx + gap, y0, (rx - lx) - 2 * gap, y1 - y0], w_img, h_img)
        else:
            x0 = min(m[0] for m in row)
            x1 = max(m[0] + m[2] for m in row)
            spots.update({
                k: clamp_box(v, w_img, h_img) for k, v in
                split_row(start_id, count, x0, y0, x1 - x0, y1 - y0).items()})

    if len(spots) != 16:
        return None, markers
    return spots, markers


# ---------------- Persistence ----------------
def read_spots():
    if not os.path.exists(SPOTS_PATH):
        return None
    with open(SPOTS_PATH) as f:
        return json.load(f)


def write_spots(spots):
    ordered = {str(i): spots[str(i)] for i in range(16) if str(i) in spots}
    with open(SPOTS_PATH, "w") as f:
        json.dump({"resolution": list(RESOLUTION), "spots": ordered}, f, indent=2)


def draw_preview(frame, spots, path):
    vis = frame.copy()
    for i in range(16):
        box = spots.get(str(i))
        if not box:
            continue
        x, y, w, h = box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, str(i), (x + 4, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(path, vis)


def capture_refs(frame, spots):
    """Crop every bay from `frame` into refs/. Returns (ok, missing_ids)."""
    missing = [i for i in range(16) if str(i) not in spots]
    if missing:
        return False, missing
    os.makedirs(REFS_DIR, exist_ok=True)
    h_img, w_img = frame.shape[:2]
    tiles = []
    for i in range(16):
        x, y, w, h = clamp_box(spots[str(i)], w_img, h_img)
        gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, REF_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(REFS_DIR, f"{i:02d}.png"), gray)
        tile = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.putText(tile, str(i), (4, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(tile)
    montage = np.vstack([np.hstack(tiles[r:r + 8]) for r in (0, 8)])
    cv2.imwrite(REFS_PREVIEW_PATH, montage)
    return True, []


def clean_spots(raw):
    out = {}
    for i in range(16):
        b = raw.get(str(i), raw.get(i))
        if isinstance(b, (list, tuple)) and len(b) == 4:
            out[str(i)] = [int(round(float(v))) for v in b]
    return out


# ---------------- Web app ----------------
app = Flask(__name__)


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.get("/frame.jpg")
def frame_jpg():
    try:
        frame = grab_frame()
    except RuntimeError as e:
        return jsonify(error=str(e)), 503
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    resp = send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/api/spots")
def api_spots_get():
    saved = read_spots()
    if saved:
        return jsonify(native=saved.get("resolution", list(RESOLUTION)),
                       spots=saved.get("spots", {}), source="spots.json")
    w, h = RESOLUTION
    spots, source = None, "default grid"
    try:
        frame = grab_frame()
        h, w = frame.shape[:2]
        spots, _ = auto_detect(frame)
        source = "auto-detect"
    except RuntimeError:
        pass
    return jsonify(native=[w, h], spots=spots or default_grid(w, h),
                   source=source if spots else "default grid")


@app.post("/api/spots")
def api_spots_post():
    spots = clean_spots(request.get_json(force=True).get("spots", {}))
    if len(spots) != 16:
        return jsonify(ok=False, error="need all 16 boxes"), 400
    write_spots(spots)
    try:
        draw_preview(grab_frame(), spots, PREVIEW_PATH)
    except RuntimeError:
        pass
    return jsonify(ok=True)


@app.get("/api/grid")
def api_grid():
    w, h = RESOLUTION
    try:
        f = grab_frame()
        h, w = f.shape[:2]
    except RuntimeError:
        pass
    return jsonify(spots=default_grid(w, h))


@app.post("/api/auto")
def api_auto():
    try:
        frame = grab_frame()
    except RuntimeError as e:
        return jsonify(ok=False, error=str(e), markers=0), 503
    spots, markers = auto_detect(frame)
    return jsonify(ok=bool(spots), spots=spots or {}, markers=len(markers))


@app.post("/api/references")
def api_references():
    spots = clean_spots(request.get_json(force=True).get("spots", {}))
    if len(spots) != 16:
        return jsonify(ok=False, error="need all 16 boxes"), 400
    write_spots(spots)
    try:
        frame = grab_frame()
    except RuntimeError as e:
        return jsonify(ok=False, error=str(e)), 503
    ok, missing = capture_refs(frame, spots)
    if not ok:
        return jsonify(ok=False, missing=missing), 400
    draw_preview(frame, spots, PREVIEW_PATH)
    return jsonify(ok=True)


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Spot calibration</title><style>
body{font:14px system-ui,sans-serif;margin:0;background:#1e1e1e;color:#ddd}
header{padding:8px 12px;background:#252526;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button{padding:6px 10px;background:#0e639c;color:#fff;border:0;border-radius:4px;cursor:pointer}
button.alt{background:#3a3d41}
#wrap{padding:12px}
#stage{position:relative;overflow:hidden;border:1px solid #444;user-select:none}
#frame{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.box{position:absolute;border:2px solid #35d07f;background:rgba(53,208,127,.12);box-sizing:border-box}
.box.sel{border-color:#f5d90a;background:rgba(245,217,10,.15);z-index:5}
.box .lbl{position:absolute;left:0;top:0;background:#000a;color:#fff;font-size:12px;padding:0 4px}
.box .hnd{position:absolute;right:-6px;bottom:-6px;width:12px;height:12px;background:#f5d90a;border:1px solid #000;cursor:nwse-resize}
#readout{padding:6px 0;font-family:monospace;min-height:1.4em}
#readout input{width:5em;background:#3c3c3c;color:#eee;border:1px solid #555;border-radius:3px}
#hint{color:#888}
#toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:#333;padding:8px 14px;border-radius:4px;opacity:0;transition:.2s;pointer-events:none}
#toast.show{opacity:1}
</style></head><body>
<header>
  <strong>Spot calibration</strong>
  <button id="btn-refresh" class="alt">Refresh view</button>
  <label><input type="checkbox" id="chk-auto"> auto&nbsp;2s</label>
  <button id="btn-auto">Auto-detect</button>
  <button id="btn-grid" class="alt">Reset grid</button>
  <span style="flex:1"></span>
  <button id="btn-save">Save layout</button>
  <button id="btn-refs">Capture empty references</button>
</header>
<div id="wrap">
  <div id="readout">loading&hellip;</div>
  <div id="stage"><img id="frame" alt=""></div>
  <p id="hint">Drag a box to move, drag the yellow corner to resize, click to select,
     arrow keys nudge (Shift = &times;10). "Save layout" writes spots.json.
     "Capture empty references" must be done with the lot empty.</p>
</div>
<div id="toast"></div>
<script>
const stage=document.getElementById('stage'), frameImg=document.getElementById('frame');
let NATIVE=[1920,1080], SCALE=1, boxes=[], els=[], selected=-1, drag=null, autoTimer=null;

const toDisp=b=>({x:b[0]/SCALE,y:b[1]/SCALE,w:b[2]/SCALE,h:b[3]/SCALE});
function toNative(){const o={};boxes.forEach((b,i)=>o[i]=[Math.round(b.x*SCALE),Math.round(b.y*SCALE),Math.round(b.w*SCALE),Math.round(b.h*SCALE)]);return o;}
function seq(obj){const a=[];for(let i=0;i<16;i++){const b=obj[i]||obj[String(i)];if(b)a.push(toDisp(b));}return a;}

function toast(msg){const t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');
  clearTimeout(toast._t);toast._t=setTimeout(()=>t.classList.remove('show'),3000);}

function refreshView(){frameImg.src='/frame.jpg?ts='+Date.now();}

function clampBox(b){
  const W=stage.clientWidth, H=stage.clientHeight;
  b.w=Math.max(8,Math.min(b.w,W)); b.h=Math.max(8,Math.min(b.h,H));
  b.x=Math.max(0,Math.min(b.x,W-b.w)); b.y=Math.max(0,Math.min(b.y,H-b.h));
}
function place(i){const e=els[i],b=boxes[i];if(!e)return;
  e.style.left=b.x+'px';e.style.top=b.y+'px';e.style.width=b.w+'px';e.style.height=b.h+'px';
  e.classList.toggle('sel',i===selected);}

function render(){
  els.forEach(e=>e.remove()); els=[];
  boxes.forEach((b,i)=>{
    const e=document.createElement('div'); e.className='box';
    e.innerHTML='<span class="lbl">'+i+'</span><span class="hnd"></span>';
    e.addEventListener('mousedown',ev=>startDrag(ev,i));
    stage.appendChild(e); els[i]=e; place(i);
  });
  updateReadout();
}

function startDrag(ev,i){
  ev.preventDefault(); ev.stopPropagation();
  selected=i;
  drag={i,resize:ev.target.classList.contains('hnd'),mx:ev.clientX,my:ev.clientY,
        ox:boxes[i].x,oy:boxes[i].y,ow:boxes[i].w,oh:boxes[i].h};
  els.forEach((_,j)=>place(j));
  updateReadout();
}
document.addEventListener('mousemove',ev=>{
  if(!drag)return;
  const dx=ev.clientX-drag.mx, dy=ev.clientY-drag.my, b=boxes[drag.i];
  if(drag.resize){b.w=drag.ow+dx;b.h=drag.oh+dy;} else {b.x=drag.ox+dx;b.y=drag.oy+dy;}
  clampBox(b); place(drag.i); updateReadout();
});
document.addEventListener('mouseup',()=>{drag=null;});
stage.addEventListener('mousedown',ev=>{if(ev.target===stage||ev.target===frameImg){selected=-1;els.forEach((_,j)=>place(j));updateReadout();}});

document.addEventListener('keydown',ev=>{
  if(selected<0||!ev.key.startsWith('Arrow'))return;
  if(document.activeElement&&document.activeElement.tagName==='INPUT')return;
  ev.preventDefault();
  const s=ev.shiftKey?10:1, b=boxes[selected];
  if(ev.key==='ArrowLeft')b.x-=s; else if(ev.key==='ArrowRight')b.x+=s;
  else if(ev.key==='ArrowUp')b.y-=s; else if(ev.key==='ArrowDown')b.y+=s;
  clampBox(b); place(selected); updateReadout();
});

function updateReadout(){
  const r=document.getElementById('readout');
  if(selected<0){r.textContent='no spot selected';return;}
  const b=boxes[selected], n=[b.x,b.y,b.w,b.h].map(v=>Math.round(v*SCALE));
  r.innerHTML='spot <b>'+selected+'</b> native px &nbsp;'+
    ['x','y','w','h'].map((k,j)=>k+' <input data-k="'+j+'" value="'+n[j]+'">').join(' &nbsp;');
  r.querySelectorAll('input').forEach(inp=>inp.addEventListener('change',()=>{
    const j=+inp.dataset.k, v=parseFloat(inp.value)/SCALE, b=boxes[selected];
    if(isNaN(v))return;
    if(j===0)b.x=v; else if(j===1)b.y=v; else if(j===2)b.w=v; else b.h=v;
    clampBox(b); place(selected); updateReadout();
  }));
}

async function jget(u){return (await fetch(u)).json();}
async function jpost(u,body){return (await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(body||{})})).json();}

document.getElementById('btn-refresh').onclick=refreshView;
document.getElementById('chk-auto').onchange=e=>{
  clearInterval(autoTimer);
  if(e.target.checked)autoTimer=setInterval(refreshView,2000);
};
document.getElementById('btn-auto').onclick=async()=>{
  const r=await jpost('/api/auto');
  if(!r.ok){toast('auto-detect failed ('+(r.markers||0)+' markers) - adjust MARKER_* in cal.py');return;}
  boxes=seq(r.spots); selected=-1; render(); toast('auto-detected from '+r.markers+' markers');
};
document.getElementById('btn-grid').onclick=async()=>{
  const r=await jget('/api/grid'); boxes=seq(r.spots); selected=-1; render(); toast('reset to even grid');
};
document.getElementById('btn-save').onclick=async()=>{
  const r=await jpost('/api/spots',{spots:toNative()});
  toast(r.ok?'saved spots.json':('save failed: '+(r.error||'')));
};
document.getElementById('btn-refs').onclick=async()=>{
  if(!confirm('The lot must be EMPTY. Capture references now?'))return;
  const r=await jpost('/api/references',{spots:toNative()});
  if(r.ok)toast('saved 16 references + spots.json');
  else toast('failed: '+(r.error||'')+(r.missing?(' missing '+r.missing.join(',')):''));
};

async function boot(){
  const s=await jget('/api/spots');
  NATIVE=s.native;
  const dispW=Math.min(1280,NATIVE[0]);
  SCALE=NATIVE[0]/dispW;
  stage.style.width=(NATIVE[0]/SCALE)+'px';
  stage.style.height=(NATIVE[1]/SCALE)+'px';
  refreshView();
  boxes=seq(s.spots); render();
  toast('loaded: '+s.source);
}
boot();
</script></body></html>"""


if __name__ == "__main__":
    print(f"calibration UI on http://{HOST}:{PORT}/  (open it from a laptop)")
    app.run(host=HOST, port=PORT, threaded=True)
