#!/usr/bin/env python3
"""
cal.py - browser-based calibration for the 16 parking bays.

The spot-watcher Pi is headless, so calibration runs as a small web app:

    uv run cal.py
    # then open  http://<pi-ip>:8000/  from a laptop on the same network

The page shows the live camera view with a row of 16 numbered pads above it
(6 top row, 4 middle, 6 bottom - spot ids 0-5, 6-9, 10-15, the order the web
server expects). To place a bay: click its pad, then click the two opposite
corners of that bay in the image. It advances to the next unplaced bay
automatically. Placed boxes can then be dragged / resized / nudged to fine-tune.

  Save layout               write spots.json (partial is allowed - resume later)
  Capture empty references  (all 16 placed, lot empty) write refs/00.png .. 15.png

main.py then reads spots.json + refs/.
"""
import io
import json
import os
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


# ---------------- Helpers ----------------
def clamp_box(box, width, height):
    x, y, w, h = box
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    w = max(1, min(int(w), width - x))
    h = max(1, min(int(h), height - y))
    return [x, y, w, h]


def read_spots():
    if not os.path.exists(SPOTS_PATH):
        return None
    with open(SPOTS_PATH) as f:
        return json.load(f)


def write_spots(spots):
    ordered = {str(i): spots[str(i)] for i in range(16) if str(i) in spots}
    with open(SPOTS_PATH, "w") as f:
        json.dump({"resolution": list(RESOLUTION), "spots": ordered}, f, indent=2)


def clean_spots(raw):
    out = {}
    for i in range(16):
        b = raw.get(str(i), raw.get(i))
        if isinstance(b, (list, tuple)) and len(b) == 4:
            out[str(i)] = [int(round(float(v))) for v in b]
    return out


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
    try:
        f = grab_frame()
        h, w = f.shape[:2]
    except RuntimeError:
        pass
    return jsonify(native=[w, h], spots={}, source="new")


@app.post("/api/spots")
def api_spots_post():
    spots = clean_spots(request.get_json(force=True).get("spots", {}))
    write_spots(spots)
    try:
        draw_preview(grab_frame(), spots, PREVIEW_PATH)
    except RuntimeError:
        pass
    return jsonify(ok=True, count=len(spots))


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
#pads{display:flex;gap:4px;flex-wrap:wrap;padding:8px 12px;background:#2d2d2d}
.pad{min-width:34px;padding:6px 0;background:#3a3d41;color:#ccc;border:2px solid transparent;border-radius:4px;cursor:pointer;font-family:monospace}
.pad.placed{background:#1c3a2a;color:#7fe0a8;border-color:#35d07f}
.pad.armed{border-color:#f5d90a;color:#fff;background:#4a4324}
#status{padding:6px 12px;color:#f5d90a;font-family:monospace;min-height:1.3em}
#wrap{padding:12px}
#stage{position:relative;overflow:hidden;border:1px solid #444;user-select:none}
#frame{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.box{position:absolute;border:2px solid #35d07f;background:rgba(53,208,127,.12);box-sizing:border-box}
.box.sel{border-color:#f5d90a;background:rgba(245,217,10,.15);z-index:5}
.box .lbl{position:absolute;left:0;top:0;background:#000a;color:#fff;font-size:12px;padding:0 4px}
.box .hnd{position:absolute;right:-6px;bottom:-6px;width:12px;height:12px;background:#f5d90a;border:1px solid #000;cursor:nwse-resize}
#catch{position:absolute;inset:0;z-index:20;cursor:crosshair;display:none}
#rubber{position:absolute;z-index:21;border:2px dashed #f5d90a;background:rgba(245,217,10,.12);display:none;pointer-events:none}
#mark{position:absolute;z-index:22;width:14px;height:14px;margin:-7px 0 0 -7px;border:2px solid #f5d90a;border-radius:50%;display:none;pointer-events:none}
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
  <button id="btn-clear" class="alt">Clear all</button>
  <button id="btn-stop" class="alt" style="display:none">Stop placing</button>
  <span style="flex:1"></span>
  <button id="btn-save">Save layout</button>
  <button id="btn-refs">Capture empty references</button>
</header>
<div id="pads"></div>
<div id="status"></div>
<div id="wrap">
  <div id="readout">loading&hellip;</div>
  <div id="stage">
    <img id="frame" alt="">
    <div id="catch"></div>
    <div id="rubber"></div>
    <div id="mark"></div>
  </div>
  <p id="hint">Click a numbered pad, then click the two opposite corners of that
     bay in the image &mdash; it advances to the next unplaced bay automatically.
     Esc or "Stop placing" leaves placement mode; then drag a box or its corner
     to fine-tune, or edit the numbers in the readout (arrow keys nudge,
     Shift = &times;10). "Capture empty references" needs all 16 placed and the
     lot empty.</p>
</div>
<div id="toast"></div>
<script>
const stage=document.getElementById('stage'), frameImg=document.getElementById('frame');
const cat=document.getElementById('catch'), rubber=document.getElementById('rubber'),
      mark=document.getElementById('mark');
let NATIVE=[1920,1080], SCALE=1;
let boxes=new Array(16).fill(null);   // display-px {x,y,w,h} or null
let els=new Array(16).fill(null);
let selected=-1, armed=-1, cornerA=null, drag=null, autoTimer=null;

function toNative(){
  const o={};
  boxes.forEach((b,i)=>{ if(b) o[i]=[Math.round(b.x*SCALE),Math.round(b.y*SCALE),
                                     Math.round(b.w*SCALE),Math.round(b.h*SCALE)]; });
  return o;
}
function placedCount(){return boxes.filter(Boolean).length;}
function nextUnplaced(from){
  for(let k=1;k<=16;k++){const i=(from+k)%16; if(!boxes[i])return i;}
  return -1;
}

function toast(msg){const t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');
  clearTimeout(toast._t);toast._t=setTimeout(()=>t.classList.remove('show'),3000);}
function status(msg){document.getElementById('status').textContent=msg;}
function refreshView(){frameImg.src='/frame.jpg?ts='+Date.now();}

function localPoint(ev){
  const r=stage.getBoundingClientRect();
  return {x:Math.max(0,Math.min(ev.clientX-r.left, stage.clientWidth)),
          y:Math.max(0,Math.min(ev.clientY-r.top, stage.clientHeight))};
}
function clampBox(b){
  const W=stage.clientWidth, H=stage.clientHeight;
  b.w=Math.max(8,Math.min(b.w,W)); b.h=Math.max(8,Math.min(b.h,H));
  b.x=Math.max(0,Math.min(b.x,W-b.w)); b.y=Math.max(0,Math.min(b.y,H-b.h));
}
function place(i){const e=els[i],b=boxes[i];if(!e||!b)return;
  e.style.left=b.x+'px';e.style.top=b.y+'px';e.style.width=b.w+'px';e.style.height=b.h+'px';
  e.classList.toggle('sel',i===selected);}

function renderPads(){
  const pads=document.getElementById('pads'); pads.innerHTML='';
  for(let i=0;i<16;i++){
    const b=document.createElement('button');
    b.textContent=i;
    b.className='pad'+(boxes[i]?' placed':'')+(i===armed?' armed':'');
    b.onclick=()=>arm(i);
    pads.appendChild(b);
  }
}
function renderBoxes(){
  els.forEach(e=>e&&e.remove()); els=new Array(16).fill(null);
  boxes.forEach((b,i)=>{
    if(!b)return;
    const e=document.createElement('div'); e.className='box';
    e.innerHTML='<span class="lbl">'+i+'</span><span class="hnd"></span>';
    e.addEventListener('mousedown',ev=>startDrag(ev,i));
    stage.appendChild(e); els[i]=e; place(i);
  });
  renderPads(); updateReadout();
}

// ---- placement mode ----
function arm(i){
  armed=i; selected=i; cornerA=null;
  rubber.style.display='none'; mark.style.display='none';
  cat.style.display='block';
  document.getElementById('btn-stop').style.display='';
  status('Spot '+i+': click the FIRST corner in the image');
  renderPads(); els.forEach((_,j)=>place(j));
}
function disarm(){
  armed=-1; cornerA=null;
  rubber.style.display='none'; mark.style.display='none';
  cat.style.display='none';
  document.getElementById('btn-stop').style.display='none';
  status(placedCount()+' / 16 placed');
  renderPads();
}
cat.addEventListener('click',ev=>{
  const p=localPoint(ev);
  if(!cornerA){
    cornerA=p;
    mark.style.display='block'; mark.style.left=p.x+'px'; mark.style.top=p.y+'px';
    status('Spot '+armed+': click the OPPOSITE corner');
  }else{
    const b={x:Math.min(cornerA.x,p.x),y:Math.min(cornerA.y,p.y),
             w:Math.abs(cornerA.x-p.x),h:Math.abs(cornerA.y-p.y)};
    if(b.w<6||b.h<6){status('Spot '+armed+': too small - click the FIRST corner again');
      cornerA=null; rubber.style.display='none'; mark.style.display='none'; return;}
    boxes[armed]=b; renderBoxes();
    cornerA=null; rubber.style.display='none'; mark.style.display='none';
    const nxt=nextUnplaced(armed);
    if(nxt<0){disarm(); status('all 16 placed - Save layout, then Capture references');}
    else arm(nxt);
  }
});
cat.addEventListener('mousemove',ev=>{
  if(!cornerA)return;
  const p=localPoint(ev);
  rubber.style.display='block';
  rubber.style.left=Math.min(cornerA.x,p.x)+'px';
  rubber.style.top=Math.min(cornerA.y,p.y)+'px';
  rubber.style.width=Math.abs(cornerA.x-p.x)+'px';
  rubber.style.height=Math.abs(cornerA.y-p.y)+'px';
});
cat.addEventListener('contextmenu',ev=>{
  ev.preventDefault();
  cornerA=null; rubber.style.display='none'; mark.style.display='none';
  status('Spot '+armed+': click the FIRST corner in the image');
});

// ---- edit mode (drag / resize / nudge) ----
function startDrag(ev,i){
  if(armed>=0)return;
  ev.preventDefault(); ev.stopPropagation();
  selected=i;
  drag={i,resize:ev.target.classList.contains('hnd'),mx:ev.clientX,my:ev.clientY,
        ox:boxes[i].x,oy:boxes[i].y,ow:boxes[i].w,oh:boxes[i].h};
  els.forEach((_,j)=>place(j)); updateReadout();
}
document.addEventListener('mousemove',ev=>{
  if(!drag)return;
  const dx=ev.clientX-drag.mx, dy=ev.clientY-drag.my, b=boxes[drag.i];
  if(drag.resize){b.w=drag.ow+dx;b.h=drag.oh+dy;} else {b.x=drag.ox+dx;b.y=drag.oy+dy;}
  clampBox(b); place(drag.i); updateReadout();
});
document.addEventListener('mouseup',()=>{drag=null;});
document.addEventListener('keydown',ev=>{
  if(ev.key==='Escape'&&armed>=0){disarm();return;}
  if(selected<0||!boxes[selected]||!ev.key.startsWith('Arrow'))return;
  if(document.activeElement&&document.activeElement.tagName==='INPUT')return;
  ev.preventDefault();
  const s=ev.shiftKey?10:1, b=boxes[selected];
  if(ev.key==='ArrowLeft')b.x-=s; else if(ev.key==='ArrowRight')b.x+=s;
  else if(ev.key==='ArrowUp')b.y-=s; else if(ev.key==='ArrowDown')b.y+=s;
  clampBox(b); place(selected); updateReadout();
});

function updateReadout(){
  const r=document.getElementById('readout');
  if(selected<0||!boxes[selected]){r.textContent=placedCount()+' / 16 placed';return;}
  const b=boxes[selected], n=[b.x,b.y,b.w,b.h].map(v=>Math.round(v*SCALE));
  r.innerHTML='spot <b>'+selected+'</b> native px &nbsp;'+
    ['x','y','w','h'].map((k,j)=>k+' <input data-k="'+j+'" value="'+n[j]+'">').join(' &nbsp;');
  r.querySelectorAll('input').forEach(inp=>inp.addEventListener('change',()=>{
    const j=+inp.dataset.k, v=parseFloat(inp.value)/SCALE, b=boxes[selected];
    if(isNaN(v)||!b)return;
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
document.getElementById('btn-stop').onclick=disarm;
document.getElementById('btn-clear').onclick=()=>{
  if(!confirm('Clear all 16 boxes?'))return;
  boxes=new Array(16).fill(null); selected=-1; renderBoxes(); arm(0);
};
document.getElementById('btn-save').onclick=async()=>{
  const r=await jpost('/api/spots',{spots:toNative()});
  toast(r.ok?('saved '+r.count+'/16 to spots.json'):('save failed: '+(r.error||'')));
};
document.getElementById('btn-refs').onclick=async()=>{
  if(placedCount()<16){toast('place all 16 first ('+placedCount()+'/16)');return;}
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
  boxes=new Array(16).fill(null);
  for(let i=0;i<16;i++){
    const b=s.spots[i]||s.spots[String(i)];
    if(b)boxes[i]={x:b[0]/SCALE,y:b[1]/SCALE,w:b[2]/SCALE,h:b[3]/SCALE};
  }
  renderBoxes();
  const first=nextUnplaced(-1);
  if(first>=0)arm(first); else disarm();
  toast('loaded: '+s.source);
}
boot();
</script></body></html>"""


if __name__ == "__main__":
    print(f"calibration UI on http://{HOST}:{PORT}/  (open it from a laptop)")
    app.run(host=HOST, port=PORT, threaded=True)
