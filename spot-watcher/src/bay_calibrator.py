#!/usr/bin/env python3
import argparse, json, os
from flask import Flask, jsonify, request, send_file, render_template_string

HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Parking Bay Calibrator</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  body { margin:0; font-family: system-ui, Arial, sans-serif; background:#0b0b0b; color:#eaeaea; }
  .toolbar { display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; padding:.75rem; position:sticky; top:0; background:#111; border-bottom:1px solid #222; }
  .toolbar input, .toolbar button, .toolbar select { background:#1b1b1b; color:#eaeaea; border:1px solid #333; padding:.35rem .6rem; border-radius:.4rem; }
  .toolbar button:hover { background:#262626; cursor:pointer; }
  .wrap { display:flex; justify-content:center; padding:1rem; }
  #canvas { max-width:95vw; max-height:85vh; background:#000; border:1px solid #333; }
  .hint { color:#9aa; font-size:.9rem; padding:.5rem 1rem; }
  .badge { display:inline-block; padding:.25rem .45rem; border:1px solid #333; border-radius:.35rem; background:#161616; }
</style>
</head>
<body>
  <div class="toolbar">
    <span class="badge">Click = add point · Enter = commit</span>
    <button id="undoPt">Undo point (u)</button>
    <button id="delBay">Delete last bay (d)</button>
    <button id="save">Save (s)</button>
    <button id="load">Load</button>
    <button id="clear">Clear all</button>
    <button id="confirm">Confirm Calibration</button>
    <label>Prefix <input id="prefix" size="3"></label>
    <label>Next ID <input id="nextId" type="number" style="width:5ch"></label>
    <label>Bay Name <input id="bayName" size="6" placeholder="auto"></label>
    <span class="badge">Drag = move point · Alt+Click = insert · Shift+Click = delete</span>
  </div>

  <div class="wrap"><canvas id="canvas"></canvas></div>
  <div class="hint" id="status">Loading…</div>

<script>
(() => {
  const imgUrl = new URLSearchParams(window.location.search).get('img') || '/image';
  const prefixDefault = new URLSearchParams(window.location.search).get('prefix') || 'B';
  const startIdDefault = parseInt(new URLSearchParams(window.location.search).get('start') || '0');

  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d', {alpha:false});
  const status = document.getElementById('status');
  const btnUndoPt = document.getElementById('undoPt');
  const btnDelBay = document.getElementById('delBay');
  const btnSave = document.getElementById('save');
  const btnLoad = document.getElementById('load');
  const btnClear = document.getElementById('clear');
  const btnConfirm = document.getElementById('confirm');
  const inpPrefix = document.getElementById('prefix');
  const inpNextId = document.getElementById('nextId');
  const inpBayName = document.getElementById('bayName');

  const img = new Image();
  let bays = [];
  let current = [];
  let draggingIdx = null;
  let scale = 1, imgW=0, imgH=0;

  function setStatus(t){ status.textContent = t; }

  function layoutCanvas() {
    const vw = Math.min(window.innerWidth*0.95, 1600);
    const vh = Math.min(window.innerHeight*0.85, 1200);
    scale = Math.min(vw / imgW, vh / imgH);
    canvas.width = Math.round(imgW * scale);
    canvas.height = Math.round(imgH * scale);
    draw();
  }

  function toImgCoords(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    return [
      Math.round((clientX - rect.left) / scale),
      Math.round((clientY - rect.top) / scale)
    ];
  }

  function draw() {
    ctx.fillStyle = '#000';
    ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(img, 0,0, canvas.width, canvas.height);

    // draw saved bays
    for (let b of bays) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#39ff14';
      ctx.beginPath();
      b.points.forEach((p,i)=>{
        const [x,y]=p; const sx=x*scale, sy=y*scale;
        if(i===0) ctx.moveTo(sx,sy); else ctx.lineTo(sx,sy);
      });
      if(b.points.length>=2) ctx.closePath();
      ctx.stroke();
      // label + points
      b.points.forEach(([x,y])=>{
        ctx.fillStyle='#111';ctx.strokeStyle='#39ff14';
        ctx.beginPath();ctx.arc(x*scale,y*scale,5,0,2*Math.PI);
        ctx.fill();ctx.stroke();
      });
      const [x0,y0]=b.points[0];
      ctx.fillStyle='#39ff14';
      ctx.font='16px system-ui';
      ctx.fillText(b.name,x0*scale+6,y0*scale-8);
    }

    // current polygon
    if(current.length>0){
      ctx.lineWidth=2;ctx.strokeStyle='#00c8ff';
      ctx.beginPath();
      current.forEach((p,i)=>{
        const [x,y]=p; const sx=x*scale,sy=y*scale;
        if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy);
      });
      ctx.stroke();
      for(let [x,y] of current){
        ctx.fillStyle='#111';ctx.strokeStyle='#00c8ff';
        ctx.beginPath();ctx.arc(x*scale,y*scale,5,0,2*Math.PI);
        ctx.fill();ctx.stroke();
      }
      const [x0,y0]=current[0];
      ctx.fillStyle='#00c8ff';
      ctx.font='14px system-ui';
      ctx.fillText(nextName()+" (Enter to commit)",x0*scale+6,y0*scale-8);
    }
  }

  function nextName(){
    const pf=inpPrefix.value;
    const nm=inpBayName.value.trim();
    return nm || pf + inpNextId.value;
  }

  function nearestPointIdx(mx,my){
    const r=8/scale;
    for(let bi=0;bi<bays.length;bi++){
      for(let pi=0;pi<bays[bi].points.length;pi++){
        const [x,y]=bays[bi].points[pi];
        if(Math.hypot(mx-x,my-y)<=r) return [bi,pi];
      }
    }
    return null;
  }

  // Mouse interaction
  canvas.addEventListener('mousedown',e=>{
    const [ix,iy]=toImgCoords(e.clientX,e.clientY);
    if(e.shiftKey){
      const hit=nearestPointIdx(ix,iy);
      if(hit){const[bi,pi]=hit;bays[bi].points.splice(pi,1);
        if(bays[bi].points.length<3)bays.splice(bi,1);
        draw();return;}
    }
    const hit=nearestPointIdx(ix,iy);
    if(hit){draggingIdx=hit;return;}
    current.push([ix,iy]);
    draw();
  });
  canvas.addEventListener('mousemove',e=>{
    if(!draggingIdx)return;
    const [ix,iy]=toImgCoords(e.clientX,e.clientY);
    const[bi,pi]=draggingIdx;bays[bi].points[pi]=[ix,iy];draw();
  });
  window.addEventListener('mouseup',()=>draggingIdx=null);

  // Keyboard shortcuts
  window.addEventListener('keydown',e=>{
    if(e.key==='Enter'){
      if(current.length>=3){
        bays.push({name:nextName(),points:current.slice()});
        inpBayName.value='';inpNextId.value=parseInt(inpNextId.value)+1;
        current=[];draw();
      }
    }else if(e.key==='u'){if(current.length)current.pop();draw();}
    else if(e.key==='d'){bays.pop();draw();}
    else if(e.key==='s'){save();}
  });

  btnUndoPt.onclick=()=>{if(current.length)current.pop();draw();};
  btnDelBay.onclick=()=>{bays.pop();draw();};
  btnSave.onclick=()=>save();
  btnLoad.onclick=async()=>{await load();draw();};
  btnClear.onclick=()=>{bays=[];current=[];draw();};
  btnConfirm.onclick=async()=>{
    if(!confirm("Are you sure you want to confirm and finalize calibration?"))return;
    const r=await fetch('/confirm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({bays})});
    const j=await r.json();
    setStatus(j.message||'Calibration confirmed');
  };

  async function save(){
    const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({bays})});
    const j=await r.json();
    setStatus(j.message||'Saved');
  }

  async function load(){
    const r=await fetch('/load');const j=await r.json();
    bays=j.bays||[];setStatus(`Loaded ${bays.length} bays`);
  }

  // Init
  inpPrefix.value=prefixDefault;inpNextId.value=startIdDefault;
  img.onload=()=>{imgW=img.width;imgH=img.height;layoutCanvas();setStatus('Ready: click to add points; Enter to commit.');};
  img.onerror=()=>setStatus('Failed to load image');
  window.addEventListener('resize',layoutCanvas);
  img.src=imgUrl;
})();
</script>
</body>
</html>
"""

# --------------------- Flask backend --------------------- #
def make_app(image_path: str, out_json: str):
    app = Flask(__name__)
    app.config["IMAGE_PATH"] = image_path
    app.config["OUT_JSON"] = out_json

    @app.get("/")
    def index():
        return render_template_string(HTML)

    @app.get("/image")
    def image():
        return send_file(app.config["IMAGE_PATH"])

    @app.get("/load")
    def load():
        p = app.config["OUT_JSON"]
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    return jsonify({"bays": json.load(f)})
            except Exception:
                return jsonify({"bays": []})
        return jsonify({"bays": []})

    @app.post("/save")
    def save():
        data = request.get_json(silent=True) or {}
        bays = data.get("bays", [])
        with open(app.config["OUT_JSON"], "w") as f:
            json.dump(bays, f, indent=2)
        return jsonify({"ok": True, "message": f"💾 Saved {len(bays)} bays to {app.config['OUT_JSON']}"})

    @app.post("/confirm")
    def confirm():
        data = request.get_json(silent=True) or {}
        bays = data.get("bays", [])
        with open(app.config["OUT_JSON"], "w") as f:
            json.dump(bays, f, indent=2)
        # Create a ".confirmed" flag file
        with open(app.config["OUT_JSON"] + ".confirmed", "w") as f:
            f.write("Calibration confirmed.\n")
        return jsonify({"ok": True, "message": f"Calibration confirmed. {len(bays)} bays saved."})

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Reference image to annotate (e.g., CarParkReference.jpg)")
    ap.add_argument("--out", default="bays.json", help="Output file for bays JSON")
    ap.add_argument("--host", default="0.0.0.0", help="Host (default 0.0.0.0)")
    ap.add_argument("--port", type=int, default=5000, help="Port (default 5000)")
    args = ap.parse_args()

    app = make_app(args.image, args.out)
    print(f"Bay calibration server running — open http://<pi-ip>:{args.port}/ in your browser.")
    print(f"Using image: {args.image}")
    print(f"Output file: {args.out}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
