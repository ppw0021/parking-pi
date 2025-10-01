#!/usr/bin/env python3
"""
Detects parking bay availability and saves results locally.
Outputs:
 - latest_spots.json  (IDs + taken status)
 - bayStatus.jpg      (visual annotated image)
"""

import argparse, json, re, cv2, numpy as np
from skimage.metrics import structural_similarity as ssim

# -------- thresholds --------
SSIM_MIN = 0.85
SAT_DELTA_MIN = 50.0
LUMA_DELTA_MIN = 50.0
CENTER_SIGMA_FRAC = 0.25
RESIZE_TO = (120, 120)
# ----------------------------

def load_bays(path):
    with open(path, "r") as f:
        return json.load(f)

def crop_polygon(img, points):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    pts = np.array(points, np.int32)
    cv2.fillPoly(mask, [pts], 255)
    x, y, w, h = cv2.boundingRect(pts)
    roi = cv2.bitwise_and(img, img, mask=mask)[y:y+h, x:x+w]
    return roi

def center_weight(h, w, sigma_frac=CENTER_SIGMA_FRAC):
    y, x = np.ogrid[:h, :w]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    sigma = min(w, h) * sigma_frac
    wgt = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma * sigma))
    return wgt / (wgt.mean() + 1e-9)

def patch_metrics(ref_patch, cur_patch):
    ref = cv2.resize(ref_patch, RESIZE_TO)
    cur = cv2.resize(cur_patch, RESIZE_TO)
    ref_hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
    cur_hsv = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)
    ref_v, ref_s = ref_hsv[...,2].astype(np.float32), ref_hsv[...,1].astype(np.float32)
    cur_v, cur_s = cur_hsv[...,2].astype(np.float32), cur_hsv[...,1].astype(np.float32)

    H, W = RESIZE_TO[1], RESIZE_TO[0]
    WGT = center_weight(H, W)

    ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    cur_g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    s, diff = ssim(ref_g, cur_g, full=True)
    diff = 1.0 - diff
    s_weighted = 1.0 - float((diff * WGT).mean())

    sat_delta = float(((cur_s - ref_s) * WGT).mean())
    luma_delta = float(((cur_v - ref_v) * WGT).mean())

    return s_weighted, abs(sat_delta), luma_delta

def parse_id_from_name(name, fallback):
    import re
    m = re.search(r'(\d+)$', name)
    return int(m.group(1)) if m else fallback

def analyze_and_save(bays_path, ref_path, img_path, out_json="latest_spots.json"):
    bays = load_bays(bays_path)
    ref = cv2.imread(ref_path)
    cur = cv2.imread(img_path)
    if ref is None or cur is None:
        raise SystemExit("Could not read reference or current image")

    temp, lumas = {}, []
    for b in bays:
        pts = b["points"]
        ref_p, cur_p = crop_polygon(ref, pts), crop_polygon(cur, pts)
        if ref_p.size == 0 or cur_p.size == 0:
            continue
        s_w, d_sat, d_luma = patch_metrics(ref_p, cur_p)
        temp[b["name"]] = (s_w, d_sat, d_luma)
        lumas.append(d_luma)

    global_luma_shift = np.median(lumas) if lumas else 0.0
    payload = []
    fallback_id = 0
    annotated = cur.copy()

    for b in bays:
        name = b["name"]
        if name not in temp:
            continue
        s_w, d_sat, d_luma = temp[name]
        adj_luma = abs(d_luma - global_luma_shift)
        taken = (s_w < SSIM_MIN) or (d_sat > SAT_DELTA_MIN) or (adj_luma > LUMA_DELTA_MIN)
        bay_id = parse_id_from_name(name, fallback_id)
        fallback_id += 1
        payload.append({"id": bay_id, "taken": bool(taken)})

        color = (0, 0, 255) if taken else (0, 255, 0)
        pts = np.array(b["points"], np.int32)
        cv2.polylines(annotated, [pts], True, color, 2)
        p = pts[0]
        cv2.putText(annotated, f"{bay_id}", (p[0]+5, p[1]+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save annotated image
    cv2.imwrite("bayStatus.jpg", annotated)

    # Save payload JSON
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {out_json} and bayStatus.jpg")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bays", default="bays.json")
    ap.add_argument("--ref", default="empty_reference.jpg")
    ap.add_argument("--img", default="current.jpg")
    ap.add_argument("--out", default="latest_spots.json")
    args = ap.parse_args()

    analyze_and_save(args.bays, args.ref, args.img, args.out)

if __name__ == "__main__":
    main()
