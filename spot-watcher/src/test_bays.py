#!/usr/bin/env python3
"""
Improved bay tester with automatic brightness compensation.
Detects which parking bays differ from the empty reference.
"""

import cv2, json, numpy as np, argparse
from skimage.metrics import structural_similarity as ssim

# -------- thresholds (tune only if needed) --------
SSIM_MIN = 0.90          # lower = less sensitive
SAT_DELTA_MIN = 40.0     # higher = less sensitive to colour
LUMA_DELTA_MIN = 40.0    # higher = less sensitive to brightness
CENTER_SIGMA_FRAC = 0.25 # how tight the center weighting is
RESIZE_TO = (120, 120)
# --------------------------------------------------

def load_bays(p):
    with open(p, "r") as f:
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
    wgt = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma))
    wgt = wgt / (wgt.mean() + 1e-9)
    return wgt

def metrics(ref_patch, test_patch):
    ref = cv2.resize(ref_patch, RESIZE_TO)
    cur = cv2.resize(test_patch, RESIZE_TO)
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

    return s_weighted, abs(sat_delta), luma_delta  # keep signed luma for compensation

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bays", default="bays.json")
    ap.add_argument("--ref", default="empty_reference.jpg")
    ap.add_argument("--img", default="CarParkTest.jpg")
    ap.add_argument("--out", default="bays_test_result.jpg")
    args = ap.parse_args()

    bays = load_bays(args.bays)
    ref = cv2.imread(args.ref)
    cur = cv2.imread(args.img)
    if ref is None or cur is None:
        raise SystemExit("❌ Could not read reference or test image")

    print("Bay  |  SSIM_w  |  ΔSat  |  ΔLuma  |  status")
    print("-----+----------+--------+---------+---------")

    annotated = cur.copy()
    results, luma_all = {}, []

    # first pass – collect luma deltas
    temp_metrics = {}
    for b in bays:
        name, pts = b["name"], b["points"]
        ref_p, cur_p = crop_polygon(ref, pts), crop_polygon(cur, pts)
        if ref_p.size == 0 or cur_p.size == 0:
            continue
        s_w, d_sat, d_luma = metrics(ref_p, cur_p)
        temp_metrics[name] = (s_w, d_sat, d_luma)
        luma_all.append(d_luma)

    # compensate for global brightness drift
    global_luma_shift = np.median(luma_all)
    print(f"\nGlobal brightness offset: {global_luma_shift:.1f}\n")

    free = 0
    for b in bays:
        name, pts = b["name"], b["points"]
        if name not in temp_metrics:
            continue
        s_w, d_sat, d_luma = temp_metrics[name]
        adj_luma = abs(d_luma - global_luma_shift)

        occupied = (s_w < SSIM_MIN) or (d_sat > SAT_DELTA_MIN) or (adj_luma > LUMA_DELTA_MIN)
        status = "occupied" if occupied else "free"
        results[name] = status
        if not occupied: free += 1

        print(f"{name:<4} | {s_w:7.3f}  | {d_sat:5.1f} | {adj_luma:6.1f} | {status}")

        color = (0,0,255) if occupied else (0,255,0)
        pts_np = np.array(pts, np.int32)
        cv2.polylines(annotated, [pts_np], True, color, 2)
        p0 = pts_np[0]
        cv2.putText(annotated, name, (p0[0]+4, p0[1]+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(args.out, annotated)
    print(f"\n✅ Free: {free}/{len(bays)}  (saved {args.out})")

if __name__ == "__main__":
    main()
