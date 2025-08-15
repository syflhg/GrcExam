
"""
Markers-only (or fallback) detector with **auto corner inference** for each marker.
It samples a small band just outside each marker side to detect the two dark borders
(thick square frame), then infers whether the marker is TL/TR/BR/BL of the square.
This fixes the "script thinks it's top-left" issue when phone photos are rotated.

Usage:
  python detect_squares_aruco_markers_auto.py image.jpg --out out --dict 5X5_100 --markers-only --debug
Options:
  --marker-frac, --pad-frac describe your design ratios. Corner is auto unless --marker-corner is set.
"""
import cv2
import numpy as np
import argparse, os
from math import hypot

def get_dict(dict_name: str):
    name = dict_name.upper()
    mapping = {
        "4X4_50": cv2.aruco.DICT_4X4_50, "4X4_100": cv2.aruco.DICT_4X4_100,
        "5X5_50": cv2.aruco.DICT_5X5_50, "5X5_100": cv2.aruco.DICT_5X5_100,
        "6X6_250": cv2.aruco.DICT_6X6_250, "7X7_1000": cv2.aruco.DICT_7X7_1000,
    }
    if name not in mapping:
        raise SystemExit(f"Unsupported dict '{dict_name}'")
    return cv2.aruco.getPredefinedDictionary(mapping[name])

def detect_markers(img_bgr, dict_name):
    DICT = get_dict(dict_name)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    try:
        det = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(gray)
    except AttributeError:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, DICT, parameters=cv2.aruco.DetectorParameters())
    if ids is None or len(ids) == 0:
        return []
    return [(int(ids[i][0]), corners[i].reshape(-1,2)) for i in range(len(ids))]

def clamp(x, a, b): return int(max(a, min(b, x)))

def infer_corner_from_borders(gray, bbox, margin=4, band=6):
    # bbox: (left, top, right, bottom)
    h, w = gray.shape
    l,t,r,b = map(int, bbox)
    # sample bands OUTSIDE the marker on each side
    # top band: just above top
    tb_y0 = max(0, t - margin - band); tb_y1 = max(0, t - margin)
    tb_x0 = max(0, l); tb_x1 = min(w, r)
    # bottom band: just below bottom
    bb_y0 = min(h, b + margin); bb_y1 = min(h, b + margin + band)
    bb_x0 = tb_x0; bb_x1 = tb_x1
    # left band: just left of left
    lb_x0 = max(0, l - margin - band); lb_x1 = max(0, l - margin)
    lb_y0 = max(0, t); lb_y1 = min(h, b)
    # right band: just right of right
    rb_x0 = min(w, r + margin); rb_x1 = min(w, r + margin + band)
    rb_y0 = lb_y0; rb_y1 = lb_y1

    def mean_band(y0,y1,x0,x1):
        if y1<=y0 or x1<=x0: return 255.0
        roi = gray[y0:y1, x0:x1]
        return float(np.mean(roi))

    top_m = mean_band(tb_y0, tb_y1, tb_x0, tb_x1)
    bot_m = mean_band(bb_y0, bb_y1, bb_x0, bb_x1)
    left_m = mean_band(lb_y0, lb_y1, lb_x0, lb_x1)
    right_m = mean_band(rb_y0, rb_y1, rb_x0, rb_x1)

    # lower mean => darker => border line present
    # pick two darkest
    vals = {'top': top_m, 'bottom': bot_m, 'left': left_m, 'right': right_m}
    sides = sorted(vals.keys(), key=lambda k: vals[k])[:2]
    sides = set(sides)
    if {'top','right'} == sides: return 'tr'
    if {'top','left'}  == sides: return 'tl'
    if {'bottom','right'} == sides: return 'br'
    if {'bottom','left'}  == sides: return 'bl'
    # fallback: choose side with darkest, pair with next best; default 'tr'
    return 'tr'

def propose_square(mcorners, inferred_corner, marker_frac, pad_frac, W, H, fudge_frac=0.0):
    xs = mcorners[:,0]; ys = mcorners[:,1]
    left, right = xs.min(), xs.max()
    top, bottom = ys.min(), ys.max()
    mw = right - left; mh = bottom - top
    ms = (mw + mh) * 0.5
    if ms <= 1: return None

    S = ms / max(1e-6, marker_frac)
    pad_px = ms * (pad_frac / max(1e-6, marker_frac))

    if inferred_corner == "tr":
        x1 = right + pad_px; y0 = top - pad_px; x0 = x1 - S; y1 = y0 + S
    elif inferred_corner == "tl":
        x0 = left - pad_px; y0 = top - pad_px; x1 = x0 + S; y1 = y0 + S
    elif inferred_corner == "br":
        x1 = right + pad_px; y1 = bottom + pad_px; x0 = x1 - S; y0 = y1 - S
    else:
        x0 = left - pad_px; y1 = bottom + pad_px; x1 = x0 + S; y0 = y1 - S

    if fudge_frac != 0.0:
        f = S * fudge_frac; x0 += f; y0 += f; x1 -= f; y1 -= f

    x0 = clamp(round(x0), 0, W); x1 = clamp(round(x1), 0, W)
    y0 = clamp(round(y0), 0, H); y1 = clamp(round(y1), 0, H)
    if x1-x0 < 10 or y1-y0 < 10: return None
    return np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.float32)

def warp_square(img, poly):
    # order as TL,TR,BR,BL
    pts = poly.astype(np.float32)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).ravel()
    tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
    ordered = np.array([tl,tr,br,bl], dtype=np.float32)
    w = int(round((np.linalg.norm(ordered[1]-ordered[0]) + np.linalg.norm(ordered[2]-ordered[3]))/2))
    h = int(round((np.linalg.norm(ordered[3]-ordered[0]) + np.linalg.norm(ordered[2]-ordered[1]))/2))
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(img, M, (w, h))

def remove_marker_region(warped, corner, marker_frac, pad_frac, strategy):
    H, W = warped.shape[:2]
    m = int(round(min(W,H)*marker_frac)); p = int(round(min(W,H)*pad_frac))
    if corner == "tr":
        x0, y0, x1, y1 = W-m-p, 0+p, W-p, 0+p+m
    elif corner == "tl":
        x0, y0, x1, y1 = 0+p, 0+p, 0+p+m, 0+p+m
    elif corner == "br":
        x0, y0, x1, y1 = W-m-p, H-m-p, W-p, H-p
    else:
        x0, y0, x1, y1 = 0+p, H-m-p, 0+p+m, H-p
    x0 = max(0, min(W, x0)); x1 = max(0, min(W, x1))
    y0 = max(0, min(H, y0)); y1 = max(0, min(H, y1))
    if x1<=x0 or y1<=y0: return warped
    if strategy=="whiteout":
        warped[y0:y1, x0:x1] = 255; return warped
    if strategy=="inpaint":
        roi = warped[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        th = cv2.dilate(th, np.ones((3,3), np.uint8), iterations=1)
        warped[y0:y1, x0:x1] = cv2.inpaint(roi, th, 3, cv2.INPAINT_TELEA); return warped
    if strategy=="crop":
        if corner=="tr": return warped[y1:H, 0:W]
        if corner=="tl": return warped[y1:H, 0:W]
        if corner=="br": return warped[0:H, 0:x0]
        if corner=="bl": return warped[0:H, x1:W]
    return warped

def clean_digits(img_or_path,
                        area_min_ratio: float = 0.0015,
                        top_right_box: tuple = (0.70, 0.40),
                        dilate_px: int = 1,
                        clahe_clip: float = 3.0,
                        clahe_grid: int = 8):
    """
    More robust cleaner for faint/low-contrast digits.
    - CLAHE -> adaptive threshold -> (optional) 1px dilation
    - Remove frame (components touching border)
    - Remove ArUco (components near top-right; can adjust box)
    Returns: uint8 image, digits black (0) on white (255).
    """
    # --- load image ---
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path)
        if img is None:
            raise FileNotFoundError(img_or_path)
    else:
        img = np.asarray(img_or_path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]

    # --- grayscale + local contrast (CLAHE) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    g = clahe.apply(gray)

    # --- adaptive threshold (good for faint pencil) ---
    # Note: subtract C makes it a bit more aggressive; tweak if needed.
    bw = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # digits become white in the intermediate
        blockSize=max(11, (min(h, w)//9)*2+1),  # odd, scales with image size
        C=5
    )
    # bw: digits=255, background=0 (we inverted). Continue using this as the "ink" mask.

    # Optional: thicken faint strokes slightly
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
        bw = cv2.dilate(bw, k, iterations=1)

    # --- connected components on ink (white=255 here) ---
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    area_min = max(5, int(area_min_ratio * (h * w)))
    x_cut, y_cut = top_right_box

    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        cx, cy = centroids[i]

        if area < area_min:
            continue

        # drop anything touching image border (the frame)
        touches_border = (x == 0) or (y == 0) or (x + ww >= w) or (y + hh >= h)
        if touches_border:
            continue

        # drop components in the top-right box (AruCo region)
        in_top_right = (cx > x_cut * w) and (cy < y_cut * h)
        if in_top_right:
            # extra check: likely square & dense (helps if digits happen to cross that box)
            aspect = ww / float(hh + 1e-6)
            fill = area / float(ww * hh + 1e-6)
            if 0.6 <= aspect <= 1.6 and fill > 0.25:
                continue
        keep[i] = True

    # Fallback: keep the two largest non-border components if nothing survived
    if not np.any(keep):
        candidates = []
        for i in range(1, n):
            x, y, ww, hh, area = stats[i]
            if not ((x == 0) or (y == 0) or (x + ww >= w) or (y + hh >= h)):
                candidates.append((area, i))
        candidates.sort(reverse=True)
        for _, idx in candidates[:2]:
            keep[idx] = True

    # Rebuild clean mask
    clean_mask = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        if keep[i]:
            clean_mask[labels == i] = 255

    # Convert to digits black on white
    result = 255 - clean_mask
    return result

def _sauvola_threshold(gray, win: int, k: float = 0.34, R: float = 128.0):
    """
    Compute Sauvola threshold map using box filters (no external deps).
    gray: uint8 grayscale, win: odd window size.
    Returns a uint8 binary (255=foreground) with digits as white.
    """
    # Convert to float for precision
    g = gray.astype(np.float32)
    # local mean
    m = cv2.boxFilter(g, ddepth=-1, ksize=(win, win), borderType=cv2.BORDER_REPLICATE)
    # local mean of squares
    g2 = g * g
    m2 = cv2.boxFilter(g2, ddepth=-1, ksize=(win, win), borderType=cv2.BORDER_REPLICATE)
    # local std-dev
    s = cv2.sqrt(cv2.max(m2 - m * m, 0))
    # Sauvola threshold
    T = m * (1.0 + k * (s / R - 1.0))
    # Foreground = darker than threshold (pencil)
    bin_img = (g <= T).astype(np.uint8) * 255
    return bin_img

def digits_for_ocr_v11b(
    img_or_path,
    # denoise / contrast
    use_mild_clahe: bool = True, clahe_clip: float = 1.5, clahe_grid: int = 8,
    pre_blur_sigma: float = 0.8,
    # Sauvola threshold params
    window_frac: float = 0.11, sauvola_k: float = 0.25, sauvola_R: float = 128.0,
    # component pruning
    area_min_ratio: float = 0.002, min_dim_ratio: float = 0.05,
    center_keep_frac: float = 0.85, top_right_box=(0.70, 0.40),
    keep_max_components: int = 2,
    # NEW: bridge cut (break 1–2 px necks, remove small lobes)
    bridge_kernel: int = 3,          # 3x3 ellipse by default (targets 1px links)
    bridge_iters: int = 1,           # increase to 2 only if needed
    # small crumb cleanup
    final_open_ksize: int = 0,
    # output
    out_size: int = 96, pad_frac: float = 0.10
):
    """
    OCR-ready cleaner with Sauvola binarization + bridge-cut to remove
    blobs connected by hairline bridges. Returns black-on-white uint8.
    """
    # --- load & grayscale ---
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path)
        if img is None: raise FileNotFoundError(img_or_path)
    else:
        img = np.asarray(img_or_path)
        if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # mild CLAHE + denoise
    if use_mild_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
        gray = clahe.apply(gray)
    if pre_blur_sigma and pre_blur_sigma > 0:
        k = int(max(3, round(pre_blur_sigma * 6) | 1))
        gray = cv2.GaussianBlur(gray, (k, k), pre_blur_sigma, borderType=cv2.BORDER_REPLICATE)

    # Sauvola
    win = int(max(11, round(window_frac * min(H, W))))
    if win % 2 == 0: win += 1
    bw = _sauvola_threshold(gray, win=win, k=sauvola_k, R=sauvola_R)  # digits=255

    # --- remove frame + ArUco by components ---
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, 8)
    area_min   = max(12, int(area_min_ratio * (H * W)))
    dim_min_px = max(4, int(min_dim_ratio * min(H, W)))
    cx_img, cy_img = W/2.0, H/2.0
    max_center_dist = 0.5 * (W**2 + H**2) ** 0.5 * center_keep_frac
    x_cut, y_cut = top_right_box

    survivors = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]; cx, cy = centroids[i]
        if area < area_min: continue
        if w < dim_min_px or h < dim_min_px: continue
        if x == 0 or y == 0 or (x+w) >= W or (y+h) >= H: continue
        if ((cx-cx_img)**2 + (cy-cy_img)**2) ** 0.5 > max_center_dist: continue
        if (cx > x_cut*W) and (cy < y_cut*H):
            extent = area / float(w*h + 1e-6); aspect = w / float(h + 1e-6)
            if extent > 0.25 and 0.6 <= aspect <= 1.6: continue
        survivors.append((area, i))
    survivors.sort(reverse=True)
    survivors = [i for _, i in survivors[:keep_max_components]]

    clean = np.zeros((H, W), np.uint8)
    for i in survivors: clean[labels == i] = 255
    if clean.max() == 0:
        return 255*np.ones((out_size, out_size), np.uint8)

    # ================== BRIDGE CUT ==================
    # Erode to break 1–2 px necks, then keep the largest pieces only,
    # then dilate back and intersect with original mask (preserve width).
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_kernel, bridge_kernel))
    eroded = cv2.erode(clean, k, iterations=bridge_iters)

    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(eroded, 8)
    parts = [(st2[i, cv2.CC_STAT_AREA], i) for i in range(1, n2)]
    parts.sort(reverse=True)
    keep_ids = {i for _, i in parts[:keep_max_components]}  # keep 1–2 biggest only

    kept_eroded = np.zeros_like(eroded)
    for i in keep_ids:
        kept_eroded[lab2 == i] = 255

    restored = cv2.dilate(kept_eroded, k, iterations=bridge_iters)
    clean = cv2.bitwise_and(restored, clean)  # never grow beyond original
    # =================================================

    # optional tiny opening for 1px crumbs
    if final_open_ksize and final_open_ksize > 1:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (final_open_ksize, final_open_ksize))
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, k2, iterations=1)

    # --- crop, center, resize ---
    ys, xs = np.where(clean > 0)
    if ys.size == 0 or xs.size == 0:
        return 255*np.ones((out_size, out_size), np.uint8)
    y0, y1 = ys.min(), ys.max()+1; x0, x1 = xs.min(), xs.max()+1
    crop = clean[y0:y1, x0:x1]
    h, w = crop.shape; pad = int(max(h, w) * pad_frac); side = max(h, w) + 2*pad
    canvas = np.zeros((side, side), np.uint8)
    canvas[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = crop
    canvas = 255 - canvas
    out = cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="out")
    ap.add_argument("--dict", default="5X5_100")
    ap.add_argument("--markers-only", action="store_true", help="Skip contour detection; build squares from markers only")
    ap.add_argument("--marker-frac", type=float, default=0.28)
    ap.add_argument("--pad-frac", type=float, default=0.04)
    ap.add_argument("--fallback-fudge-frac", type=float, default=0.0)
    ap.add_argument("--remove-marker", action="store_true")
    ap.add_argument("--remove-strategy", choices=["whiteout","inpaint","crop"], default="whiteout")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--clean", action="store_true", default=True)
    ap.add_argument("--save-results", action="store_true", default=False)
    ap.add_argument("--testclean", action="store_true", default=False, help="Run the cleaner, but save as TEST_*.png instead of OCR_*")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    img = cv2.imread(args.image)
    if img is None: raise SystemExit(f"Cannot read {args.image}")
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detections = detect_markers(img, args.dict)
    if not detections:
        raise SystemExit("No markers detected. Try better lighting or a larger dictionary.")

    vis = img.copy()
    for mid, mc in detections:
        # infer corner from dark borders around the marker
        l, t = mc[:,0].min(), mc[:,1].min()
        r, b = mc[:,0].max(), mc[:,1].max()
        inferred = infer_corner_from_borders(gray, (l,t,r,b))
        poly = propose_square(mc, inferred, args.marker_frac, args.pad_frac, W, H, args.fallback_fudge_frac)
        if poly is None:
            continue

        warped = warp_square(img, poly)

        # save w/ ID (no need to detect marker again after warp)
        out_img = warped.copy()
        if args.remove_marker:
            out_img = remove_marker_region(out_img, inferred, args.marker_frac, args.pad_frac, args.remove_strategy)

        if args.save_results:
            cv2.imwrite(os.path.join(args.out, f"{mid}.png"), out_img)

        if args.clean:
            out_img = clean_digits(out_img)
            out_img = digits_for_ocr_v11b(out_img)
            cv2.imwrite(os.path.join(args.out, f"OCR_{mid}.png"), out_img)

        #if args.testclean:
        #   out_img = digits_for_ocr_v11b(out_img)
        #    cv2.imwrite(os.path.join(args.out, f"TEST_V11b_{mid}.png"), out_img)

        if args.debug:
            pts = poly.reshape(-1,1,2).astype(int)
            cv2.polylines(vis, [pts], True, (0,0,255), 3)
            tl = tuple(poly[0].astype(int))
            cv2.putText(vis, f"ID:{mid} {inferred}", tl, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

    if args.debug:
        cv2.imwrite(os.path.join(args.out, "debug_vis.png"), vis)

if __name__ == "__main__":
    main()
