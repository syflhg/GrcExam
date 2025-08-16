import os, io, csv, time, glob, pathlib
import numpy as np
import cv2
import streamlit as st
import detect_squares as DS  # this is the file you uploaded
import base64

# Optional: quiet TF logs before importing TF
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# --------------------------
# Model + OCR dependencies
# --------------------------
from typing import List, Tuple
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from filelock import FileLock
import pytesseract

import os
PERSIST_DIR = "/data" if os.environ.get("SYSTEM") == "spaces" else "."
MODEL_USER_PATH = os.path.join(PERSIST_DIR, "mnist_cnn_user.h5")
MODEL_BASE_PATH = os.path.join(PERSIST_DIR, "mnist_cnn.h5")
FEEDBACK_DIR    = os.path.join(PERSIST_DIR, "feedback")
LOCK_PATH       = os.path.join(PERSIST_DIR, "train.lock")

# =========================
# Feedback manager
# =========================
class FeedbackManager:
    def __init__(self, root_dir="feedback"):
        self.root = pathlib.Path(root_dir)
        self.img_dir = self.root / "digits"
        self.meta_csv = self.root / "labels.csv"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        if not self.meta_csv.exists():
            with open(self.meta_csv, "w", newline="") as f:
                csv.writer(f).writerow(["file","label","src_image","index","ts"])

    def save_digits(self, src_image_name: str, digit_patches28, truth_str: str):
        assert len(digit_patches28) == len(truth_str), \
            f"patches={len(digit_patches28)} truth={len(truth_str)}"
        saved = []
        ts = int(time.time())
        for i, (patch, ch) in enumerate(zip(digit_patches28, truth_str)):
            fn = f"d_{ts}_{i}_{ch}.png"
            path = self.img_dir / fn
            patch_u8 = (np.clip(patch,0,1) * 255).astype(np.uint8)
            cv2.imwrite(str(path), patch_u8)
            saved.append((fn, ch, src_image_name, i, ts))
        with open(self.meta_csv, "a", newline="") as f:
            csv.writer(f).writerows(saved)
        return len(saved)

# =========================
# Model (load or train)
# =========================
def get_or_train_model(weights_path="mnist_cnn_user.h5", base_path="mnist_cnn.h5"):
    # prefer user-tuned; fall back to base; else train base
    if os.path.exists(weights_path):
        return models.load_model(weights_path), weights_path
    if os.path.exists(base_path):
        return models.load_model(base_path), base_path

    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        layers.Conv2D(64, 3, activation="relu"),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    (Xtr, ytr), (Xte, yte) = mnist.load_data()
    Xtr = (Xtr.astype("float32")/255.0)[..., None]
    Xte = (Xte.astype("float32")/255.0)[..., None]
    ytr = to_categorical(ytr, 10); yte = to_categorical(yte, 10)
    model.fit(Xtr, ytr, epochs=5, batch_size=128, validation_data=(Xte, yte), verbose=2)
    model.save(base_path)
    return model, base_path

# =========================
# Preprocess & segmentation
# =========================
def preprocess_page(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 25, 8)
    # comment next 2 lines if no top-right marker in your layout
    #h, w = thr.shape
    #thr[0:int(0.25*h), w-int(0.25*w):w] = 0
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    return thr

def find_digit_boxes(thr: np.ndarray) -> List[Tuple[int,int,int,int]]:
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w*h < 80 or h < 12 or w < 6:
            continue
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])
    return boxes

def roi_to_28x28(thr: np.ndarray, box: Tuple[int,int,int,int]) -> np.ndarray:
    x, y, w, h = box
    roi = thr[y:y+h, x:x+w]
    s = max(w, h)
    canvas = np.zeros((s, s), dtype=np.uint8)
    ox, oy = (s - w)//2, (s - h)//2
    canvas[oy:oy+h, ox:ox+w] = roi
    digit = cv2.resize(canvas, (20, 20), interpolation=cv2.INTER_AREA)
    digit = cv2.copyMakeBorder(digit, 4,4,4,4, cv2.BORDER_CONSTANT, value=0)
    digit = (digit.astype("float32")/255.0)[..., None]  # (28,28,1)
    return digit

# =========================
# CNN prediction (with TTA)
# =========================
def augment_digit(img28: np.ndarray) -> np.ndarray:
    base = (img28*255).astype(np.uint8)
    outs = []
    for dx, dy in [(-1,0),(0,0),(1,0),(0,-1),(0,1)]:
        M = np.float32([[1,0,dx],[0,1,dy]])
        img = cv2.warpAffine(base, M, (28,28), flags=cv2.INTER_NEAREST, borderValue=0)
        outs.append(img.astype("float32")/255.0)
    outs = [o[...,None] for o in outs]
    return np.stack(outs, axis=0)

def predict_digit(model, img28: np.ndarray):
    X = augment_digit(img28)
    probs = model.predict(X, verbose=0).mean(axis=0)
    idx = int(np.argmax(probs))
    p1 = float(probs[idx])
    top2 = float(np.partition(probs, -2)[-2])
    margin = p1 - top2
    return idx, p1, margin

def cnn_predict_with_patches(model, thr, boxes):
    digits, p1s, margins, patches = [], [], [], []
    for b in boxes:
        d28 = roi_to_28x28(thr, b)
        patches.append(d28[...,0])
        d, p1, m = predict_digit(model, d28)
        digits.append(str(d)); p1s.append(p1); margins.append(m)
    return digits, p1s, margins, patches

# =========================
# OCR per-character
# =========================
def ocr_digits_with_conf_charwise(img_bgr: np.ndarray, thr_mask: np.ndarray):
    ys, xs = np.where(thr_mask > 0)
    if len(xs) == 0: return [], []
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    pad = 6; h, w = thr_mask.shape
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)
    crop = img_bgr[y1:y2+1, x1:x2+1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.medianBlur(bw, 3)

    boxes_str = pytesseract.image_to_boxes(
        bw, config="--psm 7 -c tessedit_char_whitelist=0123456789"
    )
    if not boxes_str.strip(): return [], []

    H, W = bw.shape[:2]
    char_boxes = []
    for line in boxes_str.strip().splitlines():
        parts = line.split()
        ch = parts[0]
        if not ch.isdigit(): continue
        xL,yB,xR,yT = map(int, parts[1:5])
        yT_img = H - yT; yB_img = H - yB
        char_boxes.append((ch, xL, yT_img, xR, yB_img))
    char_boxes.sort(key=lambda b: b[1])

    out_digits, out_conf = [], []
    for ch_pred, xL, yT_img, xR, yB_img in char_boxes:
        pad2 = 1
        xL2 = max(0, xL-pad2); yT2 = max(0, yT_img-pad2)
        xR2 = min(W-1, xR+pad2); yB2 = min(H-1, yB_img+pad2)
        char_crop = bw[yT2:yB2+1, xL2:xR2+1]

        data = pytesseract.image_to_data(
            char_crop,
            config="--psm 10 -c tessedit_char_whitelist=0123456789",
            output_type=pytesseract.Output.DICT
        )
        best_digit, best_conf = None, -1.0
        for txt, conf in zip(data["text"], data["conf"]):
            if not txt: continue
            for d in txt:
                if d.isdigit():
                    try: c = float(conf)
                    except: c = -1.0
                    if c > best_conf:
                        best_digit, best_conf = d, c
        if best_digit is None:
            best_digit, best_conf = ch_pred, 0.0
        out_digits.append(best_digit)
        out_conf.append(max(0.0, best_conf/100.0))
    return out_digits, out_conf

# =========================
# Fusion logic
# =========================
def fuse_cnn_and_ocr(cnn_digits, cnn_p1, cnn_margin, ocr_digits, ocr_p1,
                     override_margin=0.07, ocr_min_abs=0.85, low_margin=0.15):
    confusable = {('4','6'), ('6','4'), ('1','7'), ('7','1'), ('5','9'), ('9','5')}
    L = max(len(cnn_digits), len(ocr_digits))
    pad = lambda arr, val: list(arr) + [val]*(L - len(arr))
    cnn_digits = pad(cnn_digits, '?')
    cnn_p1     = pad(cnn_p1, 0.0)
    cnn_margin = pad(cnn_margin, 0.0)
    ocr_digits = pad(ocr_digits, '?')
    ocr_p1     = pad(ocr_p1, 0.0)
    fused_d, fused_c = [], []
    for cd, cp, cm, od, op in zip(cnn_digits, cnn_p1, cnn_margin, ocr_digits, ocr_p1):
        if (cd, od) in confusable and od != '?' and op >= ocr_min_abs:
            fused_d.append(od); fused_c.append(op); continue
        if cm < low_margin and od != '?' and op >= ocr_min_abs and op >= cp:
            fused_d.append(od); fused_c.append(op); continue
        if od != '?' and (op - cp) >= override_margin and op >= ocr_min_abs:
            fused_d.append(od); fused_c.append(op); continue
        fused_d.append(cd); fused_c.append(cp)
    final = "".join(d for d in fused_d if d != '?')
    return final, fused_d, fused_c

# =========================
# Fine-tune on feedback
# =========================
def finetune_from_feedback(model_path="mnist_cnn.h5",
                           user_model_path="mnist_cnn_user.h5",
                           feedback_dir="feedback",
                           epochs=3, batch_size=64, lr=1e-4, replay_mnist=1000):
    with FileLock(LOCK_PATH, timeout=60):
        from tensorflow.keras import optimizers
        # load base or user model
        base = user_model_path if os.path.exists(user_model_path) else model_path
        model = models.load_model(base)

        # freeze convs; train dense head
        for layer in model.layers:
            layer.trainable = isinstance(layer, layers.Dense)

        fb_root = pathlib.Path(feedback_dir)
        meta_csv = fb_root / "labels.csv"
        if not meta_csv.exists():
            return None, "No feedback yet."

        X_fb, y_fb = [], []
        with open(meta_csv, "r") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                p = fb_root / "digits" / row["file"]
                if not p.exists(): continue
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                X_fb.append(img.astype("float32")/255.0)
                y_fb.append(int(row["label"]))
        if not X_fb:
            return None, "Feedback CSV has no images."

        X_fb = np.stack(X_fb, axis=0)[...,None]
        y_fb = to_categorical(np.array(y_fb, dtype=np.int32), 10)

        (Xtr, ytr), _ = mnist.load_data()
        idx = np.random.choice(len(Xtr), size=min(replay_mnist, len(Xtr)), replace=False)
        Xr = (Xtr[idx].astype("float32")/255.0)[...,None]
        yr = to_categorical(ytr[idx], 10)

        X = np.concatenate([X_fb, Xr], axis=0)
        y = np.concatenate([y_fb, yr], axis=0)

        model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
        hist = model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)
        model.save(user_model_path)
        return user_model_path, f"Trained {epochs} epoch(s). Final acc: {hist.history['accuracy'][-1]:.3f}"

def run_pipeline_on_image_bytes(model, fb_mgr, img_bytes, filename,
                                override_margin=0.07, ocr_min_abs=0.85, low_margin=0.15):
    img_np = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    thr = preprocess_page(img_bgr)
    boxes = find_digit_boxes(thr)
    cnn_digits, cnn_p1, cnn_marg, patches28 = cnn_predict_with_patches(model, thr, boxes)
    ocr_digits, ocr_p1 = ocr_digits_with_conf_charwise(img_bgr, thr)
    final, fused_digits, fused_conf = fuse_cnn_and_ocr(
        cnn_digits, cnn_p1, cnn_marg, ocr_digits, ocr_p1,
        override_margin=override_margin, ocr_min_abs=ocr_min_abs, low_margin=low_margin
    )
    # return everything we need for UI + feedback
    return {
        "filename": filename,
        "final": final,
        "cnn": list(zip(cnn_digits, [float(x) for x in cnn_p1], [float(x) for x in cnn_marg])),
        "ocr": list(zip(ocr_digits, [float(x) for x in ocr_p1])),
        "fused": list(zip(fused_digits, [float(x) for x in fused_conf])),
        "patches28": patches28,
        "img_preview": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
    }

def extract_card_crops_from_page(
    page_bgr,
    dict_name="5X5_100",           # same as your CLI default
    marker_frac=0.28,              # design ratio in your script
    pad_frac=0.04,                 # design ratio in your script
    remove_marker=True,            # hide the ArUco in the corner
    remove_strategy="whiteout",    # or "inpaint" / "crop"
    clean=True,                    # run your cleaner
    return_clean=True              # return the cleaned 96x96; if False returns the warped color crop
):
    H, W = page_bgr.shape[:2]
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)

    dets = DS.detect_markers(page_bgr, dict_name)
    crops = []

    for mid, mcorners in dets:
        # infer the corner from outer black frame
        l, t = mcorners[:,0].min(), mcorners[:,1].min()
        r, b = mcorners[:,0].max(), mcorners[:,1].max()
        inferred = DS.infer_corner_from_borders(gray, (l, t, r, b))

        poly = DS.propose_square(mcorners, inferred, marker_frac, pad_frac, W, H)
        if poly is None:
            continue

        warped = DS.warp_square(page_bgr, poly)  # upright card
        # Your two-step cleaner (digits black on white -> OCR-ready 96x96 white background)
        w2 = DS.clean_digits(warped)
        ocr_img = DS.digits_for_ocr_v11b(w2)
        if return_clean:
            # Our pipeline expects color; expand to 3 channels for display
            crops.append((mid, cv2.cvtColor(ocr_img, cv2.COLOR_GRAY2BGR)))
        else:
            crops.append((mid, warped))

    # sort deterministically (by marker id)
    crops.sort(key=lambda x: x[0])
    # return just the images, plus ids if you need them
    return crops  # list of (marker_id, crop_bgr)

def run_pipeline_on_bgr_array(model, img_bgr, filename,
                              override_margin=0.07, ocr_min_abs=0.85, low_margin=0.15):
    thr = preprocess_page(img_bgr)
    boxes = find_digit_boxes(thr)
    cnn_digits, cnn_p1, cnn_marg, patches28 = cnn_predict_with_patches(model, thr, boxes)
    ocr_digits, ocr_p1 = ocr_digits_with_conf_charwise(img_bgr, thr)
    final, fused_digits, fused_conf = fuse_cnn_and_ocr(
        cnn_digits, cnn_p1, cnn_marg, ocr_digits, ocr_p1,
        override_margin=override_margin, ocr_min_abs=ocr_min_abs, low_margin=low_margin
    )
    return {
        "filename": filename,
        "final": final,
        "cnn": list(zip(cnn_digits, [float(x) for x in cnn_p1], [float(x) for x in cnn_marg])),
        "ocr": list(zip(ocr_digits, [float(x) for x in ocr_p1])),
        "fused": list(zip(fused_digits, [float(x) for x in fused_conf])),
        "patches28": patches28,
        "img_preview": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
    }

def download_file_button(file_path, label="⬇️ Download trained model"):
    p = pathlib.Path(file_path)
    if not p.exists():
        st.warning("No trained model yet.")
        return
    b64 = base64.b64encode(p.read_bytes()).decode()
    href = f'<a href="data:file/h5;base64,{b64}" download="{p.name}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Digit Reader (CNN+OCR Fusion)", page_icon="🔢", layout="centered")

if "model" not in st.session_state:
    model, used_path = get_or_train_model(weights_path=MODEL_USER_PATH, base_path=MODEL_BASE_PATH)
    st.session_state.model = model
    st.session_state.model_path = used_path
if "fb" not in st.session_state:
    st.session_state.fb = FeedbackManager(FEEDBACK_DIR)

st.title("🔢 Digit Reader — CNN + OCR (Fusion) with Feedback")

with st.sidebar:
    st.header("Settings")
    override_margin = st.slider("OCR override margin", 0.0, 0.25, 0.07, 0.01)
    ocr_min_abs     = st.slider("OCR min confidence", 0.50, 0.99, 0.85, 0.01)
    low_margin      = st.slider("CNN 'decisive' margin", 0.00, 0.40, 0.15, 0.01)
    st.caption("OCR only overrides a CNN digit when it beats it by the margin and has sufficient absolute confidence. "
               "If CNN margin is small, the higher confidence wins.")
    
    download_file_button(MODEL_USER_PATH)

    if st.button("🧠 Fine-tune from feedback"):
        with st.spinner("Fine-tuning…"):
            out_path, msg = finetune_from_feedback(model_path=MODEL_BASE_PATH, user_model_path=MODEL_USER_PATH, feedback_dir=FEEDBACK_DIR)
            if out_path:
                st.success(msg)
                st.session_state.model = models.load_model(out_path)
                st.session_state.model_path = out_path
            else:
                st.warning(msg)

tab_multi, tab_batch = st.tabs(["Ready To Read Images", "Auto Catch Image"])

with tab_multi:
    files = st.file_uploader("Drop multiple images", type=["png","jpg","jpeg"], accept_multiple_files=True)
    if files:
        # cache results for this session so re-runs don’t recompute on every keystroke
        if "batch_cache" not in st.session_state: st.session_state.batch_cache = {}
        results = []
        for f in files:
            key = f"{f.name}|{len(f.getvalue())}"
            if key not in st.session_state.batch_cache:
                st.session_state.batch_cache[key] = run_pipeline_on_image_bytes(
                    st.session_state.model, st.session_state.fb,
                    f.getvalue(), f.name,
                    override_margin=override_margin, ocr_min_abs=ocr_min_abs, low_margin=low_margin
                )
            results.append(st.session_state.batch_cache[key])

        st.write(f"Found **{len(results)}** images.")
        # editable corrections per row
        corrected = []
        for i, r in enumerate(results):
            with st.container(border=True):
                cols = st.columns([1.2, 2.0, 2.2])
                with cols[0]:
                    st.image(r["img_preview"], width=120, caption=r["filename"])
                with cols[1]:
                    st.markdown(f"**Pred:** `{r['final']}`")
                    st.caption("CNN: " + (" ".join([f"{d}:{p:.2f}|m:{m:.2f}" for d,p,m in r["cnn"]]) or "-"))
                    st.caption("OCR: " + (" ".join([f"{d}:{p:.2f}" for d,p in r['ocr']]) or "-"))
                with cols[2]:
                    default_gt = r["final"]
                    gt = st.text_input("Correction (GT)", value=default_gt, key=f"gt_{i}")
                    sel = st.checkbox("Save feedback for this image", value=True, key=f"sel_{i}")
                    corrected.append((r, gt, sel))

        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("💾 Save feedback for selected"):
                total = 0
                skipped = 0
                for r, gt, sel in corrected:
                    if not sel: 
                        skipped += 1
                        continue
                    n = min(len(gt), len(r["patches28"]))
                    if n <= 0:
                        continue
                    try:
                        total += st.session_state.fb.save_digits(r["filename"], r["patches28"][:n], gt[:n])
                    except AssertionError:
                        pass
                st.success(f"Saved {total} digit patches. (skipped {skipped} images)")
        with c2:
            if st.button("🧠 Fine-tune now"):
                out_path, msg = finetune_from_feedback()
                if out_path:
                    st.success(msg)
                    st.session_state.model = models.load_model(out_path)
                    st.session_state.model_path = out_path
                else:
                    st.warning(msg)
with tab_batch:
    st.caption("Upload a full-page photo; we’ll detect the small squares, clean them, then recognize each.")
    uploaded_full = st.file_uploader("Full page image", type=["png","jpg","jpeg"], key="fullpage")

    if uploaded_full:
        page_np = np.frombuffer(uploaded_full.read(), np.uint8)
        page_bgr = cv2.imdecode(page_np, cv2.IMREAD_COLOR)

        # 1) YOUR cropper → list of (id, crop_bgr)
        crops = extract_card_crops_from_page(
            page_bgr,
            return_clean=True,    # keep True to feed the cleaned 96x96 to recognizer
        )
        st.write(f"Detected **{len(crops)}** squares.")

        rows = []
        for idx, (mid, crop_bgr) in enumerate(crops):
            # 2) Recognize each crop via our fusion pipeline
            res = run_pipeline_on_bgr_array(
                st.session_state.model, crop_bgr, f"{uploaded_full.name}#ID{mid}",
                override_margin=override_margin, ocr_min_abs=ocr_min_abs, low_margin=low_margin
            )
            rows.append((mid, res, crop_bgr))

        # 3) Show results + allow correction and feedback save
        saves = []
        for mid, res, crop_bgr in rows:
            with st.container(border=True):
                cols = st.columns([1.2, 2.2, 2.6])
                with cols[0]:
                    st.image(res["img_preview"], caption=f"ID {mid}", width=150)
                with cols[1]:
                    st.markdown(f"**Pred:** `{res['final']}`")
                    st.caption("CNN: " + (" ".join([f"{d}:{p:.2f}|m:{m:.2f}" for d,p,m in res["cnn"]]) or "-"))
                    st.caption("OCR: " + (" ".join([f"{d}:{p:.2f}" for d,p in res["ocr"]]) or "-"))
                with cols[2]:
                    gt = st.text_input(f"Correction (ID {mid})", value=res["final"], key=f"full_gt_{mid}")
                    sel = st.checkbox(f"Save feedback (ID {mid})", value=True, key=f"full_sel_{mid}")
                    saves.append((res, gt, sel, f"{uploaded_full.name}#ID{mid}"))

        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("💾 Save feedback for selected (full page)"):
                total = 0
                for res, gt, sel, fname in saves:
                    if not sel: continue
                    n = min(len(gt), len(res["patches28"]))
                    if n > 0:
                        total += st.session_state.fb.save_digits(fname, res["patches28"][:n], gt[:n])
                st.success(f"Saved {total} digit patches from this page.")
        with c2:
            if st.button("🧠 Fine-tune now (full page)"):
                out_path, msg = finetune_from_feedback()
                if out_path:
                    st.success(msg)
                    from tensorflow.keras import models as _mdl
                    st.session_state.model = _mdl.load_model(out_path)
                    st.session_state.model_path = out_path
                else:
                    st.warning(msg)


