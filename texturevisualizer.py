import io
import csv
import zipfile
import hashlib
import traceback
from typing import List, Tuple, Optional

import numpy as np
import cv2
import streamlit as st
from PIL import Image


# =========================
# Page setup
# =========================
st.set_page_config(page_title="Textured Paint Recolor", page_icon="🎨", layout="wide")
st.title("🎨 Textured Paint Recolor (Shading Map Method + Batch Exports)")


# =========================
# Constants (no sliders)
# =========================
# You said outputs are mostly too light -> keep this
OVERALL_ANCHOR_BIAS = 0.94

# Highlight-aware overall impression anchoring
ANCHOR_LOW_PCT = 25
ANCHOR_HIGH_PCT = 85
ANCHOR_REF_PCT = 70

# Shading extraction robustness
SHADING_MID_LO = 20
SHADING_MID_HI = 80
SHADING_CLAMP_LO = 0.35
SHADING_CLAMP_HI = 2.20

# Stability caps for Streamlit Cloud
MAX_COLORS_PREVIEW = 40
MAX_COLORS_ZIP_5MB = 30
MAX_COLORS_ZIP_10MB = 18
MAX_ZIP_MB = 180  # rough safety limit for ZIP size to avoid browser/cloud issues


# =========================
# Robust upload load
# =========================
def load_uploaded_image_bytes(uploaded_file) -> Tuple[bytes, np.ndarray]:
    """Always load from bytes to avoid pointer/cache issues."""
    img_bytes = uploaded_file.getvalue()
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    return img_bytes, img

def sha1_hex(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()

def mean_rgb(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.reshape(-1, 3).mean(axis=0)


# =========================
# sRGB <-> Linear helpers
# =========================
def srgb_to_linear01(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    return np.where(x01 <= 0.04045, x01 / 12.92, ((x01 + 0.055) / 1.055) ** 2.4)

def linear_to_srgb01(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    return np.where(x01 <= 0.0031308, x01 * 12.92, 1.055 * (x01 ** (1 / 2.4)) - 0.055)


# =========================
# (Optional) reference calibration
# =========================
def robust_observed_rgb_midtone_median(img_u8: np.ndarray) -> np.ndarray:
    """
    Robust observed RGB (whole image):
    - compute luma
    - mask midtones (20–80%)
    - median RGB of masked region
    """
    img01 = img_u8.astype(np.float32) / 255.0
    Y = 0.2126 * img01[:, :, 0] + 0.7152 * img01[:, :, 1] + 0.0722 * img01[:, :, 2]
    p_lo, p_hi = np.percentile(Y, 20), np.percentile(Y, 80)
    mask = (Y >= p_lo) & (Y <= p_hi)
    if np.count_nonzero(mask) < 1000:
        mask = np.ones_like(Y, dtype=bool)

    med = np.array([
        np.median(img01[:, :, 0][mask]),
        np.median(img01[:, :, 1][mask]),
        np.median(img01[:, :, 2][mask]),
    ], dtype=np.float32)
    return med * 255.0

def compute_reference_gains(obs_rgb_u8: np.ndarray, true_rgb_u8: Tuple[int, int, int]) -> np.ndarray:
    """
    gains_lin = true_lin / observed_lin   (linear RGB)
    """
    obs01 = srgb_to_linear01(np.array(obs_rgb_u8, dtype=np.float32) / 255.0)
    tru01 = srgb_to_linear01(np.array(true_rgb_u8, dtype=np.float32) / 255.0)
    gains = tru01 / np.maximum(obs01, 1e-6)
    return np.clip(gains, 0.5, 2.0).astype(np.float32)

def apply_gains_to_target_rgb(target_rgb: Tuple[int, int, int], gains_lin: np.ndarray) -> Tuple[int, int, int]:
    """
    Apply gains to the requested target color (linear RGB), return corrected sRGB uint8.
    """
    t01 = np.array(target_rgb, dtype=np.float32) / 255.0
    t_lin = srgb_to_linear01(t01)
    t_lin_corr = np.clip(t_lin * gains_lin, 0.0, 1.0)
    t_corr = linear_to_srgb01(t_lin_corr)
    t_u8 = np.clip(t_corr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return (int(t_u8[0]), int(t_u8[1]), int(t_u8[2]))


# =========================
# Shading map method (remove color, keep texture)
# =========================
def build_shading_map_normalized(img_u8: np.ndarray) -> np.ndarray:
    """
    Build a normalized shading/texture map S in linear space:
      - Compute linear luminance Y
      - Reference Y_ref = median of midtones (20–80%)
      - S = Y / Y_ref  (so typical midtone shading = 1.0)
      - Clamp & smooth to reduce extreme specular/shadow effects
    Returns S shape (H,W,1), float32.
    """
    img01 = img_u8.astype(np.float32) / 255.0
    lin = srgb_to_linear01(img01)

    # Linear luminance
    Y = 0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2]

    # Midtone mask
    p_lo = np.percentile(Y, SHADING_MID_LO)
    p_hi = np.percentile(Y, SHADING_MID_HI)
    mid = (Y >= p_lo) & (Y <= p_hi)
    if np.count_nonzero(mid) < 1000:
        mid = np.ones_like(Y, dtype=bool)

    Y_ref = float(np.median(Y[mid]))
    Y_ref = max(Y_ref, 1e-6)

    S = (Y / Y_ref).astype(np.float32)

    # Clamp to reduce global drift from extreme lighting
    S = np.clip(S, SHADING_CLAMP_LO, SHADING_CLAMP_HI)

    # Smooth a bit but preserve edges/texture
    S = cv2.bilateralFilter(S, d=0, sigmaColor=0.08, sigmaSpace=5)

    return S[:, :, None]

def recolor_from_shading(img_u8: np.ndarray, target_rgb: Tuple[int, int, int]) -> np.ndarray:
    """
    Remove original color and re-apply requested target color using shading map.
    Output preserves texture and is stable across different input images.
    """
    S = build_shading_map_normalized(img_u8)  # (H,W,1)

    tgt01 = np.array(target_rgb, dtype=np.float32) / 255.0
    tgt_lin = srgb_to_linear01(tgt01)[None, None, :]  # (1,1,3)

    out_lin = np.clip(tgt_lin * S, 0.0, 1.0)
    out01 = linear_to_srgb01(out_lin)
    out_u8 = (np.clip(out01 * 255.0 + 0.5, 0, 255)).astype(np.uint8)
    return out_u8

def anchor_overall_impression(output_u8: np.ndarray, target_rgb: Tuple[int, int, int]) -> np.ndarray:
    """
    Fix cases where output is too light/dark overall by anchoring perceived brightness
    to target RGB. Uses highlight-aware reference and applies s *= 0.94.
    """
    img = output_u8.astype(np.float32) / 255.0
    Y = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]

    p_lo, p_hi = np.percentile(Y, ANCHOR_LOW_PCT), np.percentile(Y, ANCHOR_HIGH_PCT)
    mid = (Y >= p_lo) & (Y <= p_hi)
    if np.count_nonzero(mid) < 1000:
        mid = np.ones_like(Y, dtype=bool)

    y_ref = float(np.percentile(Y[mid], ANCHOR_REF_PCT))

    tr, tg, tb = [v / 255.0 for v in target_rgb]
    y_tgt = float(0.2126 * tr + 0.7152 * tg + 0.0722 * tb)

    s = (y_tgt / (y_ref + 1e-6)) * OVERALL_ANCHOR_BIAS
    s = float(np.clip(s, 0.70, 1.20))

    out = np.clip(img * s, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)

def process_color(base_img_u8: np.ndarray,
                  requested_target_rgb: Tuple[int, int, int],
                  use_calibration: bool,
                  gains_lin: Optional[np.ndarray]) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Pipeline:
      1) Optionally correct requested target using reference gains
      2) Shading-map recolor (paint target on texture)
      3) Overall impression anchor (fix too light/dark overall)
    Returns (output_image, used_target_rgb)
    """
    if use_calibration and gains_lin is not None:
        used_target = apply_gains_to_target_rgb(requested_target_rgb, gains_lin)
    else:
        used_target = requested_target_rgb

    out = recolor_from_shading(base_img_u8, used_target)
    out = anchor_overall_impression(out, used_target)
    return out, used_target


# =========================
# Export helpers (JPEG caps)
# =========================
def sanitize_filename(name: str) -> str:
    safe = []
    for ch in name:
        if ch.isalnum() or ch in (" ", "-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    s = "".join(safe).strip()
    s = "_".join(s.split())
    return s or "color"

def _jpeg_bytes(img: Image.Image, quality: int) -> bytes:
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=quality, optimize=True, progressive=True, subsampling=2)
    return b.getvalue()

def export_with_cap(pil_img: Image.Image, cap_bytes: int,
                    min_quality: int = 40, max_quality: int = 95,
                    max_downscales: int = 5) -> bytes:
    """
    Export JPEG bytes <= cap_bytes using binary search on quality,
    then downscale if necessary.
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    def try_compress(img: Image.Image) -> Optional[bytes]:
        lo, hi = min_quality, max_quality
        best = None
        for _ in range(8):
            q = (lo + hi) // 2
            data = _jpeg_bytes(img, q)
            if len(data) <= cap_bytes:
                best = data
                lo = q + 1
            else:
                hi = q - 1
        return best

    img = pil_img
    for _ in range(max_downscales + 1):
        best = try_compress(img)
        if best is not None:
            return best

        data_min = _jpeg_bytes(img, min_quality)
        ratio = float(np.sqrt(cap_bytes / max(len(data_min), 1)) * 0.95)
        ratio = float(max(min(ratio, 0.92), 0.60))

        new_w = max(64, int(img.width * ratio))
        new_h = max(64, int(img.height * ratio))
        if (new_w, new_h) == img.size:
            new_w = max(64, img.width - 64)
            new_h = max(64, img.height - 64)

        img = img.resize((new_w, new_h), resample=Image.LANCZOS)

    return _jpeg_bytes(img, min_quality)


# =========================
# Batch parsing
# =========================
def _safe_name_from_rgb(rgb: Tuple[int, int, int]) -> str:
    return f"{rgb[0]}-{rgb[1]}-{rgb[2]}"

def _parse_hex_token(tok: str) -> Optional[Tuple[int, int, int]]:
    s = tok.strip().lstrip("#")
    if len(s) in (3, 6):
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        try:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
            return (r, g, b)
        except ValueError:
            return None
    return None

def parse_rgb_lines(text: str) -> List[Tuple[str, Tuple[int, int, int]]]:
    colors = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.replace("\t", " ").split(",")]

        # name,r,g,b
        if len(parts) == 4 and all(parts):
            name = parts[0]
            try:
                r, g, b = int(parts[1]), int(parts[2]), int(parts[3])
                if all(0 <= v <= 255 for v in (r, g, b)):
                    colors.append((name, (r, g, b)))
                    continue
            except ValueError:
                pass

        # r,g,b
        if len(parts) == 3:
            try:
                r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                if all(0 <= v <= 255 for v in (r, g, b)):
                    colors.append((_safe_name_from_rgb((r, g, b)), (r, g, b)))
                    continue
            except ValueError:
                pass

        # hex anywhere
        words = line.replace(",", " ").split()
        hex_rgb = None
        for w in words:
            hex_rgb = _parse_hex_token(w)
            if hex_rgb:
                break

        if hex_rgb:
            name_tokens = [w for w in words if _parse_hex_token(w) is None]
            name = " ".join(name_tokens).strip() or _safe_name_from_rgb(hex_rgb)
            colors.append((name, hex_rgb))
            continue

    return colors

def parse_csv_file(file) -> List[Tuple[str, Tuple[int, int, int]]]:
    txt = file.read().decode("utf-8", errors="ignore")
    file.seek(0)
    rows = []
    rdr = csv.reader(io.StringIO(txt))
    for row in rdr:
        if not row:
            continue
        low = [c.strip().lower() for c in row[:5]]
        if any(x in ("r", "g", "b") for x in low):
            continue

        if len(row) >= 4:
            name = row[0].strip()
            try:
                r, g, b = int(row[1]), int(row[2]), int(row[3])
                if all(0 <= v <= 255 for v in (r, g, b)):
                    rows.append((name or _safe_name_from_rgb((r, g, b)), (r, g, b)))
            except ValueError:
                pass
        elif len(row) == 3:
            try:
                r, g, b = int(row[0]), int(row[1]), int(row[2])
                if all(0 <= v <= 255 for v in (r, g, b)):
                    rows.append((_safe_name_from_rgb((r, g, b)), (r, g, b)))
            except ValueError:
                pass
    return rows


# =========================
# Cached ZIP builder (one at a time)
# =========================
@st.cache_data(show_spinner=False)
def build_zip_for_cap(img_bytes: bytes,
                      colors: List[Tuple[str, Tuple[int, int, int]]],
                      cap_mb: int,
                      cache_salt: str,
                      use_calibration: bool,
                      gains_lin_tuple: Optional[Tuple[float, float, float]]) -> bytes:
    base_img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    gains_lin = np.array(gains_lin_tuple, dtype=np.float32) if (use_calibration and gains_lin_tuple is not None) else None

    cap_bytes = cap_mb * 1024 * 1024
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, rgb in colors:
            out, used_rgb = process_color(base_img, rgb, use_calibration, gains_lin)
            data = export_with_cap(Image.fromarray(out), cap_bytes)
            zf.writestr(f"{sanitize_filename(name)}.jpg", data)
    return zip_buf.getvalue()


# =========================
# UI
# =========================
uploaded = st.file_uploader("Upload a textured paint image", type=["jpg", "jpeg", "png"], key="uploader_shading")

if uploaded is None:
    st.info("Upload a photo to begin.")
    st.stop()

try:
    img_bytes, img = load_uploaded_image_bytes(uploaded)

    # ---- Reference calibration inputs ----
    st.subheader("Reference calibration (optional, but recommended for consistency)")
    c0, c1, c2, c3 = st.columns([1.4, 1, 1, 1])
    with c0:
        use_cal = st.checkbox("Use reference calibration", value=True)
    with c1:
        ref_r = st.number_input("Reference TRUE R", 0, 255, 213, 1)
    with c2:
        ref_g = st.number_input("Reference TRUE G", 0, 255, 224, 1)
    with c3:
        ref_b = st.number_input("Reference TRUE B", 0, 255, 220, 1)

    ref_true_rgb = (int(ref_r), int(ref_g), int(ref_b))

    # Observed RGBs for debug
    obs_mean = mean_rgb(img)
    obs_med = robust_observed_rgb_midtone_median(img)

    gains_lin = None
    gains_lin_tuple = None
    if use_cal:
        gains_lin = compute_reference_gains(obs_med, ref_true_rgb)
        gains_lin_tuple = (float(gains_lin[0]), float(gains_lin[1]), float(gains_lin[2]))

    with st.expander("Debug: upload + observed RGB + gains"):
        st.write("Filename:", uploaded.name)
        st.write("Size (bytes):", uploaded.size)
        st.write("SHA1:", sha1_hex(img_bytes))
        st.write("Mean RGB:", [float(f"{x:.3f}") for x in obs_mean])
        st.write("Observed midtone median RGB:", [float(f"{x:.3f}") for x in obs_med])
        st.write("Reference TRUE RGB:", ref_true_rgb)
        if gains_lin is not None:
            st.write("Linear gains (R,G,B):", [float(f"{g:.6f}") for g in gains_lin])

    st.subheader("Original Image")
    st.image(img, caption=f"{uploaded.name} • {img.shape[1]}×{img.shape[0]}", use_column_width=True)

    tab_single, tab_batch = st.tabs(["Single color", "Batch colors"])

    # ---------------- Single ----------------
    with tab_single:
        st.write("### Single color")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        with c1:
            r = st.number_input("R", 0, 255, 243, 1, key="sr")
        with c2:
            g = st.number_input("G", 0, 255, 224, 1, key="sg")
        with c3:
            b = st.number_input("B", 0, 255, 197, 1, key="sb")
        with c4:
            name = st.text_input("Color name (filename)", value="Sand Beige", key="sname")

        requested_rgb = (int(r), int(g), int(b))
        safe_name = sanitize_filename(name.strip() or _safe_name_from_rgb(requested_rgb))

        if st.button("Recolor", type="primary", key="single_go"):
            out, used_rgb = process_color(img, requested_rgb, use_cal, gains_lin)
            st.caption(f"Requested RGB: {requested_rgb}  →  Used RGB: {used_rgb}")
            st.image(out, caption=f"{name} • RGB{requested_rgb}", use_column_width=True)

            data5 = export_with_cap(Image.fromarray(out), 5 * 1024 * 1024)
            data10 = export_with_cap(Image.fromarray(out), 10 * 1024 * 1024)

            d1, d2 = st.columns(2)
            with d1:
                st.download_button("Download ≤5MB (JPG)", data=data5,
                                   file_name=f"{safe_name}.jpg", mime="image/jpeg", type="primary")
            with d2:
                st.download_button("Download ≤10MB (JPG)", data=data10,
                                   file_name=f"{safe_name}.jpg", mime="image/jpeg", type="primary")

    # ---------------- Batch ----------------
    with tab_batch:
        st.write("### Batch colors")
        st.caption("Paste: `Name,243,224,197` or `243,224,197` or `#F3E0C5, Name`. CSV also supported.")

        left, right = st.columns([2, 1])
        with left:
            txt = st.text_area(
                "Colors list",
                value="Sand Beige,243,224,197\nWarm White,246,242,234\nMisty Green,213,224,220\nCream,#F3E0C5",
                height=160,
                key="batch_text"
            )
        with right:
            csv_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")

        parsed: List[Tuple[str, Tuple[int, int, int]]] = []
        if txt.strip():
            parsed.extend(parse_rgb_lines(txt))
        if csv_file is not None:
            parsed.extend(parse_csv_file(csv_file))

        # Deduplicate
        seen = set()
        colors: List[Tuple[str, Tuple[int, int, int]]] = []
        for nm, rgb in parsed:
            nm = nm.strip() or _safe_name_from_rgb(rgb)
            key = (sanitize_filename(nm).lower(), rgb)
            if key not in seen:
                seen.add(key)
                colors.append((nm, rgb))

        st.write(f"**Detected {len(colors)} color(s).**")

        # ---- Previews only (safe) ----
        if st.button("Generate previews", type="primary", key="prev_go"):
            st.subheader("Previews")
            cols_ui = st.columns(3)

            # downscale base for previews only
            preview_base = img.copy()
            max_preview_dim = 900
            h, w = preview_base.shape[:2]
            if max(h, w) > max_preview_dim:
                scale = max_preview_dim / max(h, w)
                preview_base = cv2.resize(preview_base, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            for idx, (nm, rgb) in enumerate(colors[:MAX_COLORS_PREVIEW]):
                prev, used_rgb = process_color(preview_base, rgb, use_cal, gains_lin)
                with cols_ui[idx % 3]:
                    st.image(prev, caption=f"{nm} • RGB{rgb}", use_column_width=True)

            st.session_state["last_colors"] = colors

        # ---- Download ONE (on-demand) ----
        st.divider()
        st.write("### Download one (on-demand)")
        colors_for_select = st.session_state.get("last_colors", colors)
        if colors_for_select:
            options = [nm for nm, _ in colors_for_select]
            sel_name = st.selectbox("Select color", options, key="sel_color")
            sel_rgb = dict(colors_for_select)[sel_name]
            sel_safe = sanitize_filename(sel_name)

            b1, b2 = st.columns(2)
            with b1:
                if st.button("Prepare ≤5MB JPG", key="one5"):
                    out, used_rgb = process_color(img, sel_rgb, use_cal, gains_lin)
                    data = export_with_cap(Image.fromarray(out), 5 * 1024 * 1024)
                    st.download_button("Download ≤5MB", data=data,
                                       file_name=f"{sel_safe}.jpg", mime="image/jpeg", type="primary",
                                       key="dl_one5")
            with b2:
                if st.button("Prepare ≤10MB JPG", key="one10"):
                    out, used_rgb = process_color(img, sel_rgb, use_cal, gains_lin)
                    data = export_with_cap(Image.fromarray(out), 10 * 1024 * 1024)
                    st.download_button("Download ≤10MB", data=data,
                                       file_name=f"{sel_safe}.jpg", mime="image/jpeg", type="primary",
                                       key="dl_one10")

        # ---- ZIP exports (one at a time) ----
        st.divider()
        st.write("### Download ALL (ZIP)")
        st.caption("Build one ZIP at a time to avoid crashes. Uses calibrated targets if enabled.")

        img_hash = sha1_hex(img_bytes)
        salt = f"{img_hash}|{hash(str(colors))}|cal={use_cal}|ref={ref_true_rgb}|bias={OVERALL_ANCHOR_BIAS}"

        z1, z2 = st.columns(2)

        with z1:
            colors_5 = colors[:MAX_COLORS_ZIP_5MB]
            est_5 = len(colors_5) * 5
            if est_5 > MAX_ZIP_MB:
                st.warning(f"≤5MB ZIP estimate {est_5}MB exceeds {MAX_ZIP_MB}MB. Reduce colors.")
            if st.button("Build ZIP ≤5MB", type="primary", key="zip5_build"):
                if colors_5 and est_5 <= MAX_ZIP_MB:
                    with st.spinner("Building ZIP ≤5MB..."):
                        zip_bytes = build_zip_for_cap(img_bytes, colors_5, cap_mb=5, cache_salt=salt,
                                                      use_calibration=use_cal, gains_lin_tuple=gains_lin_tuple)
                    st.download_button("Download ZIP ≤5MB", data=zip_bytes,
                                       file_name="recolored_batch_5MB_max.zip",
                                       mime="application/zip", type="primary",
                                       key="zip5_dl")

        with z2:
            colors_10 = colors[:MAX_COLORS_ZIP_10MB]
            est_10 = len(colors_10) * 10
            if est_10 > MAX_ZIP_MB:
                st.warning(f"≤10MB ZIP estimate {est_10}MB exceeds {MAX_ZIP_MB}MB. Reduce colors.")
            if st.button("Build ZIP ≤10MB", type="primary", key="zip10_build"):
                if colors_10 and est_10 <= MAX_ZIP_MB:
                    with st.spinner("Building ZIP ≤10MB..."):
                        zip_bytes = build_zip_for_cap(img_bytes, colors_10, cap_mb=10, cache_salt=salt,
                                                      use_calibration=use_cal, gains_lin_tuple=gains_lin_tuple)
                    st.download_button("Download ZIP ≤10MB", data=zip_bytes,
                                       file_name="recolored_batch_10MB_max.zip",
                                       mime="application/zip", type="primary",
                                       key="zip10_dl")

except Exception as e:
    st.error("The app hit an error.")
    st.exception(e)
    st.text_area("Traceback", value=traceback.format_exc(), height=260)
