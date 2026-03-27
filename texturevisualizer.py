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
st.title("🎨 Textured Paint Recolor (Faster UI + RGB Swatch + Texture-Aware Matching)")


# =========================
# Constants (no sliders)
# =========================
# Streamlit Cloud stability caps
MAX_COLORS_PREVIEW = 40
MAX_COLORS_ZIP_5MB = 30
MAX_COLORS_ZIP_10MB = 18
MAX_ZIP_MB = 180  # rough ZIP size estimate guard

# Overall impression anchoring settings (perceptual sRGB luma)
ANCHOR_LOW_PCT = 25
ANCHOR_HIGH_PCT = 85
ANCHOR_REF_PCT_BASE = 70  # will be adjusted slightly with texture

# JPEG export caps
CAP_5MB = 5 * 1024 * 1024
CAP_10MB = 10 * 1024 * 1024

# Display sizes (UI only)
UPLOAD_PREVIEW_MAX_W = 700       # smaller = faster
BATCH_PREVIEW_MAX_W = 650        # per-tile display width
SWATCH_W, SWATCH_H = 140, 90     # RGB sample swatch size


# =========================
# Robust upload load
# =========================
def load_uploaded_image_bytes(uploaded_file) -> Tuple[bytes, np.ndarray]:
    """Always load from bytes to avoid pointer/caching issues."""
    img_bytes = uploaded_file.getvalue()
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    return img_bytes, img

def sha1_hex(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


# =========================
# UI helpers (fast preview + swatch)
# =========================
def make_small_preview(img_u8: np.ndarray, max_w: int) -> np.ndarray:
    """Downscale only for display (keeps processing full-res)."""
    h, w = img_u8.shape[:2]
    if w <= max_w:
        return img_u8
    scale = max_w / w
    new_size = (max_w, max(1, int(h * scale)))
    return cv2.resize(img_u8, new_size, interpolation=cv2.INTER_AREA)

def rgb_swatch(rgb: Tuple[int, int, int], w: int = SWATCH_W, h: int = SWATCH_H) -> np.ndarray:
    """Create a solid RGB swatch image."""
    sw = np.zeros((h, w, 3), dtype=np.uint8)
    sw[:, :] = np.array(rgb, dtype=np.uint8)
    return sw


# =========================
# White balance (internal)
# =========================
def white_balance_preserve_luma(img_rgb: np.ndarray) -> np.ndarray:
    """
    Gray-world WB but preserve overall luminance so WB doesn't brighten/darken globally.
    Often close to identity (gains ~1) for neutral photos.
    """
    img = img_rgb.astype(np.float32)
    means = img.mean(axis=(0, 1))  # [R,G,B]
    target = float(np.mean(means))
    gains = target / (means + 1e-6)

    balanced = img * gains

    # Preserve luma (Rec.709 proxy in gamma-encoded RGB)
    luma_before = (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]).mean()
    luma_after  = (0.2126 * balanced[:, :, 0] + 0.7152 * balanced[:, :, 1] + 0.0722 * balanced[:, :, 2]).mean()
    scale = luma_before / (luma_after + 1e-6)

    balanced *= scale
    return np.clip(balanced, 0, 255).astype(np.uint8)


# =========================
# Texture detection -> adjust brightness behavior automatically
# =========================
def texture_strength_score(img_u8: np.ndarray) -> float:
    """
    Estimate texture strength from high-frequency energy.
    Uses variance of Laplacian on grayscale (simple, fast).
    Returns score in [0,1] roughly.
    """
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    # Normalize a bit to reduce exposure dependence
    gray = cv2.equalizeHist(gray)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    var = float(lap.var())

    # Map var -> [0,1] using log scale + clamp
    # These bounds are empirical; they work well for most wall textures.
    vmin, vmax = 50.0, 2500.0
    x = (np.log(var + 1.0) - np.log(vmin + 1.0)) / (np.log(vmax + 1.0) - np.log(vmin + 1.0))
    return float(np.clip(x, 0.0, 1.0))

def texture_aware_bias_and_anchor(texture_score: float) -> Tuple[float, int]:
    """
    Highly textured surfaces tend to look lighter due to micro-highlights.
    So we darken a bit more when texture is high.

    Returns:
      - bias: multiplier applied in overall brightness anchoring
      - anchor_ref_pct: percentile used for highlight-aware reference (higher for more texture)
    """
    # Bias range: low texture ~0.96, high texture ~0.92 (stronger darkening)
    bias = 0.96 - 0.04 * texture_score

    # Anchor ref percentile: low texture 65, high texture 75 (more highlight-aware)
    anchor_ref = int(round(65 + 10 * texture_score))
    anchor_ref = int(np.clip(anchor_ref, 60, 80))
    return float(bias), anchor_ref


# =========================
# Stable Lab recolor pipeline
# =========================
def rgb_to_lab_color(rgb: Tuple[int, int, int]) -> np.ndarray:
    """Convert (R,G,B) to LAB in OpenCV uint8 scale (0..255)."""
    r, g, b = [int(max(0, min(255, v))) for v in rgb]
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[0, 0]  # (L,a,b)

def adjust_lightness_to_target(L: np.ndarray, target_L: float) -> np.ndarray:
    """
    Robustly remap L so median moves to target_L while preserving contrast and avoiding clipping.
    """
    Lf = L.astype(np.float32)
    p10 = np.percentile(Lf, 10.0)
    p50 = np.percentile(Lf, 50.0)
    p90 = np.percentile(Lf, 90.0)

    d_hi = max(1.0, float(p90 - p50))
    d_lo = max(1.0, float(p50 - p10))

    k_hi_max = (255.0 - target_L) / d_hi if d_hi > 0 else 1e9
    k_lo_max = (target_L - 0.0) / d_lo if d_lo > 0 else 1e9

    k_safe = max(0.0, min(k_hi_max, k_lo_max)) * 0.98
    k = min(1.0, k_safe)

    L_adj = (Lf - p50) * k + target_L
    return np.clip(L_adj, 0, 255).astype(np.uint8)

def recolor_lab_texture_preserved(img_rgb: np.ndarray, target_rgb: Tuple[int, int, int]) -> np.ndarray:
    """
    Stable method:
      - Convert to LAB
      - Robustly remap L toward target_L (preserves texture contrast)
      - Apply target chroma with soft mix (avoids artifacts for near-neutrals)
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    orig_a = lab[:, :, 1].astype(np.float32)
    orig_b = lab[:, :, 2].astype(np.float32)

    tL, tA, tB = rgb_to_lab_color(target_rgb)
    tA, tB = float(tA), float(tB)

    # Soft chroma mix based on neutrality in OpenCV LAB (center ~128,128)
    chroma_mag = float(np.hypot(tA - 128.0, tB - 128.0))
    if chroma_mag < 3.0:
        alpha = 0.55
    elif chroma_mag < 10.0:
        alpha = 0.70
    else:
        alpha = 0.85

    a_out = alpha * tA + (1.0 - alpha) * orig_a
    b_out = alpha * tB + (1.0 - alpha) * orig_b
    a_u8 = np.clip(a_out, 0, 255).astype(np.uint8)
    b_u8 = np.clip(b_out, 0, 255).astype(np.uint8)

    L_out = adjust_lightness_to_target(L, float(tL))

    out_lab = cv2.merge([L_out, a_u8, b_u8])
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)

def anchor_overall_impression_L_only(output_u8: np.ndarray, target_rgb: Tuple[int, int, int],
                                     bias: float, anchor_ref_pct: int) -> np.ndarray:
    """
    Overall impression correction (texture-aware):
    - compute sRGB luma
    - reference = anchor_ref_pct percentile within 25–85% band
    - apply scale s * bias to LAB L only (preserves chroma)
    """
    img = output_u8.astype(np.float32) / 255.0
    Y = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]

    p_lo, p_hi = np.percentile(Y, ANCHOR_LOW_PCT), np.percentile(Y, ANCHOR_HIGH_PCT)
    mid = (Y >= p_lo) & (Y <= p_hi)
    if np.count_nonzero(mid) < 1000:
        mid = np.ones_like(Y, dtype=bool)

    y_ref = float(np.percentile(Y[mid], anchor_ref_pct))

    tr, tg, tb = [v / 255.0 for v in target_rgb]
    y_tgt = float(0.2126 * tr + 0.7152 * tg + 0.0722 * tb)

    s = (y_tgt / (y_ref + 1e-6)) * float(bias)
    s = float(np.clip(s, 0.70, 1.20))

    lab = cv2.cvtColor(output_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] * s, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

def process_color(base_img_u8: np.ndarray, target_rgb: Tuple[int, int, int],
                  bias: float, anchor_ref_pct: int) -> np.ndarray:
    out = recolor_lab_texture_preserved(base_img_u8, target_rgb)
    out = anchor_overall_impression_L_only(out, target_rgb, bias=bias, anchor_ref_pct=anchor_ref_pct)
    return out


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
    """Export JPEG bytes <= cap_bytes using quality search + downscale if needed."""
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
                      bias: float,
                      anchor_ref_pct: int) -> bytes:
    base_img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    base_img = white_balance_preserve_luma(base_img)

    cap_bytes = cap_mb * 1024 * 1024
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, rgb in colors:
            out = process_color(base_img, rgb, bias=bias, anchor_ref_pct=anchor_ref_pct)
            data = export_with_cap(Image.fromarray(out), cap_bytes)
            zf.writestr(f"{sanitize_filename(name)}.jpg", data)
    return zip_buf.getvalue()


# =========================
# UI
# =========================
uploaded = st.file_uploader("Upload a textured paint image", type=["jpg", "jpeg", "png"], key="uploader_v4")
if uploaded is None:
    st.info("Upload a photo to begin.")
    st.stop()

try:
    img_bytes, img = load_uploaded_image_bytes(uploaded)

    # Internal working image (WB applied but not shown)
    base_img = white_balance_preserve_luma(img)

    # Texture-aware parameters (computed once per upload)
    tscore = texture_strength_score(base_img)
    bias, anchor_ref_pct = texture_aware_bias_and_anchor(tscore)

    # 1) Smaller uploaded image preview (UI only)
    st.subheader("Uploaded image (preview)")
    img_small = make_small_preview(img, UPLOAD_PREVIEW_MAX_W)
    st.image(img_small, caption=f"{uploaded.name} (preview)", width="content")  # width param replaces deprecated use_column_width [1](https://docs.streamlit.io/develop/api-reference/media/st.image)[2](https://discuss.streamlit.io/t/version-1-40-0/85145)

    with st.expander("Texture diagnostics (auto)"):
        st.write(f"Texture score (0–1): {tscore:.3f}")
        st.write(f"Auto brightness bias: {bias:.3f}")
        st.write(f"Auto anchor ref percentile: {anchor_ref_pct}")

    tab_single, tab_batch = st.tabs(["Single color", "Batch colors"])

    # ---------------- Single ----------------
    with tab_single:
        st.write("### Request one color")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        with c1:
            r = st.number_input("R", 0, 255, 243, 1, key="sr")
        with c2:
            g = st.number_input("G", 0, 255, 224, 1, key="sg")
        with c3:
            b = st.number_input("B", 0, 255, 197, 1, key="sb")
        with c4:
            name = st.text_input("Color name (filename)", value="Sand Beige", key="sname")

        target_rgb = (int(r), int(g), int(b))
        safe_name = sanitize_filename(name.strip() or _safe_name_from_rgb(target_rgb))

        if st.button("Recolor", type="primary", key="single_go"):
            out = process_color(base_img, target_rgb, bias=bias, anchor_ref_pct=anchor_ref_pct)

            # 2) Output + RGB swatch side-by-side
            col_img, col_swatch = st.columns([4, 1])
            with col_img:
                out_small = make_small_preview(out, UPLOAD_PREVIEW_MAX_W)
                st.image(out_small, caption=f"{name} • RGB{target_rgb}", width="stretch")  # width API [1](https://docs.streamlit.io/develop/api-reference/media/st.image)
            with col_swatch:
                st.image(rgb_swatch(target_rgb), caption="Requested RGB", width="content")  # [1](https://docs.streamlit.io/develop/api-reference/media/st.image)

            data5 = export_with_cap(Image.fromarray(out), CAP_5MB)
            data10 = export_with_cap(Image.fromarray(out), CAP_10MB)

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

        # Previews (fast + safe)
        if st.button("Generate previews", type="primary", key="prev_go"):
            st.subheader("Previews")
            cols_ui = st.columns(3)

            # use a smaller base for preview only
            preview_base = make_small_preview(base_img, BATCH_PREVIEW_MAX_W)

            for idx, (nm, rgb) in enumerate(colors[:MAX_COLORS_PREVIEW]):
                prev = process_color(preview_base, rgb, bias=bias, anchor_ref_pct=anchor_ref_pct)

                with cols_ui[idx % 3]:
                    st.image(prev, caption=f"{nm} • RGB{rgb}", width="stretch")  # width API [1](https://docs.streamlit.io/develop/api-reference/media/st.image)
                    st.image(rgb_swatch(rgb), caption="Requested RGB", width="content")  # [1](https://docs.streamlit.io/develop/api-reference/media/st.image)

            st.session_state["last_colors"] = colors

        # Download one (on demand)
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
                    out = process_color(base_img, sel_rgb, bias=bias, anchor_ref_pct=anchor_ref_pct)
                    data = export_with_cap(Image.fromarray(out), CAP_5MB)
                    st.download_button("Download ≤5MB", data=data,
                                       file_name=f"{sel_safe}.jpg", mime="image/jpeg", type="primary",
                                       key="dl_one5")
            with b2:
                if st.button("Prepare ≤10MB JPG", key="one10"):
                    out = process_color(base_img, sel_rgb, bias=bias, anchor_ref_pct=anchor_ref_pct)
                    data = export_with_cap(Image.fromarray(out), CAP_10MB)
                    st.download_button("Download ≤10MB", data=data,
                                       file_name=f"{sel_safe}.jpg", mime="image/jpeg", type="primary",
                                       key="dl_one10")

        # ZIP exports (one at a time)
        st.divider()
        st.write("### Download ALL (ZIP)")
        st.caption("Build one ZIP at a time to avoid crashes. Uses texture-aware auto settings.")

        salt = f"{sha1_hex(img_bytes)}|{hash(str(colors))}|bias={bias:.3f}|ref={anchor_ref_pct}"

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
                                                      bias=bias, anchor_ref_pct=anchor_ref_pct)
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
                                                      bias=bias, anchor_ref_pct=anchor_ref_pct)
                    st.download_button("Download ZIP ≤10MB", data=zip_bytes,
                                       file_name="recolored_batch_10MB_max.zip",
                                       mime="application/zip", type="primary",
                                       key="zip10_dl")

except Exception as e:
    st.error("The app hit an error.")
    st.exception(e)
    st.text_area("Traceback", value=traceback.format_exc(), height=260)
