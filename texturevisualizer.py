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

st.set_page_config(page_title="Textured Paint Recolor", page_icon="🎨", layout="wide")
st.title("🎨 Textured Paint Recolor (Stable Batch + Size‑Capped Exports)")

# --------------------------- Core pipeline ---------------------------

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1 / 2.4)) - 0.055)

def white_balance_preserve_luma(img_rgb: np.ndarray) -> np.ndarray:
    """Gray-world WB with luminance preserved (prevents global brightening)."""
    img = img_rgb.astype(np.float32)
    means = img.mean(axis=(0, 1))
    target = float(np.mean(means))
    gains = target / (means + 1e-6)
    balanced = img * gains

    luma_before = (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]).mean()
    luma_after  = (0.2126 * balanced[:, :, 0] + 0.7152 * balanced[:, :, 1] + 0.0722 * balanced[:, :, 2]).mean()
    scale = luma_before / (luma_after + 1e-6)

    balanced = balanced * scale
    return np.clip(balanced, 0, 255).astype(np.uint8)

def rgb_to_lab_color(rgb: Tuple[int, int, int]) -> np.ndarray:
    r, g, b = [int(max(0, min(255, v))) for v in rgb]
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[0, 0]  # uint8 L,a,b in 0..255

def adjust_lightness_to_target(L: np.ndarray, target_L: float) -> np.ndarray:
    """Robust L remap: shift median to target while preserving contrast and avoiding clipping."""
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

def recolor_preserve_texture_with_L_match(img_rgb: np.ndarray, target_rgb: Tuple[int, int, int]) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    orig_a = lab[:, :, 1].astype(np.float32)
    orig_b = lab[:, :, 2].astype(np.float32)

    tL, tA, tB = rgb_to_lab_color(target_rgb)
    tA, tB = float(tA), float(tB)

    # Soft chroma mix (reduces pink/green in near-neutrals)
    chroma_mag = float(np.hypot(tA - 128.0, tB - 128.0))
    if chroma_mag < 3.0:
        alpha = 0.55
    elif chroma_mag < 10.0:
        alpha = 0.70
    else:
        alpha = 0.85

    a_final = alpha * tA + (1.0 - alpha) * orig_a
    b_final = alpha * tB + (1.0 - alpha) * orig_b
    a_u8 = np.clip(a_final, 0, 255).astype(np.uint8)
    b_u8 = np.clip(b_final, 0, 255).astype(np.uint8)

    L_matched = adjust_lightness_to_target(L, float(tL))
    out_lab = cv2.merge([L_matched, a_u8, b_u8])
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)

def match_midtone_luminance_to_target(img_u8: np.ndarray, target_rgb: Tuple[int, int, int]) -> np.ndarray:
    """Final overall-impression fix: match midtone luminance to target luminance."""
    img = img_u8.astype(np.float32) / 255.0
    lin = srgb_to_linear(img)
    Y = 0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2]

    p10, p90 = np.percentile(Y, 10), np.percentile(Y, 90)
    mid = (Y >= p10) & (Y <= p90)
    if np.count_nonzero(mid) < 1000:
        mid = np.ones_like(Y, dtype=bool)

    med_Y = float(np.median(Y[mid]))

    tgt = np.array(target_rgb, dtype=np.float32) / 255.0
    tgt_lin = srgb_to_linear(tgt)
    tgt_Y = float(0.2126 * tgt_lin[0] + 0.7152 * tgt_lin[1] + 0.0722 * tgt_lin[2])

    s = tgt_Y / (med_Y + 1e-6)
    s = float(np.clip(s, 0.70, 1.35))

    lin_adj = np.clip(lin * s, 0.0, 1.0)
    out = linear_to_srgb(lin_adj)
    return (np.clip(out * 255.0, 0, 255)).astype(np.uint8)

def process_color(img_rgb: np.ndarray, target_rgb: Tuple[int, int, int]) -> np.ndarray:
    out = recolor_preserve_texture_with_L_match(img_rgb, target_rgb)
    out = match_midtone_luminance_to_target(out, target_rgb)
    return out

# --------------------------- Export helpers ---------------------------

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
    """Return JPEG bytes <= cap_bytes (quality search + optional downscale)."""
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

        # Downscale and retry
        # Estimate needed shrink ratio from last attempt at min_quality
        data_min = _jpeg_bytes(img, min_quality)
        ratio = float(np.sqrt(cap_bytes / max(len(data_min), 1)) * 0.95)
        ratio = float(max(min(ratio, 0.92), 0.60))
        new_w = max(64, int(img.width * ratio))
        new_h = max(64, int(img.height * ratio))
        if (new_w, new_h) == img.size:
            new_w = max(64, img.width - 64)
            new_h = max(64, img.height - 64)
        img = img.resize((new_w, new_h), resample=Image.LANCZOS)

    # Fallback: return smallest we could make
    return _jpeg_bytes(img, min_quality)

# --------------------------- Batch parsing ---------------------------

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

        # contains hex
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

def fingerprint_image_bytes(uploaded_file) -> str:
    """Stable key for caching based on file bytes hash (small read)."""
    data = uploaded_file.getvalue()
    return hashlib.sha1(data).hexdigest()

# --------------------------- Cached ZIP builder ---------------------------

@st.cache_data(show_spinner=False)
def build_zip_for_cap(wb_img_u8: np.ndarray,
                      colors: List[Tuple[str, Tuple[int, int, int]]],
                      cap_mb: int,
                      cache_salt: str) -> bytes:
    """
    Build ZIP sequentially (memory-stable).
    cache_salt is included to invalidate cache when inputs change.
    """
    cap_bytes = cap_mb * 1024 * 1024
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, rgb in colors:
            out = process_color(wb_img_u8, rgb)
            data = export_with_cap(Image.fromarray(out), cap_bytes)
            filename = f"{sanitize_filename(name)}.jpg"
            zf.writestr(filename, data)
    return zip_buf.getvalue()

# --------------------------- Main UI ---------------------------

uploaded = st.file_uploader("Upload a textured paint image", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload a photo of the textured wall to begin.")
    st.stop()

try:
    # Load original full-res image
    img = np.array(Image.open(uploaded).convert("RGB"))
    st.subheader("Original Image")
    st.image(img, caption=f"Original ({img.shape[1]}×{img.shape[0]})", use_column_width=True)

    # White-balance once
    wb = white_balance_preserve_luma(img)
    st.subheader("White‑Balanced Preview")
    st.image(wb, caption="Used for recoloring (luminance preserved)", use_column_width=True)

    tab_single, tab_batch = st.tabs(["Single color", "Batch colors"])

    # -------------------- Single --------------------
    with tab_single:
        st.write("### Single color")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        with c1:
            r = st.number_input("R", 0, 255, 243, 1)
        with c2:
            g = st.number_input("G", 0, 255, 224, 1)
        with c3:
            b = st.number_input("B", 0, 255, 197, 1)
        with c4:
            name = st.text_input("Color name (filename)", value="Sand Beige")

        rgb = (int(r), int(g), int(b))
        safe_name = sanitize_filename(name.strip() or _safe_name_from_rgb(rgb))

        if st.button("Recolor", type="primary"):
            out = process_color(wb, rgb)
            st.image(out, caption=f"{name} • RGB{rgb}", use_column_width=True)

            pil_out = Image.fromarray(out)
            data5 = export_with_cap(pil_out, 5 * 1024 * 1024)
            data10 = export_with_cap(pil_out, 10 * 1024 * 1024)

            d1, d2 = st.columns(2)
            with d1:
                st.download_button("Download ≤5MB (JPG)", data=data5,
                                   file_name=f"{safe_name}.jpg", mime="image/jpeg")
            with d2:
                st.download_button("Download ≤10MB (JPG)", data=data10,
                                   file_name=f"{safe_name}.jpg", mime="image/jpeg")

    # -------------------- Batch --------------------
    with tab_batch:
        st.write("### Batch colors (stable mode)")
        st.caption("Paste: `Name,243,224,197` or `243,224,197` or `#F3E0C5, Name`. CSV also supported (name,r,g,b).")

        left, right = st.columns([2, 1])
        with left:
            txt = st.text_area(
                "Colors list",
                value="Sand Beige,243,224,197\nWarm White,246,242,234\nMisty Green,213,224,220\nCream,#F3E0C5",
                height=160
            )
        with right:
            csv_file = st.file_uploader("Upload CSV", type=["csv"])

        parsed: List[Tuple[str, Tuple[int, int, int]]] = []
        if txt.strip():
            parsed.extend(parse_rgb_lines(txt))
        if csv_file is not None:
            parsed.extend(parse_csv_file(csv_file))

        # Deduplicate on (sanitized name + rgb)
        seen = set()
        colors: List[Tuple[str, Tuple[int, int, int]]] = []
        for nm, rgb in parsed:
            nm = nm.strip() or _safe_name_from_rgb(rgb)
            key = (sanitize_filename(nm).lower(), rgb)
            if key not in seen:
                seen.add(key)
                colors.append((nm, rgb))

        st.write(f"**Detected {len(colors)} color(s).**")

        # Hard safety cap for Cloud stability
        max_colors = 30
        if len(colors) > max_colors:
            st.warning(f"Limiting to first {max_colors} colors to prevent crashes on Streamlit Cloud.")
            colors = colors[:max_colors]

        # --- Preview generation only (fast + low memory)
        if st.button("Generate previews", type="primary"):
            st.subheader("Previews")
            cols_ui = st.columns(3)

            # Use smaller preview processing to keep UI snappy (doesn't affect exports)
            preview_base = wb.copy()
            max_preview_dim = 900
            h, w = preview_base.shape[:2]
            if max(h, w) > max_preview_dim:
                scale = max_preview_dim / max(h, w)
                preview_base = cv2.resize(preview_base, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            for idx, (nm, rgb) in enumerate(colors):
                prev = process_color(preview_base, rgb)
                with cols_ui[idx % 3]:
                    st.image(prev, caption=f"{nm} • RGB{rgb}", use_column_width=True)

            st.session_state["last_colors"] = colors  # store names+rgbs only

        # --- Export ONE selected color (on demand)
        st.divider()
        st.write("### Download one (on-demand, avoids crashing)")
        colors_for_select = st.session_state.get("last_colors", colors)
        if colors_for_select:
            options = [nm for nm, _ in colors_for_select]
            sel = st.selectbox("Select color to export", options)
            sel_rgb = dict(colors_for_select)[sel]

            cA, cB = st.columns(2)
            with cA:
                if st.button("Prepare ≤5MB JPG"):
                    out = process_color(wb, sel_rgb)
                    data = export_with_cap(Image.fromarray(out), 5 * 1024 * 1024)
                    st.download_button("Download ≤5MB", data=data,
                                       file_name=f"{sanitize_filename(sel)}.jpg", mime="image/jpeg",
                                       type="primary")
            with cB:
                if st.button("Prepare ≤10MB JPG"):
                    out = process_color(wb, sel_rgb)
                    data = export_with_cap(Image.fromarray(out), 10 * 1024 * 1024)
                    st.download_button("Download ≤10MB", data=data,
                                       file_name=f"{sanitize_filename(sel)}.jpg", mime="image/jpeg",
                                       type="primary")

        # --- ZIP export (build one at a time, cached)
        st.divider()
        st.write("### Download ALL (ZIP)")
        st.caption("Builds sequentially and caches results. Choose only one cap at a time to keep memory stable.")

        img_hash = fingerprint_image_bytes(uploaded)
        salt = f"{img_hash}|{hash(str(colors))}"

        z1, z2 = st.columns(2)
        with z1:
            if st.button("Build ZIP ≤5MB", type="primary"):
                with st.spinner("Building ZIP ≤5MB..."):
                    zip_bytes = build_zip_for_cap(wb, colors, cap_mb=5, cache_salt=salt)
                st.download_button("Download ZIP ≤5MB", data=zip_bytes,
                                   file_name="recolored_batch_5MB_max.zip",
                                   mime="application/zip",
                                   type="primary")
        with z2:
            if st.button("Build ZIP ≤10MB", type="primary"):
                with st.spinner("Building ZIP ≤10MB..."):
                    zip_bytes = build_zip_for_cap(wb, colors, cap_mb=10, cache_salt=salt)
                st.download_button("Download ZIP ≤10MB", data=zip_bytes,
                                   file_name="recolored_batch_10MB_max.zip",
                                   mime="application/zip",
                                   type="application/zip",
                                   )

    with st.expander("Debug info"):
        st.write({"shape": img.shape, "dtype": str(img.dtype), "file": uploaded.name})

except Exception as e:
    st.error("The app hit an error.")
    st.exception(e)
    st.text_area("Traceback", value=traceback.format_exc(), height=260)
