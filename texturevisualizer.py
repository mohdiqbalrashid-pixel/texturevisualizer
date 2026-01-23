import io
import csv
import zipfile
import traceback
from typing import List, Tuple, Optional

import numpy as np
import cv2
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Textured Paint Recolor", page_icon="🎨", layout="wide")
st.title("🎨 Textured Paint Recolor (WB + Lightness Match + Size‑Capped Exports)")

# ------------------------ Core Utilities ------------------------

def gray_world_white_balance(img_rgb: np.ndarray) -> np.ndarray:
    img = img_rgb.astype(np.float32)
    means = img.mean(axis=(0, 1))  # [R,G,B]
    target = float(np.mean(means))
    gains = target / (means + 1e-6)
    balanced = img * gains
    return np.clip(balanced, 0, 255).astype(np.uint8)

def rgb_to_lab_color(rgb_tuple):
    r, g, b = [int(max(0, min(255, v))) for v in rgb_tuple]
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[0, 0]

def adjust_lightness_to_target(L: np.ndarray, target_L: float) -> np.ndarray:
    L = L.astype(np.float32)
    p10 = np.percentile(L, 10.0)
    p50 = np.percentile(L, 50.0)
    p90 = np.percentile(L, 90.0)
    d_hi = max(1.0, float(p90 - p50))
    d_lo = max(1.0, float(p50 - p10))
    k_hi_max = (255.0 - target_L) / d_hi if d_hi > 0 else 1e9
    k_lo_max = (target_L - 0.0) / d_lo if d_lo > 0 else 1e9
    k_safe = max(0.0, min(k_hi_max, k_lo_max)) * 0.98
    k = min(1.0, k_safe)
    L_adj = (L - p50) * k + target_L
    return np.clip(L_adj, 0, 255).astype(np.uint8)

def recolor_preserve_texture_with_L_match(img_rgb: np.ndarray, target_rgb) -> np.ndarray:
    lab_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L = lab_img[:, :, 0]
    orig_a = lab_img[:, :, 1].astype(np.float32)
    orig_b = lab_img[:, :, 2].astype(np.float32)

    tL, tA, tB = rgb_to_lab_color(target_rgb)
    tA = float(tA); tB = float(tB)

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
    out_rgb = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)
    return out_rgb

# ------------------------ Filenames & Export Helpers ------------------------

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
                    max_downscales: int = 5):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    def try_compress(img: Image.Image):
        lo, hi = min_quality, max_quality
        best_bytes, best_q = None, None
        for _ in range(8):
            q = (lo + hi) // 2
            data = _jpeg_bytes(img, q)
            if len(data) <= cap_bytes:
                best_bytes, best_q = data, q
                lo = q + 1
            else:
                hi = q - 1
        if best_bytes is not None:
            return best_bytes, best_q
        return _jpeg_bytes(img, min_quality), min_quality

    img = pil_img
    attempt = 0
    while True:
        data, q = try_compress(img)
        if len(data) <= cap_bytes:
            return data, q, img.size
        if attempt >= max_downscales:
            return data, q, img.size
        ratio = np.sqrt(cap_bytes / max(len(data), 1)) * 0.95
        ratio = float(max(min(ratio, 0.95), 0.60))
        new_w = max(64, int(img.width * ratio))
        new_h = max(64, int(img.height * ratio))
        if (new_w, new_h) == img.size:
            new_w = max(64, img.width - 64)
            new_h = max(64, img.height - 64)
        img = img.resize((new_w, new_h), resample=Image.LANCZOS)
        attempt += 1

# ------------------------ Parsing Utilities (Batch) ------------------------

def _safe_name_from_rgb(rgb: Tuple[int, int, int]) -> str:
    return f"{rgb[0]}-{rgb[1]}-{rgb[2]}"

def _parse_hex_token(tok: str) -> Optional[Tuple[int, int, int]]:
    s = tok.strip().lstrip("#")
    if len(s) in (3, 6):
        if len(s) == 3:
            s = "".join(ch*2 for ch in s)
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
        if len(parts) == 4 and all(parts):
            name = parts[0]
            try:
                r, g, b = int(parts[1]), int(parts[2]), int(parts[3])
                if all(0 <= v <= 255 for v in (r, g, b)):
                    colors.append((name, (r, g, b)))
                    continue
            except ValueError:
                pass
        if len(parts) == 3:
            try:
                r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                if all(0 <= v <= 255 for v in (r, g, b)):
                    colors.append((_safe_name_from_rgb((r, g, b)), (r, g, b)))
                    continue
            except ValueError:
                pass
        hex_rgb = None
        words = line.replace(",", " ").split()
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
        if len(row) >= 3 and any(h.lower() in ("r", "g", "b") for h in row[:3]):
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

# ------------------------ UI ------------------------

uploaded_file = st.file_uploader("Upload a textured paint image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        img = np.array(image)

        st.subheader("Original Image")
        st.image(img, caption=f"Original ({img.shape[1]}×{img.shape[0]})", use_column_width=True)

        wb = gray_world_white_balance(img)
        st.subheader("White-Balanced Preview")
        st.image(wb, caption="Gray-world corrected (used for recoloring)", use_column_width=True)

        tab_single, tab_batch = st.tabs(["Single color", "Batch colors"])

        # -------- Single color --------
        with tab_single:
            st.write("### Enter Target Color (RGB)")
            c1, c2, c3, c4 = st.columns([1,1,1,2])
            with c1:
                r = st.number_input("Red (0–255)", min_value=0, max_value=255, value=243, step=1)
            with c2:
                g = st.number_input("Green (0–255)", min_value=0, max_value=255, value=224, step=1)
            with c3:
                b = st.number_input("Blue (0–255)", min_value=0, max_value=255, value=197, step=1)
            with c4:
                name_input = st.text_input("Color name (optional)", value="Sand Beige")

            target_rgb = (int(r), int(g), int(b))
            color_name = name_input.strip() or f"{r}-{g}-{b}"
            safe_name = sanitize_filename(color_name)

            if st.button("Recolor", type="primary"):
                recolored = recolor_preserve_texture_with_L_match(wb, target_rgb)
                st.subheader("Recolored (High‑Res)")
                st.image(recolored, caption=f"{color_name}  •  RGB{target_rgb}", use_column_width=True)

                cap5 = 5 * 1024 * 1024
                cap10 = 10 * 1024 * 1024
                pil_out = Image.fromarray(recolored)

                data5, q5, size5 = export_with_cap(pil_out, cap5)
                data10, q10, size10 = export_with_cap(pil_out, cap10)

                cA, cB = st.columns(2)
                with cA:
                    st.download_button(
                        "Download ≤5MB (JPG)",
                        data=data5,
                        file_name=f"{safe_name}.jpg",
                        mime="image/jpeg",
                        key="single_5mb"
                    )
                    st.caption(f"≤5MB • q≈{q5} • {size5[0]}×{size5[1]} • {len(data5)/1024/1024:.2f} MB")
                with cB:
                    st.download_button(
                        "Download ≤10MB (JPG)",
                        data=data10,
                        file_name=f"{safe_name}.jpg",
                        mime="image/jpeg",
                        key="single_10mb"
                    )
                    st.caption(f"≤10MB • q≈{q10} • {size10[0]}×{size10[1]} • {len(data10)/1024/1024:.2f} MB")

        # -------- Batch colors --------
        with tab_batch:
            st.write("### Paste colors or upload CSV")
            st.caption(
                "Per line: `243,224,197` • `Sand Beige,243,224,197` • `#F3E0C5` • `F3E0C5,Sand Beige` • `Sand Beige #F3E0C5`.\n"
                "CSV: `name,r,g,b` (header optional)."
            )
            cols = st.columns([2, 1])
            with cols[0]:
                txt = st.text_area(
                    "Paste RGB/Hex lines here",
                    value="Sand Beige,243,224,197\n#F6F2EA, Warm White\nMisty Green #D5E0DC\n243,224,197",
                    height=160
                )
            with cols[1]:
                csv_file = st.file_uploader("...or upload CSV", type=["csv"], key="csv_upl")

            parsed = []
            if txt.strip():
                parsed.extend(parse_rgb_lines(txt))
            if csv_file is not None:
                parsed.extend(parse_csv_file(csv_file))

            # Deduplicate (name+rgb)
            seen = set()
            batch = []
            for name, rgb in parsed:
                key = (name.strip().lower(), rgb)
                if key not in seen:
                    seen.add(key)
                    batch.append((name.strip() or _safe_name_from_rgb(rgb), rgb))

            st.write(f"**Detected {len(batch)} color(s).**")
            max_colors = 40
            if len(batch) > max_colors:
                st.warning(f"Limiting to first {max_colors} colors to keep the app responsive.")
                batch = batch[:max_colors]

            if st.button("Generate all", type="primary", key="gen_all"):
                results = []
                prog = st.progress(0)
                for i, (name, rgb) in enumerate(batch, start=1):
                    recol = recolor_preserve_texture_with_L_match(wb, rgb)
                    preview = Image.fromarray(recol).copy()
                    preview.thumbnail((1200, 1200))
                    pil_out = Image.fromarray(recol)

                    data5, q5, size5 = export_with_cap(pil_out, 5 * 1024 * 1024)
                    data10, q10, size10 = export_with_cap(pil_out, 10 * 1024 * 1024)

                    results.append({
                        "name": name,
                        "safe_name": sanitize_filename(name),
                        "rgb": rgb,
                        "preview_rgb": np.array(preview),
                        "data5": data5, "q5": q5, "size5": size5,
                        "data10": data10, "q10": q10, "size10": size10,
                    })
                    prog.progress(i / len(batch))

                # Gallery (single preview + two download buttons per color)
                st.subheader("Previews")
                cols = st.columns(3)
                for idx, item in enumerate(results):
                    with cols[idx % 3]:
                        st.image(item["preview_rgb"],
                                 caption=f"{item['name']}  •  RGB{item['rgb']}",
                                 use_column_width=True)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.download_button(
                                label="≤5MB JPG",
                                data=item["data5"],
                                file_name=f"{item['safe_name']}.jpg",
                                mime="image/jpeg",
                                key=f"dl5_{idx}"
                            )
                        with c2:
                            st.download_button(
                                label="≤10MB JPG",
                                data=item["data10"],
                                file_name=f"{item['safe_name']}.jpg",
                                mime="image/jpeg",
                                key=f"dl10_{idx}"
                            )

                # ---- NEW: Download ALL ZIPs (5MB and 10MB) ----
                if results:
                    # Build ZIP for ≤5MB
                    zip_buf_5 = io.BytesIO()
                    with zipfile.ZipFile(zip_buf_5, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for item in results:
                            zf.writestr(f"{item['safe_name']}.jpg", item["data5"])
                    st.download_button(
                        "Download ALL ≤5MB (ZIP)",
                        data=zip_buf_5.getvalue(),
                        file_name="recolored_batch_5MB_max.zip",
                        mime="application/zip",
                        type="primary",
                        key="dl_all_zip_5"
                    )

                    # Build ZIP for ≤10MB
                    zip_buf_10 = io.BytesIO()
                    with zipfile.ZipFile(zip_buf_10, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for item in results:
                            zf.writestr(f"{item['safe_name']}.jpg", item["data10"])
                    st.download_button(
                        "Download ALL ≤10MB (ZIP)",
                        data=zip_buf_10.getvalue(),
                        file_name="recolored_batch_10MB_max.zip",
                        mime="application/zip",
                        type="primary",
                        key="dl_all_zip_10"
                    )

        with st.expander("Debug info"):
            st.write({"image_shape": img.shape, "dtype": str(img.dtype)})

    except Exception as e:
        st.error("There was an error while processing the image.")
        st.exception(e)
        st.text_area("Traceback", value=traceback.format_exc(), height=240)

else:
    st.info("Upload a photo of the textured wall to begin.")
