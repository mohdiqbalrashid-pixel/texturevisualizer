import io
import traceback
import numpy as np
import cv2
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Textured Paint Recolor", page_icon="🎨", layout="centered")
st.title("🎨 Textured Paint Recolor (White Balance + Lightness Match)")

# ---------- Utilities ----------
def gray_world_white_balance(img_rgb: np.ndarray) -> np.ndarray:
    """
    Simple gray-world white balance to neutralize warm/cool cast.
    Keeps resolution, returns uint8 RGB.
    """
    img = img_rgb.astype(np.float32)
    means = img.mean(axis=(0, 1))  # [R,G,B]
    target = float(np.mean(means))
    gains = target / (means + 1e-6)
    balanced = img * gains
    return np.clip(balanced, 0, 255).astype(np.uint8)

def rgb_to_lab_color(rgb_tuple):
    """
    Convert (R,G,B) -> single LAB color in OpenCV's LAB scale (uint8 0..255 per channel).
    """
    r, g, b = [int(max(0, min(255, v))) for v in rgb_tuple]
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[0, 0]  # (L,a,b)

def adjust_lightness_to_target(L: np.ndarray, target_L: float) -> np.ndarray:
    """
    Robustly remap the lightness channel so the image median moves to target_L,
    while preserving contrast and avoiding clipping.

    Strategy (OpenCV L in 0..255):
      1) Compute percentiles (p10, median, p90) to avoid outliers.
      2) Start with contrast scale k=1 (preserve texture contrast).
      3) Compute the maximum safe k that avoids clipping after shifting to target_L.
      4) Use k = min(1, k_safe*0.98), then L' = (L - median)*k + target_L.
    """
    L = L.astype(np.float32)
    p10 = np.percentile(L, 10.0)
    p50 = np.percentile(L, 50.0)
    p90 = np.percentile(L, 90.0)

    d_hi = max(1.0, float(p90 - p50))  # avoid div by zero/very small spreads
    d_lo = max(1.0, float(p50 - p10))

    # Max scale to avoid clipping upper bound when shifting to target
    k_hi_max = (255.0 - target_L) / d_hi if d_hi > 0 else 1.0e9
    # Max scale to avoid clipping lower bound when shifting to target
    k_lo_max = (target_L - 0.0) / d_lo if d_lo > 0 else 1.0e9

    k_safe = max(0.0, min(k_hi_max, k_lo_max)) * 0.98
    k = min(1.0, k_safe)  # prefer preserving original contrast if safe

    L_adj = (L - p50) * k + target_L
    return np.clip(L_adj, 0, 255).astype(np.uint8)

def recolor_preserve_texture_with_L_match(img_rgb: np.ndarray, target_rgb) -> np.ndarray:
    """
    Pipeline:
      - Convert to LAB.
      - Compute target LAB; get target_L.
      - Soft chroma mix (avoid weird tints for near-neutrals).
      - Robustly remap L to target_L using percentile-based affine mapping.
      - Merge and convert back to RGB.
    """
    # LAB image
    lab_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L = lab_img[:, :, 0]
    orig_a = lab_img[:, :, 1].astype(np.float32)
    orig_b = lab_img[:, :, 2].astype(np.float32)

    # Target color in LAB (OpenCV scale)
    tL, tA, tB = rgb_to_lab_color(target_rgb)
    tA = float(tA); tB = float(tB)

    # Soft chroma mix:
    #   - If target is near-neutral (|a-128|,|b-128| small), keep more of original chroma to avoid pink/green cast.
    chroma_mag = np.hypot(tA - 128.0, tB - 128.0)
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

    # Robust L remap to target lightness (this fixes light targets looking too dark)
    L_matched = adjust_lightness_to_target(L, float(tL))

    # Merge and convert back
    out_lab = cv2.merge([L_matched, a_u8, b_u8])
    out_rgb = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)
    return out_rgb

# ---------- UI ----------
uploaded_file = st.file_uploader("Upload a textured paint image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load original (full-res)
        image = Image.open(uploaded_file).convert("RGB")
        img = np.array(image)

        st.subheader("Original Image")
        st.image(img, caption=f"Original ({img.shape[1]}×{img.shape[0]})", use_column_width=True)

        # White balance for more faithful recolor under various lighting
        wb = gray_world_white_balance(img)
        st.subheader("White-Balanced Preview")
        st.image(wb, caption="Gray-world corrected (used for recoloring)", use_column_width=True)

        # Target color input (no sliders)
        st.write("### Enter Target Color (RGB)")
        c1, c2, c3 = st.columns(3)
        with c1:
            r = st.number_input("Red (0–255)", min_value=0, max_value=255, value=243, step=1)
        with c2:
            g = st.number_input("Green (0–255)", min_value=0, max_value=255, value=224, step=1)
        with c3:
            b = st.number_input("Blue (0–255)", min_value=0, max_value=255, value=197, step=1)
        target_rgb = (int(r), int(g), int(b))

        # Recolor (preserve texture + lightness alignment)
        recolored = recolor_preserve_texture_with_L_match(wb, target_rgb)

        st.subheader("Recolored Image (Lightness Matched, High-Res)")
        st.image(recolored, caption=f"Recolored to RGB{target_rgb}", use_column_width=True)

        # Download high-res PNG
        buf = io.BytesIO()
        Image.fromarray(recolored).save(buf, format="PNG")
        st.download_button("Download High-Res PNG", buf.getvalue(),
                           "recolored_highres.png", "image/png")

        with st.expander("Debug info"):
            st.write({
                "shape": img.shape,
                "dtype": str(img.dtype),
                "target_rgb": target_rgb
            })

    except Exception as e:
        st.error("There was an error while processing the image.")
        st.exception(e)
        st.text_area("Traceback", value=traceback.format_exc(), height=240)
else:
    st.info("Upload a photo of the textured wall to begin.")
