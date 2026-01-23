import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import traceback

st.set_page_config(page_title="Textured Paint Color Changer", page_icon="🎨", layout="centered")
st.title("🎨 Textured Paint Color Changer (White Balance + Soft Chroma Mix)")

# ---------- Utilities ----------
def gray_world_white_balance(img_rgb: np.ndarray) -> np.ndarray:
    """
    Gray-world white balance: scale each RGB channel by a gain so their means
    converge toward a common gray target (overall mean). This neutralizes warm/cool casts.
    """
    img = img_rgb.astype(np.float32)
    # Compute per-channel means (R, G, B)
    means = img.mean(axis=(0, 1))  # shape (3,), RGB order
    # Overall gray target = average of the three means
    target = float(np.mean(means))
    eps = 1e-6  # avoid division by zero

    gains = target / (means + eps)  # shape (3,)
    balanced = img * gains  # broadcast multiply per-channel

    # Clip and return as uint8
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)
    return balanced

def rgb_to_lab_color(rgb_tuple):
    """
    Convert an (R,G,B) tuple (0-255) to a single LAB color using OpenCV.
    Returns a 3-element uint8 array in OpenCV's LAB scale (L,a,b in 0..255).
    """
    r, g, b = [int(max(0, min(255, v))) for v in rgb_tuple]  # clamp and cast to int
    # Make a (1,1,3) BGR image for OpenCV
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)  # shape (1,1,3), dtype uint8
    return lab[0, 0]  # (L,a,b)

def recolor_soft_chroma_mix_preserve_L(img_rgb: np.ndarray, target_rgb) -> np.ndarray:
    """
    Recolor an RGB image while preserving texture:
      • Keep original L (lightness) channel for shading/texture.
      • Mix target a,b (chroma) with a small portion of original a,b to avoid tints.
    Adaptive mix:
      • Near-neutral target colors (very low chroma): keep more of original chroma.
      • Strong target colors: use a higher fraction of the target chroma.
    Returns an RGB uint8 image, same resolution as input.
    """
    # Convert image (already white-balanced) to LAB
    lab_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # Original channels
    L = lab_img[:, :, 0]                                  # uint8 0..255 (keep)
    orig_a = lab_img[:, :, 1].astype(np.float32)          # float for math
    orig_b = lab_img[:, :, 2].astype(np.float32)

    # Target LAB (uint8 0..255)
    target_lab = rgb_to_lab_color(target_rgb)
    tL, tA, tB = int(target_lab[0]), float(target_lab[1]), float(target_lab[2])

    # Determine how "neutral" the target is in LAB (OpenCV scale: neutral ~ a=b=128)
    # Chroma magnitude from neutral point (128,128)
    chroma_mag = np.hypot(tA - 128.0, tB - 128.0)

    # Adaptive mixing amount (alpha): fraction of target chroma we apply
    # - Very low chroma (near gray): use smaller alpha to avoid color casts
    # - Moderate chroma: medium alpha
    # - Strong chroma: higher alpha, but still < 1 for stability
    if chroma_mag < 3.0:
        alpha = 0.55  # very neutral; keep more original chroma
    elif chroma_mag < 10.0:
        alpha = 0.70
    else:
        alpha = 0.85  # strong color but still keep some original to preserve realism

    # Compute final a,b as soft mix (broadcast over image)
    # a_final = alpha * target_a + (1-alpha) * original_a
    a_final = alpha * tA + (1.0 - alpha) * orig_a
    b_final = alpha * tB + (1.0 - alpha) * orig_b

    # Clip to valid OpenCV LAB range and cast back to uint8
    a_final_u8 = np.clip(a_final, 0, 255).astype(np.uint8)
    b_final_u8 = np.clip(b_final, 0, 255).astype(np.uint8)

    # Merge preserved L with mixed a,b and convert to RGB
    lab_merged = cv2.merge([L, a_final_u8, b_final_u8])
    recolored_rgb = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2RGB)

    return recolored_rgb

# ---------- App ----------
uploaded_file = st.file_uploader("Upload a textured paint image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load original image (full resolution)
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)  # HxWx3 uint8

        st.subheader("Original Image")
        st.image(img_array, caption=f"Original ({img_array.shape[1]}×{img_array.shape[0]})", use_column_width=True)

        # 1) Automatic white balance correction (gray-world)
        wb_img = gray_world_white_balance(img_array)

        st.subheader("White-Balanced Preview")
        st.image(wb_img, caption="Gray-world corrected (used for recoloring)", use_column_width=True)

        # 2) RGB inputs (no sliders, exact values)
        st.write("### Enter Target Color (RGB)")
        c1, c2, c3 = st.columns(3)
        with c1:
            r = st.number_input("Red (0–255)", min_value=0, max_value=255, value=213, step=1)
        with c2:
            g = st.number_input("Green (0–255)", min_value=0, max_value=255, value=224, step=1)
        with c3:
            b = st.number_input("Blue (0–255)", min_value=0, max_value=255, value=220, step=1)

        # Force ints and clamp safely
        r, g, b = int(r), int(g), int(b)
        target_rgb = (r, g, b)

        # 3) Recolor with preserved L and soft chroma mix
        recolored_img = recolor_soft_chroma_mix_preserve_L(wb_img, target_rgb)

        st.subheader("Recolored Image (High-Res, Texture Preserved)")
        st.image(recolored_img, caption=f"Recolored to RGB{target_rgb} (Soft Chroma Mix)", use_column_width=True)

        # Download (PNG to avoid quality loss)
        buf = io.BytesIO()
        Image.fromarray(recolored_img).save(buf, format="PNG")
        st.download_button(
            label="Download High-Res PNG",
            data=buf.getvalue(),
            file_name="recolored_highres.png",
            mime="image/png"
        )

        # Optional debug info toggle
        with st.expander("Debug info"):
            st.write({
                "input_shape": img_array.shape,
                "input_dtype": str(img_array.dtype),
                "target_rgb": target_rgb,
            })

    except Exception as e:
        st.error("There was an error running the app while processing the RGB recolor.")
        st.exception(e)  # show full traceback in the UI for fast debugging
        st.text_area("Traceback", value=traceback.format_exc(), height=240)
else:
    st.info("Upload a photo of the textured paint surface to begin.")
