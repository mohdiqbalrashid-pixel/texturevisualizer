import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import traceback

st.set_page_config(page_title="Textured Paint Color Changer", page_icon="🎨", layout="centered")
st.title("🎨 Textured Paint Color Changer (Stable, High-Res)")

# ---------- Utilities ----------
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

def recolor_preserve_texture(img_rgb, target_rgb):
    """
    Recolor an RGB image by preserving the original L (lightness/texture) channel
    and replacing the a,b (chroma) channels with those of the target color in LAB space.
    Returns an RGB uint8 image, same resolution as input.
    """
    # Convert image to LAB (uint8, OpenCV scale)
    lab_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # Compute target LAB from the provided RGB
    target_lab = rgb_to_lab_color(target_rgb)  # uint8 (L,a,b)

    # Split channels (uint8)
    L = lab_img[:, :, 0]                 # Keep original lightness for texture/shading
    a = np.full_like(L, target_lab[1])   # Replace chroma a
    b = np.full_like(L, target_lab[2])   # Replace chroma b

    # Merge back and convert to RGB
    lab_merged = cv2.merge([L, a, b])
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

        # RGB inputs (no sliders, exact values)
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

        # Recolor with texture preserved
        recolored_img = recolor_preserve_texture(img_array, target_rgb)

        st.subheader("Recolored Image (High-Res, Texture Preserved)")
        st.image(recolored_img, caption=f"Recolored to RGB{target_rgb}", use_column_width=True)

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
    
# Gray-world white balance on the input before recolor
balanced = (img_array.astype(np.float32) / (img_array.mean(axis=(0,1)) + 1e-6)) * 128.0
balanced = np.clip(balanced, 0, 255).astype(np.uint8)
# Then recolor_preserve_texture(balanced, target_rgb)

orig_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
orig_a = orig_lab[:, :, 1].astype(np.float32)
orig_b = orig_lab[:, :, 2].astype(np.float32)
target_lab = rgb_to_lab_color(target_rgb)
a = (0.2 *.uint8)
b = (0.2 * orig_b + 0.8 * target_lab[2]).astype(np.uint8)
# Keep L from orig_lab[:, :, 0] as in the main code
