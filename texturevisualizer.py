import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.title("Textured Paint Color Changer (Optimized with Texture Boost)")

uploaded_file = st.file_uploader("Upload a textured paint image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Downscale for performance
    max_dim = 1024
    h, w = img_array.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_array = cv2.resize(img_array, (int(w * scale), int(h * scale)))

    st.subheader("Original Image")
    st.image(img_array, caption="Original", use_column_width=True)

    # Inputs
    st.write("### Enter Target Color (RGB)")
    r = st.number_input("Red (0-255)", min_value=0, max_value=255, value=255)
    g = st.number_input("Green (0-255)", min_value=0, max_value=255, value=0)
    b = st.number_input("Blue (0-255)", min_value=0, max_value=255, value=0)
    new_color_rgb = (r, g, b)

    blend_ratio = st.slider("Color Blend Strength", 0.0, 1.0, 0.4)
    texture_strength = st.slider("Texture Strength", 0.0, 1.0, 0.5)

    # Color blending in Lab
    lab_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
    color_bgr = np.uint8([[new_color_rgb[::-1]]])
    color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)[0][0]

    lab_img[:, :, 0] += (color_lab[0] - lab_img[:, :, 0]) * blend_ratio
    lab_img[:, :, 1] += (color_lab[1] - lab_img[:, :, 1]) * blend_ratio
    lab_img[:, :, 2] += (color_lab[2] - lab_img[:, :, 2]) * blend_ratio

    lab_img = np.clip(lab_img, 0, 255).astype(np.uint8)
    recolored_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    # Texture enhancement
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced_texture = clahe.apply(gray)
    texture_norm = enhanced_texture.astype(np.float32) / 255.0
    texture_norm = cv2.merge([texture_norm] * 3)

    # Multiply blend for texture
    final_img = recolored_img * (1 - texture_strength + texture_norm * texture_strength)
    final_img = np.clip(final_img * 255, 0, 255).astype(np.uint8)

    st.subheader("Recolored Image with Enhanced Texture")
    st.image(final_img, caption="Recolored + Texture Boost", use_column_width=True)

    # Download
    img_pil = Image.fromarray(final_img)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(label="Download Image", data=byte_im, file_name="recolored_texture.png", mime="image/png")
