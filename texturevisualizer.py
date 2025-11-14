import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.title("Textured Paint Color Changer (Color + Texture Boost)")

# Upload image
uploaded_file = st.file_uploader("Upload a textured paint image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("Original Image")
    st.image(img_array, caption="Original", use_column_width=True)

    # Target color input
    st.write("### Enter Target Color (RGB)")
    r = st.number_input("Red (0-255)", min_value=0, max_value=255, value=255)
    g = st.number_input("Green (0-255)", min_value=0, max_value=255, value=0)
    b = st.number_input("Blue (0-255)", min_value=0, max_value=255, value=0)
    new_color_rgb = (r, g, b)

    # Sliders for control
    blend_ratio = st.slider("Color Blend Strength", 0.0, 1.0, 0.4)
    texture_strength = st.slider("Texture Strength", 0.0, 1.0, 0.5)

    # Convert image to Lab for color blending
    lab_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Convert target color to Lab
    color_bgr = np.uint8([[new_color_rgb[::-1]]])
    color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)[0][0]

    # Apply color blending
    lab_img[:, :, 0] = lab_img[:, :, 0] + (color_lab[0] - lab_img[:, :, 0]) * blend_ratio
    lab_img[:, :, 1] = lab_img[:, :, 1] + (color_lab[1] - lab_img[:, :, 1]) * blend_ratio
    lab_img[:, :, 2] = lab_img[:, :, 2] + (color_lab[2] - lab_img[:, :, 2]) * blend_ratio

    lab_img = np.clip(lab_img, 0, 255).astype(np.uint8)

    # Convert back to RGB
    recolored_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)

    # Enhance texture using CLAHE
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_texture = clahe.apply(gray)

    # Convert texture to 3-channel
    texture_3ch = cv2.merge([enhanced_texture] * 3)

    # Blend texture with recolored image
    final_img = cv2.addWeighted(recolored_img, 1.0, texture_3ch, texture_strength, 0)

    st.subheader("Recolored Image with Enhanced Texture")
    st.image(final_img, caption="Recolored + Texture Boost", use_column_width=True)

    # Download button
    img_pil = Image.fromarray(final_img)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(label="Download Image", data=byte_im, file_name="recolored_texture.png", mime="image/png")
