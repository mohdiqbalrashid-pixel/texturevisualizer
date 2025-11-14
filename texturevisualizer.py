import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.title("Textured Paint Color Changer (Accurate Lab Blending)")

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
    blend_ratio = st.slider("Blend Strength", 0.0, 1.0, 0.4)
    brightness_adjust = st.slider("Brightness Adjustment", -50.0, 50.0, 0.0)

    # Convert image to Lab
    lab_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Normalize Lab values
    L = lab_img[:, :, 0] / 255 * 100
    a = lab_img[:, :, 1] - 128
    b_ = lab_img[:, :, 2] - 128

    # Convert target color to Lab and normalize
    color_bgr = np.uint8([[new_color_rgb[::-1]]])
    color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)[0][0]
    target_L = color_lab[0] / 255 * 100
    target_a = color_lab[1] - 128
    target_b = color_lab[2] - 128

    # Apply blending
    L = L + (target_L - L) * blend_ratio + brightness_adjust
    a = a + (target_a - a) * blend_ratio
    b_ = b_ + (target_b - b_) * blend_ratio

    # Convert back to OpenCV Lab scale
    lab_img[:, :, 0] = np.clip(L / 100 * 255, 0, 255)
    lab_img[:, :, 1] = np.clip(a + 128, 0, 255)
    lab_img[:, :, 2] = np.clip(b_ + 128, 0, 255)

    lab_img = lab_img.astype(np.uint8)

    # Convert back to RGB
    recolored_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)

    st.subheader("Recolored Image")
    st.image(recolored_img, caption="Recolored", use_column_width=True)

    # Download button
    img_pil = Image.fromarray(recolored_img)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(label="Download Recolored Image", data=byte_im, file_name="recolored.png", mime="image/png")
