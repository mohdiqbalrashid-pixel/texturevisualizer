import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.title("Textured Paint Color Changer (Optimized Per-Pixel Adjustment)")

uploaded_file = st.file_uploader("Upload a textured paint image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("Original Image")
    st.image(img_array, caption="Original", use_column_width=True)

    st.write("### Enter Target Color (RGB)")
    r = st.number_input("Red (0-255)", min_value=0, max_value=255, value=255)
    g = st.number_input("Green (0-255)", min_value=0, max_value=255, value=0)
    b = st.number_input("Blue (0-255)", min_value=0, max_value=255, value=0)
    new_color_rgb = (r, g, b)

    # Convert image to Lab
    lab_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Convert target color to Lab
    color_bgr = np.uint8([[new_color_rgb[::-1]]])
    color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)[0][0]

    # Compute difference for all pixels
    diff_L = color_lab[0] - lab_img[:, :, 0]
    diff_a = color_lab[1] - lab_img[:, :, 1]
    diff_b = color_lab[2] - lab_img[:, :, 2]

    # Approximate Delta E (Euclidean distance)
    delta_e = np.sqrt(diff_L**2 + diff_a**2 + diff_b**2)

    # Normalize adjustment factor
    adjustment_factor = 0.2
    scale = adjustment_factor * (delta_e / (np.max(delta_e) + 1e-6))

    # Apply proportional adjustment
    lab_img[:, :, 0] += diff_L * scale
    lab_img[:, :, 1] += diff_a * scale
    lab_img[:, :, 2] += diff_b * scale

    lab_img = np.clip(lab_img, 0, 255).astype(np.uint8)

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
