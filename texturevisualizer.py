import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

st.title("Textured Paint Color Changer (Delta E Accurate Matching)")

# Upload image
uploaded_file = st.file_uploader("Upload a textured paint image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("Original Image")
    st.image(img_array, caption="Original", use_column_width=True)

    st.write("### Enter Target Color (RGB)")
    r = st.number_input("Red (0-255)", min_value=0, max_value=255, value=255)
    g = st.number_input("Green (0-255)", min_value=0, max_value=255, value=0)
    b = st.number_input("Blue (0-255)", min_value=0, max_value=255, value=0)
    new_color_rgb = (r, g, b)

    # Convert image to Lab color space
    lab_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

    # Convert target color to Lab
    color_bgr = np.uint8([[new_color_rgb[::-1]]])
    color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)[0][0]

    # Initial recoloring: replace a and b, scale L
    mean_L = np.mean(lab_img[:, :, 0])
    target_Y = 0.299 * r + 0.587 * g + 0.114 * b
    lab_img[:, :, 0] = np.clip(lab_img[:, :, 0] * (target_Y / mean_L), 0, 255)
    lab_img[:, :, 1] = color_lab[1]
    lab_img[:, :, 2] = color_lab[2]

    # Iterative Delta E adjustment
    max_iterations = 10
    threshold = 2.0  # Delta E threshold for perceptual match
    for _ in range(max_iterations):
        avg_L = np.mean(lab_img[:, :, 0])
        avg_a = np.mean(lab_img[:, :, 1])
        avg_b = np.mean(lab_img[:, :, 2])

        img_lab_color = LabColor(avg_L, avg_a, avg_b)
        target_lab_color = LabColor(color_lab[0], color_lab[1], color_lab[2])
        delta_e = delta_e_cie2000(img_lab_color, target_lab_color)

        if delta_e <= threshold:
            break

        # Adjust channels slightly toward target
        lab_img[:, :, 0] += (color_lab[0] - avg_L) * 0.1
        lab_img[:, :, 1] += (color_lab[1] - avg_a) * 0.1
        lab_img[:, :, 2] += (color_lab[2] - avg_b) * 0.1

        lab_img = np.clip(lab_img, 0, 255)

    # Convert back to RGB
    recolored_img = cv2.cvtColor(lab_img.astype(np.uint8), cv2.COLOR_LAB2RGB)

    st.subheader("Recolored Image")
    st.image(recolored_img, caption="Recolored", use_column_width=True)

    # Download button
    img_pil = Image.fromarray(recolored_img)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(label="Download Recolored Image", data=byte_im, file_name="recolored.png", mime="image/png")
