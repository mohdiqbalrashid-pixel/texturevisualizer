import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.title("Textured Paint Color Changer (Accurate Color Matching)")

# Upload image
uploaded_file = st.file_uploader("Upload a textured paint image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("Original Image")
    st.image(img_array, caption="Original", use_column_width=True)

    # Color picker
    new_color = st.color_picker("Pick a new color", "#FF0000")  # Default red
    # Convert hex to RGB
    new_color_rgb = tuple(int(new_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    # Convert image to Lab color space
    lab_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

    # Convert chosen color to Lab
    color_bgr = np.uint8([[new_color_rgb[::-1]]])  # RGB to BGR for OpenCV
    color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)[0][0]

    # Replace a and b channels with chosen color's a and b
    lab_img[:, :, 1] = color_lab[1]
    lab_img[:, :, 2] = color_lab[2]

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
