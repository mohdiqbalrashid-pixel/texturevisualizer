import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.title("Textured Paint Color Changer (Improved Color Matching)")

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

    # Convert to grayscale for texture
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray_3ch = cv2.merge([gray, gray, gray])

    # Create color layer
    color_layer = np.full_like(gray_3ch, new_color_rgb)

    # Intensity slider
    intensity = st.slider("Color Intensity", 0.0, 1.0, 0.8)

    # Blend using multiply (preserve texture)
    recolored_img = (gray_3ch.astype(np.float32) / 255.0) * (np.array(color_layer, dtype=np.float32))
    recolored_img = recolored_img * intensity + gray_3ch * (1 - intensity)
    recolored_img = np.clip(recolored_img, 0, 255).astype(np.uint8)

    st.subheader("Recolored Image")
    st.image(recolored_img, caption="Recolored", use_column_width=True)

    # Download button
    img_pil = Image.fromarray(recolored_img)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(label="Download Recolored Image", data=byte_im, file_name="recolored.png", mime="image/png")
