import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.title("Textured Paint Color Changer (Exact RGB Match, High-Res Output)")

# Upload image
uploaded_file = st.file_uploader("Upload a textured paint image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image in original resolution
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("Original Image")
    st.image(img_array, caption="Original", use_column_width=True)

    # RGB input
    st.write("### Enter Target Color (RGB)")
    r = st.number_input("Red (0-255)", min_value=0, max_value=255, value=213)
    g = st.number_input("Green (0-255)", min_value=0, max_value=255, value=224)
    b = st.number_input("Blue (0-255)", min_value=0, max_value=255, value=220)
    new_color_rgb = (r, g, b)

    # Convert image to Lab
    lab_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Convert target color to Lab
    color_bgr = np.uint8([[new_color_rgb[::-1]]])
    color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)[0][0]

    # Extract channels
    L = lab_img[:, :, 0]
    a = lab_img[:, :, 1]
    b_ = lab_img[:, :, 2]

    # Compute scaling for L channel
    mean_L = np.mean(L)
    target_L = color_lab[0]
    scale_L = target_L / mean_L

    # Apply adjustments
    L = np.clip(L * scale_L, 0, 255)
    a[:] = color_lab[1]  # Replace chroma completely
    b_[:] = color_lab[2]

    # Merge back
    lab_img[:, :, 0] = L
    lab_img[:, :, 1] = a
    lab_img[:, :, 2] = b_

    lab_img = lab_img.astype(np.uint8)

    # Convert back to RGB
    recolored_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)

    # Show high-resolution output
    st.subheader("Recolored Image (High-Res)")
    st.image(recolored_img, caption="Recolored", use_column_width=True)

    # Download button for high-res image
    img_pil = Image.fromarray(recolored_img)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(label="Download High-Res Image", data=byte_im, file_name="recolored_highres.png", mime="image/png")
