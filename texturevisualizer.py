import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from streamlit_image_coordinates import streamlit_image_coordinates

st.title("Textured Paint Color Changer (Click-to-Pick Reference Color)")

# Upload image
uploaded_file = st.file_uploader("Upload a textured paint image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("Original Image - Click to select reference color")
    coords = streamlit_image_coordinates(img_array)

    st.write("### Enter Target Color (RGB)")
    r = st.number_input("Red (0-255)", min_value=0, max_value=255, value=255)
    g = st.number_input("Green (0-255)", min_value=0, max_value=255, value=0)
    b = st.number_input("Blue (0-255)", min_value=0, max_value=255, value=0)
    new_color_rgb = (r, g, b)

    # Determine reference color
    if coords:
        x, y = int(coords["x"]), int(coords["y"])
        ref_color = img_array[y, x]  # RGB from clicked pixel
        st.info(f"Reference color selected: {ref_color}")
    else:
        st.warning("Click on the image to select a reference color.")
        ref_color = np.mean(img_array, axis=(0, 1))  # fallback: average color

    # Convert image to Lab
    lab_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

    # Convert target color to Lab
    color_bgr = np.uint8([[new_color_rgb[::-1]]])
    color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)[0][0]

    # Convert reference color to Lab
    ref_bgr = np.uint8([[ref_color[::-1]]])
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)[0][0]

    # Compute scaling factor based on reference vs target brightness
    scale_L = color_lab[0] / max(ref_lab[0], 1)
    lab_img[:, :, 0] = np.clip(lab_img[:, :, 0] * scale_L, 0, 255)

    # Replace a and b channels with target color's chroma
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
