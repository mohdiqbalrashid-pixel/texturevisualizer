import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import math

st.title("Textured Paint Color Changer (Custom Delta E Accurate Matching)")

# --- Custom Delta E (CIEDE2000) Implementation ---
def delta_e_ciede2000(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    avg_L = (L1 + L2) / 2.0
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    avg_C = (C1 + C2) / 2.0

    G = 0.5 * (1 - math.sqrt((avg_C**7) / (avg_C**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = math.sqrt(a1p**2 + b1**2)
    C2p = math.sqrt(a2p**2 + b2**2)
    avg_Cp = (C1p + C2p) / 2.0

    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360

    avg_hp = h1p + h2p
    if abs(h1p - h2p) > 180:
        avg_hp += 360
    avg_hp /= 2.0

    T = 1 - 0.17 * math.cos(math.radians(avg_hp - 30)) + \
        0.24 * math.cos(math.radians(2 * avg_hp)) + \
        0.32 * math.cos(math.radians(3 * avg_hp + 6)) - \
        0.20 * math.cos(math.radians(4 * avg_hp - 63))

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    if abs(dhp) > 180:
        dhp -= 360 if dhp > 0 else -360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))

    SL = 1 + (0.015 * (avg_L - 50)**2) / math.sqrt(20 + (avg_L - 50)**2)
    SC = 1 + 0.045 * avg_Cp
    SH = 1 + 0.015 * avg_Cp * T

    RT = -2 * math.sqrt((avg_Cp**7) / (avg_Cp**7 + 25**7)) * \
         math.sin(math.radians(60 * math.exp(-((avg_hp - 275) / 25)**2)))

    delta_E = math.sqrt((dLp / SL)**2 + (dCp / SC)**2 + (dHp / SH)**2 + RT * (dCp / SC) * (dHp / SH))
    return delta_E

# --- Streamlit UI ---
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
    lab_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

    # Convert target color to Lab
    color_bgr = np.uint8([[new_color_rgb[::-1]]])
    color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)[0][0]

    # Initial recoloring
    mean_L = np.mean(lab_img[:, :, 0])
    target_Y = 0.299 * r + 0.587 * g + 0.114 * b
    lab_img[:, :, 0] = np.clip(lab_img[:, :, 0] * (target_Y / mean_L), 0, 255)
    lab_img[:, :, 1] = color_lab[1]
    lab_img[:, :, 2] = color_lab[2]

    # Iterative Delta E adjustment
    max_iterations = 10
    threshold = 2.0
    for _ in range(max_iterations):
        avg_L = np.mean(lab_img[:, :, 0])
        avg_a = np.mean(lab_img[:, :, 1])
        avg_b = np.mean(lab_img[:, :, 2])

        delta_e = delta_e_ciede2000((avg_L, avg_a, avg_b), (color_lab[0], color_lab[1], color_lab[2]))
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
