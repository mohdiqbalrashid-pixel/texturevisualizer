
import os
import io
import base64
import json
import traceback
import requests
import numpy as np
import cv2
import streamlit as st
from PIL import Image

st.set_page_config(page_title="AI Textured Paint Recolor", page_icon="🎨", layout="centered")
st.title("🎨 AI-Assisted Textured Paint Recolor (White Balance + Soft Chroma)")

# ---------------- Utilities ----------------

def gray_world_white_balance(img_rgb: np.ndarray) -> np.ndarray:
    """Gray-world WB: balance each channel toward a common gray target."""
    img = img_rgb.astype(np.float32)
    means = img.mean(axis=(0, 1))              # [R,G,B]
    target = float(np.mean(means))
    eps = 1e-6
    gains = target / (means + eps)
    balanced = img * gains
    return np.clip(balanced, 0, 255).astype(np.uint8)

def rgb_to_lab_color(rgb_tuple):
    """(R,G,B)-> LAB (OpenCV scale uint8)."""
    r, g, b = [int(max(0, min(255, v))) for v in rgb_tuple]
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[0, 0]

def recolor_soft_chroma_mix_preserve_L(img_rgb: np.ndarray, target_rgb, mask: np.ndarray | None) -> np.ndarray:
    """
    Preserve texture: keep original L (lightness) and apply a soft mix of target chroma (a,b).
    If mask is provided (bool HxW), recolor only inside mask.
    """
    lab_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L = lab_img[:, :, 0]                       # keep
    orig_a = lab_img[:, :, 1].astype(np.float32)
    orig_b = lab_img[:, :, 2].astype(np.float32)

    tL, tA, tB = rgb_to_lab_color(target_rgb)
    tA = float(tA); tB = float(tB)

    # How neutral is the target? (OpenCV neutral ~ (a,b)=(128,128))
    chroma_mag = np.hypot(tA - 128.0, tB - 128.0)
    if chroma_mag < 3.0:
        alpha = 0.55  # very neutral: retain more original chroma to avoid tint
    elif chroma_mag < 10.0:
        alpha = 0.70
    else:
        alpha = 0.85

    a_final = alpha * tA + (1.0 - alpha) * orig_a
    b_final = alpha * tB + (1.0 - alpha) * orig_b

    a_final_u8 = np.clip(a_final, 0, 255).astype(np.uint8)
    b_final_u8 = np.clip(b_final, 0, 255).astype(np.uint8)

    lab_merged = cv2.merge([L, a_final_u8, b_final_u8])

    if mask is not None:
        # Only apply a,b inside mask: outside, keep original a,b
        out_lab = lab_img.copy()
        out_lab[:, :, 1][mask] = lab_merged[:, :, 1][mask]
        out_lab[:, :, 2][mask] = lab_merged[:, :, 2][mask]
        lab_merged = out_lab

    recolored_rgb = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2RGB)
    return recolored_rgb

# ---------------- AI Segmentation (Hugging Face) ----------------

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/maskformer-swin-base-coco"
# Put your token in Streamlit -> Settings -> Secrets as HF_TOKEN
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))

def decode_mask_to_bool(mask_str: str, target_shape) -> np.ndarray:
    """
    HF returns a base64-encoded PNG mask. Convert to boolean mask (HxW).
    The string may be 'data:image/png;base64,...' or raw base64.
    """
    if "," in mask_str:
        mask_str = mask_str.split(",", 1)[1]
    mask_bytes = base64.b64decode(mask_str)
    m = Image.open(io.BytesIO(mask_bytes)).convert("L")
    m = np.array(m)
    # Ensure size matches image (some models return original; HF typically does)
    if m.shape != target_shape[:2]:
        m = cv2.resize(m, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 0  # boolean

def ai_wall_mask_hf(img_rgb: np.ndarray) -> np.ndarray | None:
    """
    Call HF Inference API to get segmentation, return combined boolean mask for 'wall'.
    Falls back to None on error (caller should handle gracefully).
    """
    if not HF_TOKEN:
        return None

    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        # Send original PNG bytes
        buf = io.BytesIO()
        Image.fromarray(img_rgb).save(buf, format="PNG")
        resp = requests.post(HF_API_URL, headers=headers, data=buf.getvalue(), timeout=60)

        if resp.status_code != 200:
            st.warning(f"Segmentation API returned {resp.status_code}: {resp.text[:200]}")
            return None

        data = resp.json()
        if not isinstance(data, list):
            st.warning("Unexpected segmentation response format.")
            return None

        # Combine all masks with 'wall' in label (case-insensitive)
        H, W = img_rgb.shape[:2]
        wall_mask = np.zeros((H, W), dtype=bool)
        for item in data:
            label = str(item.get("label", "")).lower()
            mask_str = item.get("mask", "")
            if "wall" in label and mask_str:
                m = decode_mask_to_bool(mask_str, img_rgb.shape)
                wall_mask |= m

        # If no 'wall' detected, try fallback: 'building' surfaces
        if not wall_mask.any():
            for item in data:
                label = str(item.get("label", "")).lower()
                mask_str = item.get("mask", "")
                if any(k in label for k in ["building", "house", "facade"]) and mask_str:
                    m = decode_mask_to_bool(mask_str, img_rgb.shape)
                    wall_mask |= m

        return wall_mask if wall_mask.any() else None

    except Exception as e:
        st.warning(f"Segmentation request failed: {e}")
        return None

# ---------------- App ----------------

uploaded_file = st.file_uploader("Upload a textured paint image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # 1) Load original (full-res)
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)  # HxWx3 uint8

        st.subheader("Original Image")
        st.image(img_array, caption=f"Original ({img_array.shape[1]}×{img_array.shape[0]})", use_column_width=True)

        # 2) White balance correction
        wb_img = gray_world_white_balance(img_array)
        st.subheader("White-Balanced Preview")
        st.image(wb_img, caption="Gray-world corrected (used for recoloring)", use_column_width=True)

        # 3) Target RGB (no sliders)
        st.write("### Enter Target Color (RGB)")
        c1, c2, c3 = st.columns(3)
        with c1:
            r = st.number_input("Red (0–255)", min_value=0, max_value=255, value=213, step=1)
        with c2:
            g = st.number_input("Green (0–255)", min_value=0, max_value=255, value=224, step=1)
        with c3:
            b = st.number_input("Blue (0–255)", min_value=0, max_value=255, value=220, step=1)
        target_rgb = (int(r), int(g), int(b))

        # 4) AI segmentation (optional)
        mask = None
        if HF_TOKEN:
            st.info("Running AI wall segmentation…")
            mask = ai_wall_mask_hf(wb_img)
            if mask is not None:
                st.success("Wall detected. Recoloring will apply only to the wall.")
                st.image((mask * 255).astype(np.uint8), caption="Wall Mask", use_column_width=True)
            else:
                st.warning("No wall found (or API unavailable). Recoloring will apply to the whole image.")

        # 5) Recolor with preserved texture + soft chroma mix (masked if available)
        recolored_img = recolor_soft_chroma_mix_preserve_L(wb_img, target_rgb, mask)

        st.subheader("Recolored Image (High-Res, Texture Preserved)")
        st.image(recolored_img, caption=f"Recolored to RGB{target_rgb}", use_column_width=True)

        # 6) Download
        buf = io.BytesIO()
        Image.fromarray(recolored_img).save(buf, format="PNG")
        st.download_button(
            label="Download High-Res PNG",
            data=buf.getvalue(),
            file_name="recolored_highres.png",
            mime="image/png"
        )

        # Debug info
        with st.expander("Debug info"):
            st.write({
                "input_shape": img_array.shape,
                "has_hf_token": bool(HF_TOKEN),
                "mask_applied": mask is not None,
                "target_rgb": target_rgb
            })

    except Exception as e:
        st.error("There was an error running the app while processing the RGB recolor.")
        st.exception(e)
        st.text_area("Traceback", value=traceback.format_exc(), height=240)
else:
    st.info("Upload a photo of the textured paint surface to begin.\n"
            "Tip: Add your Hugging Face token as HF_TOKEN in Streamlit Secrets to enable AI wall detection.")
