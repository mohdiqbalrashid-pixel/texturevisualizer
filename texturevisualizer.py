import os
import io
import base64
import traceback
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from huggingface_hub import InferenceClient

st.set_page_config(page_title="AI Textured Paint Recolor", page_icon="🎨", layout="centered")
st.title("🎨 AI-Assisted Textured Paint Recolor (WB + Soft Chroma + Wall Mask)")

# ---------------- Utilities ----------------
def gray_world_white_balance(img_rgb: np.ndarray) -> np.ndarray:
    img = img_rgb.astype(np.float32)
    means = img.mean(axis=(0, 1))               # [R,G,B]
    target = float(np.mean(means))
    gains = target / (means + 1e-6)
    balanced = img * gains
    return np.clip(balanced, 0, 255).astype(np.uint8)

def rgb_to_lab_color(rgb_tuple):
    r, g, b = [int(max(0, min(255, v))) for v in rgb_tuple]
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[0, 0]  # (L,a,b) in OpenCV scale 0..255

def recolor_soft_chroma_mix_preserve_L(img_rgb: np.ndarray, target_rgb, mask: np.ndarray | None) -> np.ndarray:
    """
    Keep L (lightness/texture), mix target chroma (a,b) with a small portion of original chroma
    to avoid color casts for near-neutral shades. If mask is provided, recolor only inside mask.
    """
    lab_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L = lab_img[:, :, 0]  # keep
    orig_a = lab_img[:, :, 1].astype(np.float32)
    orig_b = lab_img[:, :, 2].astype(np.float32)

    tL, tA, tB = rgb_to_lab_color(target_rgb)
    tA, tB = float(tA), float(tB)

    # Neutrality check (OpenCV neutral center is (a,b) ≈ (128,128))
    chroma_mag = np.hypot(tA - 128.0, tB - 128.0)
    if chroma_mag < 3.0:
        alpha = 0.55
    elif chroma_mag < 10.0:
        alpha = 0.70
    else:
        alpha = 0.85

    a_final = alpha * tA + (1.0 - alpha) * orig_a
    b_final = alpha * tB + (1.0 - alpha) * orig_b

    a_final_u8 = np.clip(a_final, 0, 255).astype(np.uint8)
    b_final_u8 = np.clip(b_final, 0, 255).astype(np.uint8)

    out_lab = lab_img.copy()
    if mask is None:
        out_lab[:, :, 1] = a_final_u8
        out_lab[:, :, 2] = b_final_u8
    else:
        out_lab[:, :, 1][mask] = a_final_u8[mask]
        out_lab[:, :, 2][mask] = b_final_u8[mask]

    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)

# ---- Helpers to decode masks returned by InferenceClient.image_segmentation
def _decode_mask_to_bool(mask_field, target_shape):
    """
    mask_field can be:
      - a PIL Image (mode 'L') when using InferenceClient (most providers)
      - a base64 string "data:image/png;base64,...." for some providers
    Returns boolean mask (H,W)
    """
    H, W = target_shape[:2]

    # Case 1: PIL Image
    if hasattr(mask_field, "size"):
        m = np.array(mask_field.convert("L"))
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        return m > 0

    # Case 2: base64 string
    if isinstance(mask_field, str):
        s = mask_field
        if "," in s:
            s = s.split(",", 1)[1]
        try:
            mask_bytes = base64.b64decode(s)
            m = Image.open(io.BytesIO(mask_bytes)).convert("L")
            m = np.array(m)
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            return m > 0
        except Exception:
            return None

    return None

# ---------------- AI wall segmentation (via InferenceClient / router) ----------------
# Use a widely supported panoptic model so response includes labels+binary masks.
HF_MODEL = "facebook/mask2former-swin-base-coco-panoptic"  # 'wall' exists in COCO classes. [3](https://huggingface.co/facebook/mask2former-swin-base-coco-panoptic)

def get_wall_mask_with_hf(img_rgb: np.ndarray) -> np.ndarray | None:
    """
    Calls Hugging Face Inference Providers via InferenceClient (router-compatible)
    and returns a boolean mask of wall pixels (HxW). Returns None on failure.
    """
    token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
    if not token:
        # No token provided; skip AI mask
        return None

    try:
        client = InferenceClient(provider="hf-inference", api_key=token)  # new router-based flow [5](https://huggingface.co/docs/inference-providers/main/tasks/image-segmentation)
        pil_img = Image.fromarray(img_rgb)

        # You can pass subtask='panoptic' explicitly; most providers infer it from the model
        results = client.image_segmentation(pil_img, model=HF_MODEL)  # returns list of dicts with label/mask/score [5](https://huggingface.co/docs/inference-providers/main/tasks/image-segmentation)
        if not isinstance(results, list) or len(results) == 0:
            return None

        H, W = img_rgb.shape[:2]
        wall_mask = np.zeros((H, W), dtype=bool)

        # Combine segments labeled 'wall' (and optionally 'building' / 'facade' as fallback)
        keywords = ("wall",)
        fallback_keywords = ("building", "facade")
        found_wall = False

        for seg in results:
            label = str(seg.get("label", "")).lower()
            mask_field = seg.get("mask")
            if any(k in label for k in keywords):
                m = _decode_mask_to_bool(mask_field, img_rgb.shape)
                if m is not None:
                    wall_mask |= m
                    found_wall = True

        if not found_wall:
            for seg in results:
                label = str(seg.get("label", "")).lower()
                mask_field = seg.get("mask")
                if any(k in label for k in fallback_keywords):
                    m = _decode_mask_to_bool(mask_field, img_rgb.shape)
                    if m is not None:
                        wall_mask |= m

        return wall_mask if wall_mask.any() else None

    # If you accidentally hit the deprecated endpoint, HF returns 410 and guidance to use router
    # We rely on InferenceClient which already targets the router, but we still guard for any provider errors.
    except Exception as e:
        st.warning(f"Segmentation unavailable ({e}). Proceeding without AI mask.")
        return None

# ---------------- App ----------------
uploaded_file = st.file_uploader("Upload a textured paint image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # 1) Load original (full-res)
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        st.subheader("Original Image")
        st.image(img_array, caption=f"Original ({img_array.shape[1]}×{img_array.shape[0]})", use_column_width=True)

        # 2) White balance
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

        # 4) AI wall segmentation (router via InferenceClient). If missing or fails -> None
        mask = get_wall_mask_with_hf(wb_img)
        if mask is not None:
            st.success("AI wall mask detected — recoloring will apply to the wall area.")
            st.image((mask.astype(np.uint8) * 255), caption="Wall Mask", use_column_width=True)
        else:
            st.info("No AI mask — applying recolor to the entire image.")

        # 5) Recolor with preserved L + soft chroma mix; apply mask if available
        recolored_img = recolor_soft_chroma_mix_preserve_L(wb_img, target_rgb, mask)

        st.subheader("Recolored Image (High-Res, Texture Preserved)")
        st.image(recolored_img, caption=f"Recolored to RGB{target_rgb}", use_column_width=True)

        # 6) Download PNG
        buf = io.BytesIO()
        Image.fromarray(recolored_img).save(buf, format="PNG")
        st.download_button("Download High-Res PNG", buf.getvalue(), "recolored_highres.png", "image/png")

        # Debug info
        with st.expander("Debug info"):
            st.write({
                "input_shape": img_array.shape,
                "has_hf_token": bool(st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))),
                "mask_applied": mask is not None,
                "model": HF_MODEL
            })

    except Exception as e:
        st.error("There was an error running the app.")
        st.exception(e)
        st.text_area("Traceback", value=traceback.format_exc(), height=240)
else:
    st.info("Upload a textured wall photo to begin.\nTip: set your Hugging Face token in Secrets as HF_TOKEN to enable AI wall detection.")
