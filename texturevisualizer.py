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
st.title("🎨 AI-Assisted Textured Paint Recolor (White Balance + Soft Chroma Mix + Wall Mask)")

# ---------------- Utilities ----------------
def gray_world_white_balance(img_rgb: np.ndarray) -> np.ndarray:
    """
    Gray-world white balance: balance each channel toward a common gray target.
    Neutralizes warm/cool casts for more faithful recoloring.
    """
    img = img_rgb.astype(np.float32)
    means = img.mean(axis=(0, 1))               # [R,G,B]
    target = float(np.mean(means))
    gains = target / (means + 1e-6)
    balanced = img * gains
    return np.clip(balanced, 0, 255).astype(np.uint8)

def rgb_to_lab_color(rgb_tuple):
    """
    Convert (R,G,B) -> single LAB color in OpenCV's LAB scale (uint8, 0..255 per channel).
    """
    r, g, b = [int(max(0, min(255, v))) for v in rgb_tuple]
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[0, 0]  # (L,a,b)

def recolor_soft_chroma_mix_preserve_L(img_rgb: np.ndarray, target_rgb, mask: np.ndarray | None) -> np.ndarray:
    """
    Preserve texture by keeping original L (lightness) and softly mixing target chroma (a,b)
    with a portion of the original chroma. This avoids weird tints on near-neutral colors.
    If `mask` is provided (HxW bool), apply recolor only inside the mask.
    """
    lab_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L = lab_img[:, :, 0]  # keep original lightness
    orig_a = lab_img[:, :, 1].astype(np.float32)
    orig_b = lab_img[:, :, 2].astype(np.float32)

    tL, tA, tB = rgb_to_lab_color(target_rgb)
    tA, tB = float(tA), float(tB)

    # How neutral is the target? (OpenCV neutral center ~ (a,b) = (128,128))
    chroma_mag = np.hypot(tA - 128.0, tB - 128.0)
    if chroma_mag < 3.0:
        alpha = 0.55  # very neutral -> keep more of original chroma
    elif chroma_mag < 10.0:
        alpha = 0.70
    else:
        alpha = 0.85  # strong color but keep a bit of original for realism

    a_final = alpha * tA + (1.0 - alpha) * orig_a
    b_final = alpha * tB + (1.0 - alpha) * orig_b

    a_u8 = np.clip(a_final, 0, 255).astype(np.uint8)
    b_u8 = np.clip(b_final, 0, 255).astype(np.uint8)

    out_lab = lab_img.copy()
    if mask is None:
        out_lab[:, :, 1] = a_u8
        out_lab[:, :, 2] = b_u8
    else:
        out_lab[:, :, 1][mask] = a_u8[mask]
        out_lab[:, :, 2][mask] = b_u8[mask]

    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)

# ---- Mask decoding helper (handles both PIL Image and base64 providers) ----
def _decode_mask_to_bool(mask_field, target_shape):
    """
    Normalize provider mask to boolean array (H,W).
    Accepts a PIL.Image (L) or a base64 data URL string.
    """
    H, W = target_shape[:2]

    # Case: PIL Image object
    if hasattr(mask_field, "size"):
        m = np.array(mask_field.convert("L"))
    elif isinstance(mask_field, str):
        s = mask_field.split(",", 1)[1] if "," in mask_field else mask_field
        m = Image.open(io.BytesIO(base64.b64decode(s))).convert("L")
        m = np.array(m)
    else:
        return None

    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return m > 0

# ---------------- AI wall segmentation (Inference Providers via router) ----------------
HF_MODEL = "facebook/mask2former-swin-base-coco-panoptic"  # COCO panoptic, includes 'wall' label.  # [1](https://huggingface.co/facebook/mask2former-swin-base-coco-panoptic)

def get_wall_mask_with_hf(img_rgb: np.ndarray) -> np.ndarray | None:
    """
    Calls Hugging Face Inference Providers via InferenceClient and returns a boolean mask (HxW)
    for 'wall'. Returns None if token missing, provider unavailable, or no wall detected.
    """
    token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
    if not token:
        return None  # no token -> skip AI mask

    try:
        # Use router-compatible client (new Inference Providers flow)
        client = InferenceClient(provider="hf-inference", api_key=token)  # [2](https://huggingface.co/docs/inference-providers/main/tasks/image-segmentation)

        # IMPORTANT: send raw bytes (PNG), not a PIL.Image (Providers expect bytes/path/URL)
        pil_img = Image.fromarray(img_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        results = client.image_segmentation(
            img_bytes,
            model=HF_MODEL
            # parameters={"subtask": "panoptic"}  # optional; model implies panoptic
        )
        # Result is a list of {label, mask, score} segments  # [2](https://huggingface.co/docs/inference-providers/main/tasks/image-segmentation)[3](https://github.com/huggingface/hub-docs/blob/main/docs/inference-providers/tasks/image-segmentation.md)
        if not isinstance(results, list) or len(results) == 0:
            return None

        H, W = img_rgb.shape[:2]
        wall_mask = np.zeros((H, W), dtype=bool)

        # Prefer 'wall'
        found_wall = False
        for seg in results:
            label = str(seg.get("label", "")).lower()
            if "wall" in label:
                m = _decode_mask_to_bool(seg.get("mask"), img_rgb.shape)
                if m is not None:
                    wall_mask |= m
                    found_wall = True
        # Fallback to building/facade if no explicit wall
        if not found_wall:
            for seg in results:
                label = str(seg.get("label", "")).lower()
                if any(k in label for k in ("building", "facade")):
                    m = _decode_mask_to_bool(seg.get("mask"), img_rgb.shape)
                    if m is not None:
                        wall_mask |= m

        return wall_mask if wall_mask.any() else None

    except Exception as e:
        st.warning(f"Segmentation unavailable ({e}). Proceeding without AI mask.")
        return None

# ---------------- Streamlit UI ----------------
uploaded_file = st.file_uploader("Upload a textured paint image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load original (full-res)
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        st.subheader("Original Image")
        st.image(img_array, caption=f"Original ({img_array.shape[1]}×{img_array.shape[0]})", use_column_width=True)

        # White balance
        wb_img = gray_world_white_balance(img_array)
        st.subheader("White-Balanced Preview")
        st.image(wb_img, caption="Gray-world corrected (used for recoloring)", use_column_width=True)

        # Target RGB input (no sliders)
        st.write("### Enter Target Color (RGB)")
        c1, c2, c3 = st.columns(3)
        with c1:
            r = st.number_input("Red (0–255)", min_value=0, max_value=255, value=213, step=1)
        with c2:
            g = st.number_input("Green (0–255)", min_value=0, max_value=255, value=224, step=1)
        with c3:
            b = st.number_input("Blue (0–255)", min_value=0, max_value=255, value=220, step=1)
        target_rgb = (int(r), int(g), int(b))

        # AI wall segmentation (optional; requires token)
        mask = get_wall_mask_with_hf(wb_img)
        if mask is not None:
            st.success("AI wall mask detected — recoloring will apply only to wall area.")
            st.image((mask.astype(np.uint8) * 255), caption="Wall Mask", use_column_width=True)
        else:
            st.info("No AI mask — recolor will apply to the entire image.")

        # Recolor with preserved texture + soft chroma mix (masked if available)
        recolored_img = recolor_soft_chroma_mix_preserve_L(wb_img, target_rgb, mask)

        st.subheader("Recolored Image (High-Res, Texture Preserved)")
        st.image(recolored_img, caption=f"Recolored to RGB{target_rgb}", use_column_width=True)

        # Download high-res PNG
        buf = io.BytesIO()
        Image.fromarray(recolored_img).save(buf, format="PNG")
        st.download_button("Download High-Res PNG", buf.getvalue(), "recolored_highres.png", "image/png")

        with st.expander("Debug info"):
            st.write({
                "input_shape": img_array.shape,
                "has_hf_token": bool(st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))),
                "mask_applied": mask is not None,
                "seg_model": HF_MODEL
            })

    except Exception as e:
        st.error("There was an error running the app.")
        st.exception(e)
        st.text_area("Traceback", value=traceback.format_exc(), height=240)
else:
    st.info("Upload a textured wall photo to begin.\n"
            "Tip: add HF_TOKEN in Secrets to enable AI wall detection.")
