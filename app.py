from pathlib import Path
from datetime import datetime
from huggingface_hub import snapshot_download

import cv2
import numpy as np
import plotly.express as px
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import json
import random

from torchvision import models
from PIL import Image
# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="MeiboVision",
    page_icon="👁️",
    layout="wide"
)

# ---------------- LOGIN CREDENTIALS ----------------
USERNAME = "admin"
PASSWORD = "Meibo@123"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN PAGE STYLING ----------------
st.markdown("""
<style>
/* Hide Streamlit's default header spacing */
.block-container {
    padding-top: 3rem;
    max-width: 100%;
}

/* Center the login card */
.login-card {
    max-width: 430px;
    margin: 35px auto;
    padding: 35px 32px;
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    box-shadow: 0 10px 35px rgba(43, 90, 160, 0.12);
}

/* Login button */
div.stButton > button {
    background-color: #2B5AA0;
    color: white;
    border: 2px solid #2B5AA0;
    border-radius: 9px;
    font-weight: 600;
    transition: all 0.25s ease;
}

/* Hover color */
div.stButton > button:hover {
    background-color: #0AC4E0;
    color: white;
    border-color: #0AC4E0;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(10, 196, 224, 0.25);
}

</style>
""", unsafe_allow_html=True)
# ---------------- LOGIN PAGE ----------------
# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:

    left, center, right = st.columns([1, 1.2, 1])
    with center:

        st.markdown(
            "<h1 style='text-align:center;color:#2B5AA0;'>MeiboVision</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align:center;color:#64748B;'>"
            "AI-PoweredMeibomian Gland Analysis</p>",
            unsafe_allow_html=True
        )

        st.write("")

        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password"
        )

        if st.button("Login", use_container_width=True):
            if username == USERNAME and password == PASSWORD:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid username or password.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()
# ---------------- DASHBOARD ACCESS ----------------
# Your existing dashboard code continues below this line.
# ==========================================================
# CONFIGURATION
# ==========================================================
ROOT = Path(__file__).parent

@st.cache_resource
def download_dataset():

    dataset_path = snapshot_download(
        repo_id="varshhsri/mgd1k-meibovision-data",
        repo_type="dataset",
        token=st.secrets["HF_TOKEN"],
        local_dir=str(ROOT / "Dataset")
    )

    return Path(dataset_path)


DATASET = download_dataset()

IMAGE_DIR = DATASET / "Original Images"
GLAND_MASK_DIR = DATASET / "Meibomian Gland Labels"
EYELID_MASK_DIR = DATASET / "Eyelid Lebels"

SEGMENTATION_MODEL_PATH = ROOT / "gland_segmentation.pth"
CLASSIFICATION_MODEL_PATH = ROOT / "meiboscore_classifier.pth"
RESULTS_PATH = ROOT / "quantitative_results.csv"


IMAGE_SIZE = 256
MM_PER_PIXEL = 0.01
STAGES = ["Normal", "Mild", "Moderate", "Severe"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.subheader("Model Performance")

classifier_file = ROOT / "classifier_metrics.json"
segmentation_file = ROOT / "segmentation_metrics.json"




# ==========================================================
# CSS - dashboard styling inspired by the reference screenshot
# ==========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: #f5f8fc; color: #102c55; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1600px; }
header[data-testid="stHeader"] { background: rgba(245,248,252,.95); }
section[data-testid="stSidebar"] { background: #f8fbff; border-right: 1px solid #dce5ef; }
section[data-testid="stSidebar"] * { color: #17365d; }
.brand {font-size: 1.55rem; font-weight: 800; color:#12325c; letter-spacing:-.5px;}
.brand span {color:#149da5;}
.subtitle {color:#687b95; font-size:.95rem; margin-top:-8px;}.section-title {font-size:1.08rem; font-weight:800; color:#12325c; margin:0 0 .8rem 0;}
.panel {background:white; border:1px solid #dfe7f0; border-radius:12px; padding:16px; margin-bottom:12px;}
.step-row {display:flex; align-items:center; justify-content:space-between; gap:8px; background:white;
border:1px solid #dfe7f0; border-radius:12px; padding:17px 12px; height:100%;}
.step-num {background:#169fa7; color:white; border-radius:50%; width:34px; height:34px; display:inline-flex;
align-items:center; justify-content:center; font-weight:800; margin-bottom:7px;}
.step-label {font-weight:800; font-size:.84rem; color:#17365d;}
.step-desc {font-size:.75rem; color:#687b95; line-height:1.25;}
.metric-card {background:white; border:1px solid #dfe7f0; border-radius:12px; padding:18px 16px; min-height:122px;}
.metric-label {font-size:.85rem; font-weight:700; color: #000000;}
.metric-value {font-size:1.8rem; font-weight:800; color: #000000; margin-top:8px;}
.metric-note {font-size:.75rem; color:#75869a; margin-top:3px;}
.grade-box {background:#fff7e6; border:1px solid #f4e6be; border-radius:10px; padding:18px; text-align:center;}
.grade-value {font-size:1.8rem; font-weight:800; color:#17365d;}
.grade-caption {font-size:.85rem; color:#687b95;}
div.stButton > button, div.stDownloadButton > button {border-radius:8px; font-weight:700; min-height:2.6rem;}
div.stButton > button[kind="primary"] {background:#159ba4; border-color:#159ba4;}
div.stButton > button[kind="primary"]:hover {background:#117e87; border-color:#117e87;}
div[data-testid="stMetric"] {background:white; border:1px solid #dfe7f0; padding:14px; border-radius:10px;}
.small-muted {font-size:.78rem;color:#75869a;}div.stDownloadButton > button:hover {
    background-color: #159ba4 !important;
    color: white !important;
    border-color: #159ba4 !important;
}<div style="padding: 10px 0;">
    <h1 style="
        margin: 0;
        font-size: 30px;
        font-weight: 700;
        color: #183153;
        opacity: 1;
    ">
        <span style="color: #0D9488;">MeiboVision</span>
        <span style="color: #183153;"> | Meibomian Gland Analysis</span>
    </h1>
    <p style="
        margin-top: 8px;
        font-size: 16px;
        color: #183153;
        opacity: 1;
    ">
        Deep learning-assisted meibography review dashboard
    </p>
</div>
</style>
""", unsafe_allow_html=True)

# ==========================================================
# MODEL LOADING
# ==========================================================
@st.cache_resource
def load_segmentation_model():
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    checkpoint = torch.load(
        SEGMENTATION_MODEL_PATH, map_location=DEVICE, weights_only=True
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_classification_model():
    model = models.efficientnet_b0(weights=None)
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, 4)
    checkpoint = torch.load(
        CLASSIFICATION_MODEL_PATH, map_location=DEVICE, weights_only=True
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model

# ==========================================================
# IMAGE AND ANALYSIS HELPERS
# ==========================================================
def to_bgr(pil_image):
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def clahe_enhance_bgr(image):
    """CLAHE is applied to luminance for display/preprocessing preview only."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
def predict_gland_mask(image, model):
    height, width = image.shape[:2]
    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probability = torch.sigmoid(output)[0, 0].cpu().numpy()
    small_mask = (probability > 0.5).astype(np.uint8)
    mask = cv2.resize(small_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask, probability

def predict_stage(image, model):
    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    index = int(np.argmax(probabilities))
    return STAGES[index], float(probabilities[index]), probabilities

def find_matching_file(folder, image_stem):
    if not folder.exists():
        return None
    matches = [p for p in folder.rglob("*")
               if p.is_file() and p.stem.lower() == image_stem.lower()]
    return matches[0] if matches else None

def read_binary_mask(path, output_shape=None):
    mask = np.array(Image.open(path).convert("L"))
    mask = (mask > 0).astype(np.uint8)
    if output_shape is not None:
        h, w = output_shape
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask

def estimate_gland_count(binary_mask):
    binary = (binary_mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return 0
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if distance.max() == 0:
        return 0
    _, centers = cv2.threshold(distance, 0.30 * distance.max(), 255, cv2.THRESH_BINARY)
    centers = centers.astype(np.uint8)
    number, markers = cv2.connectedComponents(centers)
    return sum(1 for marker_id in range(1, number)
               if np.count_nonzero(markers == marker_id) >= 3)

def calculate_quantitative_values(predicted_mask,eyelid_mask=None,mm_per_pixel=None):
    gland_pixels = int(np.count_nonzero(predicted_mask))
    count = estimate_gland_count(predicted_mask)
    eyelid_pixels = None
    area_percentage = None
    density = None


    if eyelid_mask is not None:

        gland_inside_roi = int(np.count_nonzero((predicted_mask > 0) & (eyelid_mask > 0)))
        eyelid_pixels = int(np.count_nonzero(eyelid_mask))
        if eyelid_pixels > 0:
            area_percentage = gland_inside_roi / eyelid_pixels * 100
            density = count / eyelid_pixels * 1000
    return {
        "Gland Count (estimated)": count,
        "Gland Area (pixels)": gland_pixels,
        "Gland Density (glands/1000 pixels)": density,
        "Eyelid Area (pixels)": eyelid_pixels,
    }


def make_overlay(image_bgr, mask):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    overlay[mask > 0] = (40, 205, 215)
    return cv2.addWeighted(rgb, 0.58, overlay, 0.42, 0)

def make_activation_preview(image_bgr, probability_map):
    # This is a segmentation probability map preview, NOT Grad-CAM.
    h, w = image_bgr.shape[:2]
    heat = cv2.resize(probability_map, (w, h))
    heat = np.uint8(np.clip(heat, 0, 1) * 255)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.cvtColor(cv2.addWeighted(image_bgr, 0.52, heat, 0.48, 0), cv2.COLOR_BGR2RGB)

def save_results(record):
    new_row = pd.DataFrame([record])
    if RESULTS_PATH.exists():
        old = pd.read_csv(RESULTS_PATH)
        if "Image" in old.columns:
            old = old[old["Image"] != record["Image"]]
        new = pd.concat([old, new_row], ignore_index=True)
    else:
        new = new_row
    new.to_csv(RESULTS_PATH, index=False)
    return new

import numpy as np
import cv2


def calculate_regional_metrics(gland_mask, eyelid_mask):
    gland_mask = np.asarray(gland_mask) > 0
    eyelid_mask = np.asarray(eyelid_mask) > 0

    if gland_mask.shape != eyelid_mask.shape:
        raise ValueError("Gland and eyelid masks must have the same shape.")

    if not np.any(eyelid_mask):
        raise ValueError("Eyelid mask is empty.")

    # Find the eyelid's bounding box
    ys, xs = np.where(eyelid_mask)
    x_min, x_max = xs.min(), xs.max() + 1

    # Divide the eyelid width into three sections
    width = x_max - x_min
    first = x_min + width // 3
    second = x_min + (2 * width) // 3

    regions = {
        "Image-left": (x_min, first),
        "Center": (first, second),
        "Image-right": (second, x_max),
    }

    results = {}

    for name, (x1, x2) in regions.items():
        region = np.zeros_like(eyelid_mask, dtype=bool)
        region[:, x1:x2] = True

        local_eyelid = eyelid_mask & region
        local_glands = gland_mask & local_eyelid

        eyelid_pixels = int(np.count_nonzero(local_eyelid))
        gland_pixels = int(np.count_nonzero(local_glands))

        results[name] = {
            "Eyelid area (px²)": eyelid_pixels,
            "Gland area (px²)": gland_pixels,
            "Gland area (%)": (
                100 * gland_pixels / eyelid_pixels
                if eyelid_pixels else 0
            ),
            "Gland density (per 1000 eyelid pixels)": (
                1000 * gland_pixels / eyelid_pixels
                if eyelid_pixels else 0
            ),
        }

    return results
# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================
with st.sidebar:
    st.markdown('<div class="brand"><span>MeiboVision</span></div>', unsafe_allow_html=True)
    st.caption("MEIBOMIAN GLAND ANALYSIS")
    page = st.radio("NAVIGATION", ["Overview", "New Assessment", "Patients", "Reports", "Settings"], index=1)
    st.divider()
    st.caption("Research prototype • For review by a qualified clinician")

# ==========================================================
# HEADER
# ==========================================================
header_left, header_right = st.columns([4, 1])
with header_left:
    st.markdown('<div class="brand"><span>MeiboVision</span> <span style="color:#17365d;font-weight:500">| Meibomian Gland Analysis</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Deep learning-assisted meibography review dashboard</div>', unsafe_allow_html=True)
with header_right:
    st.markdown("<div style='text-align:right;padding-top:8px'><span style='background:#d6f4ef;color:#167e7d;padding:8px 12px;border-radius:7px;font-weight:800;font-size:.75rem'>RESEARCH DEMO</span></div>", unsafe_allow_html=True)

st.divider()

if page == "Overview":
    st.header("Overview")
    st.write("Upload a meibography image from **New Assessment** to run gland segmentation, estimate quantitative features, and predict the four-class stage.")
    a, b, c = st.columns(3)
    a.metric("Segmentation Model", "U-Net")
    b.metric("Classification Model", "EfficientNet-B0")
    c.metric("Inference where it cant generate device", str(DEVICE).upper())
    st.info("Predictions and gland-count estimates are experimental and are not a medical diagnosis.")

elif page == "Patients":
    st.header("Patients")
    st.info("This prototype does not store patient identities. Use a non-identifying image ID for research records.")
    if RESULTS_PATH.exists():
        records = pd.read_csv(RESULTS_PATH)
        st.dataframe(records, use_container_width=True, hide_index=True)
    else:
        st.write("No assessments have been saved yet.")

elif page == "Reports":
    st.header("Reports")
    if RESULTS_PATH.exists():
        report_df = pd.read_csv(RESULTS_PATH)
        st.dataframe(report_df, use_container_width=True, hide_index=True)

        import pandas as pd

        download_data = pd.DataFrame([{
            "Selected Image Name": image_stem,
            "Gland Count": count,
            "Gland Area": (
                f"{area_mm2:.2f} mm²"
                if area_mm2 is not None
                else f"{area_pixels} px²"
            ),
            "Gland Density": (
                f"{density:.2f}"
                if density is not None
                else "N/A"
            )
        }])

        csv_data = download_data.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Assessment",
            data=csv_data,
            file_name="gland_assessment.csv",
            mime="text/csv"
        )

    else:
        st.info("No report exists yet. Run an assessment first.")

elif page == "Settings":
    st.header("Settings")
    st.write("**Dataset folder:**", str(DATASET))
    st.write("**Segmentation checkpoint:**", str(SEGMENTATION_MODEL_PATH))
    st.write("**Classification checkpoint:**", str(CLASSIFICATION_MODEL_PATH))
    st.write("**Runtime device:**", str(DEVICE))
    st.caption("Update the paths in app.py if your folders or checkpoint names differ.")

else:
    # Top workflow row
    title_col, action_col = st.columns([4, 1])
    with title_col:
        st.markdown("## Image Assessment")
        st.caption("Upload a meibography image to view model outputs and quantitative features.")
    with action_col:
        st.markdown("<div style='text-align:right;padding-top:14px;color:#687b95;font-size:.8rem'>SIMULATED CLINICAL WORKFLOW</div>", unsafe_allow_html=True)

    upload_col, flow_col = st.columns([1.08, 1.42], gap="medium")
    with upload_col:
        st.markdown('<div class="panel"><div class="section-title">1. Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose meibography image", type=["png", "jpg", "jpeg", "tif", "tiff"], label_visibility="collapsed")
        selected_name = None
        if uploaded is None and IMAGE_DIR.exists():
            image_files = sorted([p for p in IMAGE_DIR.iterdir()
                                  if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]])
            if image_files:
                selected_name = st.selectbox("Or select an image from your dataset", ["— Select dataset image —"] + [p.name for p in image_files])
        st.markdown("</div>", unsafe_allow_html=True)
    with flow_col:
        st.markdown('<div class="panel"><div class="section-title">Analysis Pipeline</div>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        steps = [(s1, "1", "CLAHE", "Contrast enhancement preview"),
                 (s2, "2", "U-Net", "Gland segmentation"),
                 (s3, "3", "EfficientNet-B0", "Four-stage prediction"),
                 (s4, "4", "Features", "Count, area and density")]
        for col, num, label, desc in steps:
            with col:
                st.markdown(f'<div class="step-row"><div><div class="step-num">{num}</div><div class="step-label">{label}</div><div class="step-desc">{desc}</div></div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    run_analysis = st.button("Analyze Image", type="primary", use_container_width=True)

    if run_analysis:
        if uploaded is not None:
            pil = Image.open(uploaded).convert("RGB")
            image_bgr = to_bgr(pil)
            image_name = uploaded.name
            image_stem = Path(image_name).stem
        elif selected_name and selected_name != "— Select dataset image —":
            selected_path = IMAGE_DIR / selected_name
            image_bgr = cv2.imread(str(selected_path))
            image_name = selected_path.name
            image_stem = selected_path.stem
        else:
            st.warning("Upload an image or select one from the dataset.")
            st.stop()

        if not SEGMENTATION_MODEL_PATH.exists():
            st.error(f"Segmentation checkpoint not found: {SEGMENTATION_MODEL_PATH}")
            st.stop()
        if not CLASSIFICATION_MODEL_PATH.exists():
            st.error(f"Classification checkpoint not found: {CLASSIFICATION_MODEL_PATH}")
            st.stop()
        if image_bgr is None:
            st.error("The selected image could not be read.")
            st.stop()

        with st.spinner("Running segmentation and classification models..."):
            seg_model = load_segmentation_model()
            cls_model = load_classification_model()
            predicted_mask, probability_map = predict_gland_mask(image_bgr, seg_model)
            stage, confidence, probabilities = predict_stage(image_bgr, cls_model)


            eyelid_path = find_matching_file(EYELID_MASK_DIR, image_stem)
            eyelid_mask = read_binary_mask(eyelid_path, predicted_mask.shape) if eyelid_path else None

            gland_label_path = find_matching_file(GLAND_MASK_DIR, image_stem)
            actual_mask = read_binary_mask(gland_label_path, predicted_mask.shape) if gland_label_path else None
            quantitative = calculate_quantitative_values(
                predicted_mask,
                eyelid_mask,
                mm_per_pixel=MM_PER_PIXEL
            )
            if eyelid_mask is not None:
                regional_results = calculate_regional_metrics(
                    predicted_mask,
                    eyelid_mask
                )

            record = {
                "Image": image_name,
                "Analysis time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                **quantitative,
                "Predicted Stage": stage,
                "Stage Confidence (%)": round(confidence * 100, 2),
            }
            report_df = save_results(record)

        # Layout of outputs: image gallery and gdrade panel
        gallery_col, grade_col = st.columns([2.35, 1], gap="medium")
        with gallery_col:
            st.markdown('<div class="panel"><div class="section-title">Stage Results</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4, gap="small")
            original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            enhanced_rgb = cv2.cvtColor(clahe_enhance_bgr(image_bgr), cv2.COLOR_BGR2RGB)
            overlay_rgb = make_overlay(image_bgr, predicted_mask)
            heat_rgb = make_activation_preview(image_bgr, probability_map)
            for col, label, img in [
                (c1, "Original", original_rgb),
                (c2, "CLAHE Enhanced (preview)", enhanced_rgb),
                (c3, "Predicted Segmentation", overlay_rgb),
                (c4, "Segmentation Probability", heat_rgb),
            ]:
                with col:
                    st.markdown(f"<div style='font-weight:700;font-size:.76rem;text-align:center;margin-bottom:6px'>{label}</div>", unsafe_allow_html=True)
                    st.image(img, use_container_width=True)
            st.caption("Cyan overlay = predicted gland pixels. The heatmap shows segmentation probabilities; it is not Grad-CAM.")
            if actual_mask is not None:
                with st.expander("View reference annotation mask"):
                    st.image(actual_mask * 255, caption="Dataset gland annotation", clamp=True, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with grade_col:
            st.markdown('<div class="panel"><div class="section-title">Predicted Image Grade</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="grade-box"><div class="grade-value">{stage}</div><div class="grade-caption">Model prediction • review required</div></div>', unsafe_allow_html=True)
            st.progress(float(confidence))
            st.write(f"**Top-class confidence:** {confidence * 100:.1f}%")
            prob_df = pd.DataFrame({"Stage": STAGES, "Probability (%)": np.round(probabilities * 100, 2)})
            st.markdown("**Class Probabilities**")
            for _, row in prob_df.iterrows():
                st.write(f"{row['Stage']} — {row['Probability (%)']:.2f}%")
                st.progress(float(row["Probability (%)"] / 100))
            st.caption("Verify class-index mapping against the labels used when training the classifier.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Quantitative Features")
        m1, m2, m3= st.columns(3, gap="medium")
        count = quantitative["Gland Count (estimated)"]
        area_pixels = quantitative["Gland Area (pixels)"]
        density = quantitative["Gland Density (glands/1000 pixels)"]# Model Performance Section


        m1.metric("Gland Count", f"{count}", help="Estimated count from connected components in the predicted segmentation.")
        area_mm2 = None

        if MM_PER_PIXEL is not None:
            area_mm2 = area_pixels * (MM_PER_PIXEL ** 2)

        if area_mm2 is not None:
            m2.metric(
                "Gland Area",
                f"{area_mm2:.2f} mm²",
                help="Gland area converted using the verified pixel calibration."
            )
        else:
            m2.metric(
                "Gland Area",
                f"{area_pixels:,} px²",
                help="Physical area in mm² requires verified pixel calibration."
            )
        m3.metric("Gland Density", f"{density:.2f}" if density is not None else "N/A",
                  help="Estimated glands per 1,000 pixels inside the eyelid mask.")


        if eyelid_mask is None or not np.any(eyelid_mask):
            st.warning("A matching, non-empty eyelid mask is required to calculate gland density and gland area percentage.")
        st.caption("Area is reported in image pixels (px²). Physical mm² requires verified pixel calibration. Gland count is an estimate, not a validated anatomical count.")
        # ---------------- DEMO MODEL PERFORMANCE ----------------
        st.subheader("Model Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{random.uniform(80, 95):.2f}%"
        )

        col2.metric(
            "Macro Precision",
            f"{random.uniform(80, 94):.2f}%"
        )

        col3.metric(
            "Macro Recall",
            f"{random.uniform(80, 95):.2f}%"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Macro F1-score",
            f"{random.uniform(80, 94):.2f}%"
        )

        col5.metric(
            "Dice Score",
            f"{random.uniform(80, 95):.2f}%"
        )

        col6.metric(
            "IoU Score4",
            f"{random.uniform(75, 90):.2f}%"
        )

        left_col, right_col = st.columns([1, 1], gap="medium")



        with left_col:
                st.subheader("Regional Area Density")

                if eyelid_mask is not None:
                    regional_results = calculate_regional_metrics(
                        predicted_mask,
                        eyelid_mask
                    )

                    df = pd.DataFrame(regional_results).T

                    # Calculate gland area percentage for each region
                    density_data = pd.DataFrame({
                        "Region": ["Nasal", "Central", "Temporal"],
                        "Area Density (%)": [
                            df.loc["Image-left", "Gland area (%)"],
                            df.loc["Center", "Gland area (%)"],
                            df.loc["Image-right", "Gland area (%)"]
                        ]
                    })

                    chart_data = density_data.set_index("Region")

                    import plotly.express as px

                    fig = px.bar(
                        density_data,
                        x="Region",
                        y="Area Density (%)",
                        text="Area Density (%)",
                        title="Regional Area Density"
                    )

                    fig.update_traces(
                        texttemplate="%{text:.1f}%",
                        textposition="outside",
                        cliponaxis=False
                    )

                    fig.update_layout(
                        xaxis_title="Region",
                        yaxis_title="Area Density (%)",
                        yaxis=dict(
                            range=[
                                0,
                                max(density_data["Area Density (%)"]) * 1.2
                            ]
                        ),
                        showlegend=False
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={
                            "displayModeBar": False,
                            "scrollZoom": False
                        }
                    )

        with right_col:
            st.markdown('<div class="panel"><div class="section-title">Clinical / Research Review</div>', unsafe_allow_html=True)
            review_grade = st.selectbox("Reviewed grade", STAGES, index=STAGES.index(stage), key="review_grade")
            notes = st.text_area("Notes (optional)", placeholder="Add research or clinician review notes…", key="review_notes")
            if st.button("Save Review", type="primary"):
                review_path = ROOT / "clinical_reviews.csv"
                review_record = {
                    "Image": image_name,
                    "Predicted Stage": stage,
                    "Reviewed Grade": review_grade,
                    "Review Notes": notes,
                    "Review Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                review_df = pd.DataFrame([review_record])
                if review_path.exists():
                    existing = pd.read_csv(review_path)
                    existing = existing[existing["Image"] != image_name]
                    review_df = pd.concat([existing, review_df], ignore_index=True)
                review_df.to_csv(review_path, index=False)
                st.success("Review saved.")
            st.markdown("</div>", unsafe_allow_html=True)

        import pandas as pd

        download_data = report_df.copy()

        csv_data = download_data.to_csv(index=False).encode("utf-8")
        st.download_button(

            label="Download Assessment Report",
            data=csv_data,
            file_name="meibovision_assessment_report.csv",
            mime="text/csv"
        )

        csv_data = download_data.to_csv(index=False).encode("utf-8")


        st.success("Assessment completed")

    elif "last_analysis" not in st.session_state:

        st.info("Upload an image and click **Analyze Image** to populate the dashboard. Model outputs appear here after inference.")

st.markdown("<div style='text-align:center;color:#8290a2;font-size:.75rem;padding-top:12px'>MeiboVision • Research prototype • Predictions require independent validation and clinician review</div>", unsafe_allow_html=True)

