from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from torchvision.models import resnet18
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


# ============================================================
# URBANWATCH — INTERACTIVE DEMO
# ============================================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

YOLO_PATH = (
    RESULTS
    / "yolo_training"
    / "spacenet_building_detector-4"
    / "weights"
    / "best.pt"
)

LOVEDA_PATH = (
    RESULTS
    / "loveda_segmentation_resnet18"
    / "best_model.pt"
)

LEVIR_PREDICTIONS = (
    RESULTS
    / "change_detection_inference"
    / "predictions"
)

LEVIR_VISUALIZATIONS = (
    RESULTS
    / "change_detection_inference"
    / "visualizations"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="UrbanWatch",
    page_icon="🛰️",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛰️ UrbanWatch</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered satellite imagery analysis — '
    'building detection, land-cover segmentation, '
    'and change detection.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("UrbanWatch")

st.sidebar.write(
    f"Device: `{DEVICE}`"
)

if torch.cuda.is_available():

    st.sidebar.success(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

else:

    st.sidebar.warning(
        "CUDA unavailable — using CPU"
    )

st.sidebar.divider()

page = st.sidebar.radio(
    "Select analysis",
    [
        "🏠 Dashboard",
        "🏢 Building Detection",
        "🌍 Land-Cover Segmentation",
        "🔄 Change Detection",
        "📊 Project Results"
    ]
)


# ============================================================
# YOLO MODEL
# ============================================================

@st.cache_resource
def load_yolo():

    if not YOLO_PATH.exists():

        raise FileNotFoundError(
            f"YOLO model not found:\n{YOLO_PATH}"
        )

    return YOLO(
        str(YOLO_PATH)
    )


# ============================================================
# DEEPLABV3 + RESNET18
#
# torchvision does not provide a ready-made
# deeplabv3_resnet18() function.
#
# We construct the same type of architecture manually.
# ============================================================

class DeepLabV3ResNet18(nn.Module):

    def __init__(
        self,
        num_classes=8
    ):

        super().__init__()

        backbone = resnet18(
            weights=None
        )

        self.backbone = IntermediateLayerGetter(
            backbone,
            return_layers={
                "layer4": "out"
            }
        )

        self.classifier = DeepLabHead(
            512,
            num_classes
        )

    def forward(
        self,
        x
    ):

        input_shape = x.shape[-2:]

        features = self.backbone(
            x
        )

        x = self.classifier(
            features["out"]
        )

        x = F.interpolate(
            x,
            size=input_shape,
            mode="bilinear",
            align_corners=False
        )

        return {
            "out": x
        }


# ============================================================
# LOAD LOVEDA
# ============================================================

@st.cache_resource
def load_loveda():

    if not LOVEDA_PATH.exists():

        raise FileNotFoundError(
            f"LoveDA model not found:\n{LOVEDA_PATH}"
        )

    model = DeepLabV3ResNet18(
        num_classes=8
    )

    checkpoint = torch.load(
        LOVEDA_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:

            state_dict = (
                checkpoint["state_dict"]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # Remove possible DataParallel prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# LOVEDA PREPROCESSING
# ============================================================

def preprocess_loveda(
    image
):

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        (512, 512),
        Image.Resampling.BILINEAR
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    array /= 255.0

    # Same ImageNet normalization normally used
    # with ResNet-based segmentation models.
    mean = np.array(
        [
            0.485,
            0.456,
            0.406
        ],
        dtype=np.float32
    )

    std = np.array(
        [
            0.229,
            0.224,
            0.225
        ],
        dtype=np.float32
    )

    array = (
        array - mean
    ) / std

    tensor = torch.from_numpy(
        array.transpose(
            2,
            0,
            1
        )
    ).float()

    return tensor.unsqueeze(
        0
    )


# ============================================================
# LOVEDA INFERENCE
# ============================================================

@torch.no_grad()
def run_loveda(
    image
):

    model = load_loveda()

    tensor = preprocess_loveda(
        image
    ).to(
        DEVICE
    )

    output = model(
        tensor
    )

    logits = output["out"]

    prediction = torch.argmax(
        logits,
        dim=1
    )[0]

    return prediction.cpu().numpy()


# ============================================================
# SEGMENTATION OVERLAY
# ============================================================

def segmentation_overlay(
    image,
    mask
):

    original = image.convert(
        "RGB"
    )

    original = original.resize(
        (512, 512),
        Image.Resampling.BILINEAR
    )

    image_np = np.asarray(
        original,
        dtype=np.float32
    )

    mask = np.asarray(
        mask
    )

    palette = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 0]
        ],
        dtype=np.uint8
    )

    mask_rgb = palette[
        np.clip(
            mask,
            0,
            7
        )
    ]

    overlay = (
        0.65 * image_np
        +
        0.35 * mask_rgb
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        overlay
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header(
        "UrbanWatch Dashboard"
    )

    st.write(
        "Unified remote-sensing analysis using three "
        "trained deep-learning models."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "SpaceNet 2",
            "82.12%",
            "mAP@50"
        )

        st.caption(
            "YOLO building detection"
        )

    with col2:

        st.metric(
            "LoveDA",
            "54.02%",
            "mIoU"
        )

        st.caption(
            "8-class land-cover segmentation"
        )

    with col3:

        st.metric(
            "LEVIR-CD+",
            "56.10%",
            "IoU"
        )

        st.caption(
            "Satellite change detection"
        )

    st.divider()

    st.subheader(
        "Capabilities"
    )

    st.markdown(
        """
        **🏢 Building Detection**

        Detect buildings in satellite imagery using the
        trained SpaceNet 2 YOLO detector.

        **🌍 Land-Cover Segmentation**

        Segment satellite imagery into eight LoveDA
        land-cover classes.

        **🔄 Change Detection**

        Browse the full LEVIR-CD+ change-detection
        predictions generated by the trained model.
        """
    )


# ============================================================
# BUILDING DETECTION
# ============================================================

elif page == "🏢 Building Detection":

    st.header(
        "🏢 Building Detection"
    )

    st.write(
        "Upload satellite imagery and detect buildings."
    )

    uploaded = st.file_uploader(
        "Upload satellite image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff"
        ],
        key="building_upload"
    )

    if uploaded:

        image = Image.open(
            uploaded
        ).convert(
            "RGB"
        )

        st.subheader(
            "Input Image"
        )

        st.image(
            image,
            use_container_width=True
        )

        if st.button(
            "🔍 Detect Buildings",
            type="primary"
        ):

            with st.spinner(
                "Running YOLO..."
            ):

                model = load_yolo()

                results = model.predict(
                    source=image,
                    imgsz=640,
                    conf=0.25,
                    device=(
                        0
                        if torch.cuda.is_available()
                        else "cpu"
                    ),
                    verbose=False
                )

            result = results[0]

            plotted = result.plot()

            plotted = Image.fromarray(
                plotted[
                    :,
                    :,
                    ::-1
                ]
            )

            st.subheader(
                "Detection Result"
            )

            st.image(
                plotted,
                use_container_width=True
            )

            number = 0

            if result.boxes is not None:

                number = len(
                    result.boxes
                )

            st.metric(
                "Buildings detected",
                number
            )


# ============================================================
# LOVEDA SEGMENTATION
# ============================================================

elif page == "🌍 Land-Cover Segmentation":

    st.header(
        "🌍 Land-Cover Segmentation"
    )

    st.write(
        "Run the trained DeepLabV3 + ResNet18 model."
    )

    uploaded = st.file_uploader(
        "Upload satellite image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff"
        ],
        key="loveda_upload"
    )

    if uploaded:

        image = Image.open(
            uploaded
        ).convert(
            "RGB"
        )

        if st.button(
            "🌍 Run Segmentation",
            type="primary"
        ):

            with st.spinner(
                "Running LoveDA segmentation..."
            ):

                mask = run_loveda(
                    image
                )

                overlay = (
                    segmentation_overlay(
                        image,
                        mask
                    )
                )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "Input"
                )

                st.image(
                    image,
                    use_container_width=True
                )

            with col2:

                st.subheader(
                    "Segmentation"
                )

                st.image(
                    overlay,
                    use_container_width=True
                )

            classes = sorted(
                np.unique(
                    mask
                ).tolist()
            )

            st.success(
                "Segmentation complete."
            )

            st.write(
                "Detected class IDs:",
                classes
            )


# ============================================================
# CHANGE DETECTION
# ============================================================

elif page == "🔄 Change Detection":

    st.header(
        "🔄 LEVIR-CD+ Change Detection"
    )

    st.write(
        "Browse the 348 change-detection predictions "
        "generated by the trained LEVIR-CD+ model."
    )

    if not LEVIR_PREDICTIONS.exists():

        st.error(
            "LEVIR predictions were not found."
        )

    else:

        predictions = sorted(
            LEVIR_PREDICTIONS.glob(
                "*.png"
            )
        )

        if not predictions:

            st.warning(
                "No predictions found."
            )

        else:

            names = [
                p.name
                for p in predictions
            ]

            selected = st.selectbox(
                "Select prediction",
                names
            )

            selected_path = (
                LEVIR_PREDICTIONS
                / selected
            )

            st.subheader(
                "Change Map"
            )

            st.image(
                Image.open(
                    selected_path
                ),
                use_container_width=True
            )

            st.metric(
                "Available predictions",
                len(predictions)
            )


# ============================================================
# PROJECT RESULTS
# ============================================================

elif page == "📊 Project Results":

    st.header(
        "📊 UrbanWatch Final Results"
    )

    st.subheader(
        "SpaceNet 2 — Building Detection"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Precision",
        "82.74%"
    )

    col2.metric(
        "Recall",
        "75.16%"
    )

    col3.metric(
        "mAP@50",
        "82.12%"
    )

    col4.metric(
        "mAP@50-95",
        "50.04%"
    )

    st.divider()

    st.subheader(
        "LoveDA — Semantic Segmentation"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "mIoU",
        "54.02%"
    )

    col2.metric(
        "Dice/F1",
        "68.47%"
    )

    col3.metric(
        "Pixel Accuracy",
        "68.80%"
    )

    st.divider()

    st.subheader(
        "LEVIR-CD+ — Change Detection"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Precision",
        "70.70%"
    )

    col2.metric(
        "Recall",
        "73.10%"
    )

    col3.metric(
        "Dice/F1",
        "71.88%"
    )

    col4.metric(
        "IoU",
        "56.10%"
    )

    col5.metric(
        "Accuracy",
        "97.64%"
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "UrbanWatch"
)

st.sidebar.caption(
    "Remote Sensing AI Pipeline"
)