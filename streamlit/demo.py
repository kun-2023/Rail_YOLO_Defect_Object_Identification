from pathlib import Path
import pandas as pd
import streamlit as st
from src.inference import RailDefectDetector
import os


PROJECT_ROOT=Path(__file__).resolve().parents[2]
DEMO_DATA_PATH=(
    PROJECT_ROOT/"outputs/frontend_data/images"
)
MODEL_PATH=PROJECT_ROOT/os.getenv(
    "MODEL_PATH",
    "outputs/yolo/Rail_Defect_Detect/weights/best.pt"
)
API_DEVICE=os.getenv("API_DEVICE", "cpu")
detector=RailDefectDetector(MODEL_PATH, API_DEVICE)

def show_demo():
    st.title("Rail Defect Detection Demo")
    image_paths=sorted([
        image for image in DEMO_DATA_PATH.iterdir()
        if image.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    if not image_paths:
        st.warning("No demo images found!")
        return
    st.write(
        f"Showing {len(image_paths)} sample predictions"
    )

    cols=st.columns(2)
    for i, image_path in enumerate(image_paths):
        with cols[i % 2]:
            st.image(
                str(image_path),
                caption=image_path.name,
                use_container_width=True
            )