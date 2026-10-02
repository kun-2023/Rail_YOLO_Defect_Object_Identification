from pathlib import Path
import json
import streamlit as st
import pandas as pd

PROJECT_ROOT=Path(__file__).resolve().parent.parent
TEST_METRICS_PATH=(PROJECT_ROOT/"outputs/frontend_data/test_metrics.json")

def show_overview():
    st.title("Railway Yolo Object Identification")
    st.subheader("Project Overview")
    st.write(
        """
        This project is an end-to-end Yolo based railway defects 
        object identification machine learning Application. The 
        app can identify 7 categories of defects from railway tracks 
        including Corrugation, Cracks, Flaking, Shelling, Spalling, 
        Squats, Wheel-Burn. The model was trained with a pretrained 
        model named Yolo11s on 6260 images and tested on 310 images.
        
        When an image of railway tracks submitted to the app, it will 
        return the same image with small boxes drawn upon defects with 
        the name and confidence of percentage. The app would help railway
        inspectors to accurately and efficiently identify defects. Such 
        a technology is totally transferrable for other tasks such as 
        identifying defacts from trains, aircarfts, shipping vehicles 
        or constructions."""
    )

    st.subheader("Tech Stacks")
    st.write("""
    Python, Pandas, PyTorch, Yolo26s, MLflow, DVC, Git/GitHub, FastAPI,
    Docker, Streamlit, AWS, ECR, ECS, GitHub Actions, CI/CD Pipeline,
    Pytest.
    """)
    st.subheader("Test Metrics")
    with open(TEST_METRICS_PATH, "r") as file:
        metrics=json.load(file)
    col1, col2, col3, col4=st.columns(4)
    col1.metric("precision", f"{metrics['precision']:.2%}")
    col2.metric("recall", f"{metrics['recall']:.2%}")
    col3.metric("mAP50", f"{metrics['mAP50']:.2%}")
    col4.metric("mAP50_95", f"{metrics['mAP50_95']:.2%}")
