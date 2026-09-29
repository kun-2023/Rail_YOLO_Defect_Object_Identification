from pathlib import Path
import os
import requests
from io import BytesIO
import pandas as pd
import streamlit as st
from PIL import Image
from src.inference import RailDefectDetector

API_URL="http://127.0.0.1:8000/predict"

def show_prediction():
    st.title("Rail Defect Live Prediction")
    st.write("Upload a rail image of .jpeg, .jpg or .png")
    uploaded_file=st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        return

    image=Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    confidence=st.slider(
        "Confidence Slider",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05
    )

    if st.button("Run Prediction"):
        files={
            "file":(
                uploaded_file.name, uploaded_file.getvalue(), 
                uploaded_file.type
            )
        }
        
        data={
            "confidence": confidence
        }

        response=requests.post(API_URL, files=files, data=data)
        if response.status_code==200:
            predicted_images=Image.open(
                BytesIO(response.content)
            )
            st.subheader("Prediction")
            st.image(predicted_images,
                     use_container_width=True)
    else:
        st.error(
            f"Predictition failed: {response.status_code}"
        )
            