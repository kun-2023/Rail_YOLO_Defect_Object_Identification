import requests
import streamlit as st
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw
import os

PROJECT_ROOT=Path(__file__).resolve().parents[1]

TRY_IT_OUT_FOLDER=(
    PROJECT_ROOT/"outputs/frontend_data/try_it_out_image"
)

API_URL=os.getenv("API_URL","http://api:8000/predict")

def show_prediction():
    st.title("Rail Defect Live Prediction")
    st.subheader("Detection of Corrugation, Cracks, Flaking, Shelling, Spalling, Squats, Wheel-Burn")
    st.subheader("Try it out the photo below")

    if "prediction_source" not in st.session_state:
            st.session_state.prediction_source=None

    def use_uploaded_image():
        st.session_state.prediction_source="upload"

    
    # Try It Out session

    sample_images=sorted([
        image for image in TRY_IT_OUT_FOLDER.iterdir()
        if image.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])
    sample_image=None
    try_it_out_clicked=False
    if sample_images:
        sample_image=sample_images[0]
        st.image(str(sample_image), width=300)

        try_it_out_clicked=st.button("Try it out")
        if try_it_out_clicked:
            st.session_state.prediction_source="sample"
    else:
        st.warning("No try it out image found")


    # Upload Photo
    st.subheader("Upload Your Own Image")
    st.write("Upload a rail image of .jpeg, .jpg or .png")
    uploaded_file=st.file_uploader(
        "Upload an image", 
        type=["jpg", "jpeg", "png"],
        key="prediction_upload",
        on_change=use_uploaded_image
    )

    image_bytes=None
    image_name=None
    image_type=None

    if (
        st.session_state.prediction_source=="upload"
        and uploaded_file is not None
    ):
        image_bytes=uploaded_file.getvalue()
        image_name=uploaded_file.name
        image_type=uploaded_file.type

    elif st.session_state.prediction_source=="sample":
        if sample_image is None:
            st.warning("No try it out image found")
            return
        image_bytes=sample_image.read_bytes()
        image_name=sample_image.name

        if sample_image.suffix.lower()==".png":
            image_type="image/png"
        else:
            image_type="image/jpeg"
    else:
        return
  
    image=Image.open(BytesIO(image_bytes)).convert("RGB")
    if st.session_state.prediction_source=="sample":
        confidence=0.25
        run_prediction=try_it_out_clicked
    else:
        st.subheader("Original Image")
        st.image(
            image,width=650
        )

        confidence=st.slider(
            "Confidence Slider",
            min_value=0.1,
            max_value=0.9,
            value=0.25,
            step=0.05
        )

        run_prediction=st.button("Run Prediction")
    
    if run_prediction:
        files={
            "file":(
                image_name,
                image_bytes,
                image_type
            )
        }
        
        params={
            "confidence": confidence
        }

        try:
            response=requests.post(API_URL, 
                               files=files, 
                               params=params)

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to API: {e}")
            return

        if response.status_code==200:
            prediction=response.json()
            predicted_image=image.copy()
            draw=ImageDraw.Draw(predicted_image)

            for detection in prediction["detections"]:
                x1, y1, x2, y2=detection["bbox"]

                class_name=detection["class_name"]
                score=detection["confidence"]

                # draw bounding box
                draw.rectangle(
                    [x1,y1,x2,y2],
                    outline="red",
                    width=3
                )

                label=(
                    f"{class_name} "
                    f"{score:.1%}"
                )

                draw.text(
                    (x1,y1),
                    label,
                    fill="red"
                )
            st.subheader("prediction")

            st.image(
                predicted_image,
                width=750
            )

        else:
            st.error(
                f"Prediction failed: "
                f"{response.status_code} - {response.text}"
            )
            