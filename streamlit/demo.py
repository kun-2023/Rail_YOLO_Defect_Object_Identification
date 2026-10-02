from pathlib import Path
import streamlit as st


PROJECT_ROOT=Path(__file__).resolve().parents[1]
DEMO_DATA_PATH=(
    PROJECT_ROOT/"outputs/frontend_data/images"
)

def show_demo():
    st.title("Rail Defect Detection Demo")
    st.subheader("Detection of Corrugation, Cracks, Flaking, Shelling, Spalling, Squats, Wheel-Burn")
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
                width=320
            )

        