import streamlit as st
from overview import show_overview
from demo import show_demo
from prediction import show_prediction

st.set_page_config(
    page_title="Railway Yolo Object Identification",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
    width: 80%;
    max-width: 80%;
    margin:0 auto;
    }
    </style>
""", unsafe_allow_html=True
)

overview_page=st.Page(show_overview, title="Overview", default=True)
demo_page=st.Page(show_demo, title="Demo")
prediction_page=st.Page(show_prediction, title="Prediction")

pg=st.navigation([overview_page, demo_page, prediction_page])

pg.run()
