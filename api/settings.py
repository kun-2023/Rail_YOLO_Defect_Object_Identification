import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT=Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT/".env")
MODEL_PATH=PROJECT_ROOT/os.getenv(
    "MODEL_PATH",
    "outputs/yolo/Rail_Defect_Detect_7class/weights/best.pt"
)
DEFAULT_CONFIDENCE=float(
    os.getenv("DEFAULT_CONFIDENCE","0.25")
)
API_DEVICE=os.getenv("API_DEVICE", "cpu")

API_URL="http://api:8000/predict"