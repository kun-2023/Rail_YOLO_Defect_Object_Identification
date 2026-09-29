import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT=Path(__file__).resolve().parent.parent
MODEL_PATH=PROJECT_ROOT/os.getenv(
    "MODEL_PATH",
    "outputs/yolo/Rail_Defect_Detect/weights/best.pt"
)

DEFAULT_CONFIDENCE=float(
    os.getenv("DEFAULT_CONFIDENCE","0.25")
)