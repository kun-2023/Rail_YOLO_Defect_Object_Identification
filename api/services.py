from api.settings import MODEL_PATH, API_DEVICE
from src.inference import RailDefectDetector

detector=RailDefectDetector(MODEL_PATH, API_DEVICE)