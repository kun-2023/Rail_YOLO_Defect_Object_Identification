from unittest.mock import MagicMock, patch
import numpy as np
from src.inference import RailDefectDetector
from src.config import IMAGE_SIZE

@patch("src.inference.YOLO")
def test_detector_initializes(mock_yolo):
    mock_model=MagicMock()
    mock_yolo.return_value=mock_model

    detector=RailDefectDetector(
        model_path="fake_model.pt",
        device="cpu"
    )

    mock_yolo.assert_called_once_with("fake_model.pt")

    assert detector.model==mock_model
    assert detector.device=="cpu"

@patch("src.inference.YOLO")
def test_predict_calls_yolo(mock_yolo):
    mock_model=MagicMock()
    mock_yolo.return_value=mock_model

    fake_results=[MagicMock()]
    mock_model.predict.return_value=fake_results

    detector=RailDefectDetector(
        model_path="fake_model.pt",
        device="cpu"
    )

    fake_image=np.zeros(
        (640, 640, 3),
        dtype=np.uint8
    )

    results=detector.predict(
        fake_image,
        confidence=0.25
    )

    mock_model.predict.assert_called_once()

    assert results==fake_results

@patch("src.inference.YOLO")    
def test_predict_passes_correct_arguments(mock_yolo):
    mock_model=MagicMock()
    mock_yolo.return_value=mock_model

    mock_model.predict.return_value=[]

    detector=RailDefectDetector(
        model_path="fake_model.pt",
        device="cpu"
    )

    fake_image=np.zeros(
        (640, 640, 3),
        dtype=np.uint8
    )

    detector.predict(
        fake_image,
        confidence=0.4
    )

    mock_model.predict.assert_called_once_with(
        source=fake_image,
        imgsz=IMAGE_SIZE,
        conf=0.4,
        device="cpu",
        verbose=False
    )
