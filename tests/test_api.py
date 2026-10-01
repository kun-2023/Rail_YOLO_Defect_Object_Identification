from unittest.mock import MagicMock, patch

import numpy as np
import cv2
from fastapi.testclient import TestClient

from api.main import app

client=TestClient(app)

def create_fake_image():
    image=np.zeros(
        (640, 640, 3), dtype=np.uint8
    )

    success, encoded_image=cv2.imencode(
        ".jpg", image
    )

    assert success

    return encoded_image.tobytes()

def test_app_runs():
    response=client.get("/docs")
    assert response.status_code==200

@patch("api.services.detector.predict")    
def test_predict_endpoint(mock_predict):
    fake_result=MagicMock()
    mock_predict.return_value=[fake_result]
    image_bytes=create_fake_image()
    response=client.post("/predict",
                         files={
                             "file": (
                                 "test.jpg",
                                 image_bytes,
                                 "image/jpeg"
                             )
                         })
    assert response.status_code==200
    assert response.headers["content-type"]=="application/json"
    assert response.json()=={"detections":[]}
    assert len(response.content)>0
    mock_predict.assert_called_once()
    

def test_predict_without_file():
    response=client.post("/predict")
    assert response.status_code==422