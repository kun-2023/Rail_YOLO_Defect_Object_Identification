from pydantic import BaseModel

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]

class PredictionResponse(BaseModel):
    detections: list[Detection]
