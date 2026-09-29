from io import BytesIO
from fastapi import (
    APIRouter, 
    HTTPException,
    File,
    Query,
    UploadFile)
from api.schemas import PredictionResponse
from api.services import detector
from api.settings import DEFAULT_CONFIDENCE
from PIL import Image, UnidentifiedImageError

router=APIRouter()

ALLOWED_IMAGE_TYPES={
    "image/jpeg",
    "image/png"
}

@router.get("/health")
def health():
    return {
        "status": "ok"
    }
@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile=File(...),
    confidence: float=Query(
        default=DEFAULT_CONFIDENCE,
        ge=0.0,
        le=1.0
    )
):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400,
                            detail="Only JPEG and PNG images allowed.")

    contents=await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    try:
        image=Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    results=detector.predict(
        image=image,
        confidence=confidence
    )

    detections=[]

    for result in results:
        if result.boxes is None:
            continue

        boxes=result.boxes.xyxy.cpu().tolist()
        confidences=result.boxes.conf.cpu().tolist()
        classes=result.boxes.cls.cpu().tolist()

        for box, score, class_id in zip(boxes, confidences, classes):
            class_id=int(class_id)
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": float(score),
                    "bbox": box
                }
            )
        return {
            "detections": detections
        }


