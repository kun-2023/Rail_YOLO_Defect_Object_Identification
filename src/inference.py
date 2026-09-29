from ultralytics import YOLO
from src.config import PROJECT_ROOT,IMAGE_SIZE, DEVICE, OUTPUT_DIR, TRAIN_NAME

class RailDefectDetector:
    def __init__(self, model_path, device):
        self.model=YOLO(str(model_path))
        self.device=device

    def predict(self, image, confidence=0.25):
        results=self.model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=confidence,
            device=self.device,
            verbose=False
        )

        return results

if __name__=="__main__":
    model_path=(
        OUTPUT_DIR/TRAIN_NAME/"weights"/"best.pt"
    )

    image_path=(PROJECT_ROOT/"data"/"test"/"images"/"20231018_112728_mp4-0004_jpg.rf.741e9c5216c788ffb0fb417b518e6f5e.jpg")
    model=RailDefectDetector(model_path)
    results=model.predict(image_path, confidence=0.25)
    print(results[0].boxes)