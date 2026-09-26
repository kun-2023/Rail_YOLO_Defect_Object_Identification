from ultralytics import YOLO

from src.config import (
    DATA_YAML, PRETRAINED_MODEL, OUTPUT_DIR,
    EPOCHS, IMAGE_SIZE, BATCH_SIZE, DEVICE,
    PATIENCE, TRAIN_NAME, TRACKING_URI, EXPERIMENT_NAME
)
import mlflow
import mlflow.transformers
from mlflow import MlflowClient



def train_model():

    model=YOLO(str(PRETRAINED_MODEL))
    # mlflow experiment
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    experiment_name=EXPERIMENT_NAME
    results=model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    IMGSZ=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    patience=PATIENCE,
    project=str(OUTPUT_DIR),
    name=TRAIN_NAME
            )
    return results

if __name__=="__main__":
    train_model()