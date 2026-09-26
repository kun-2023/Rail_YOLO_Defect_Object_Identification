from ultralytics import YOLO, settings
import random
import json
import os
from src.config import (
    DATA_YAML, PRETRAINED_MODEL, OUTPUT_DIR,
    EPOCHS, IMAGE_SIZE, BATCH_SIZE, DEVICE,
    PATIENCE, TRAIN_NAME, TRACKING_URI, EXPERIMENT_NAME,
    SEED, RUN_NAME
)
import mlflow




def train_model():
    random.seed(SEED)
    
    # mlflow experiment
    mlflow.set_tracking_uri(TRACKING_URI)

    os.environ["MLFLOW_TRACKING_URI"]=TRACKING_URI
    os.environ["MLFLOW_EXPERIMENT_NAME"]=EXPERIMENT_NAME
    os.environ["MLFLOW_RUN"]=RUN_NAME

    # Enable Ultralytics MLflow integration
    settings.update({"mlflow": True})

    # Load pretrained YOLO model
    model=YOLO(str(PRETRAINED_MODEL))

    results=model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    patience=PATIENCE,
    project=str(OUTPUT_DIR),
    name=TRAIN_NAME
            )
    
    return results

if __name__=="__main__":
    train_model()