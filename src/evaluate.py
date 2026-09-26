from ultralytics import YOLO

from src.config import (
    DATA_YAML,
    OUTPUT_DIR,
    IMAGE_SIZE,
    DEVICE, TEST_NAME, TRAIN_NAME,
    FRONTEND_DATA_DIR
)

import json

def evaluate_model(model_path):
    
    model=YOLO(str(model_path))

    metrics=model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        project=str(OUTPUT_DIR),
        name=TEST_NAME
    )

    return metrics

if __name__=="__main__":
    model_path=(
        OUTPUT_DIR
        / TRAIN_NAME
        / "weights"
        / "best.pt"
    )

    metrics=evaluate_model(model_path)

    test_metrics={
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map)
    }

    metrics_path=(
        FRONTEND_DATA_DIR
        /"test_metrics.json"
    )

    with open(metrics_path, "w") as f:
        json.dump(
            test_metrics,
            f,
            indent=4
        )

    print(test_metrics)