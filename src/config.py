from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parent.parent

DATA_YAML=PROJECT_ROOT/"data.yaml"

MODELS_DIR=PROJECT_ROOT/"models"
OUTPUT_DIR=PROJECT_ROOT/"outputs"/"yolo"


# Model Train
PRETRAINED_MODEL="yolo11s.pt"
SEED=42
EPOCHS = 50
IMAGE_SIZE=640
BATCH_SIZE=8
DEVICE=0
PATIENCE=3
WORKERS=8
TRAIN_NAME="Rail_Defect_Detect"
TEST_NAME="Rail_Defect_test"

# frontend data
FRONTEND_DATA_DIR=(
    PROJECT_ROOT
    /"outputs"
    /"frontend_data"
)
TEST_METRICS=FRONTEND_DATA_DIR/"test_metrics.json"



# MLFLOW
TRACKING_URI="http://127.0.0.1:5000"
EXPERIMENT_NAME="railway_defect_detection"
CANDIDATE_ALIAS="candidate"
CHAMPION_ALIAS="champion"
RUN_NAME="register_yolo"
REGISTERED_MODEL_NAME="railway_defect_yolo"
ARTIFACT_URI=(PROJECT_ROOT/"mlartifacts").as_uri()

# Model Path
MODEL_PATH=(
    OUTPUT_DIR
    / TRAIN_NAME
    / "weights"
    / "best.pt"
)