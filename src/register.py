import json
import mlflow
import mlflow.pyfunc
from mlflow.client import MlflowClient
from ultralytics import YOLO
from src.config import (
    REGISTERED_MODEL_NAME,
    TRACKING_URI,
    EXPERIMENT_NAME,
    CANDIDATE_ALIAS,OUTPUT_DIR,
    TRAIN_NAME,
    TEST_METRICS,
    RUN_NAME
)

class YOLOModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model=YOLO(context.artifacts["model"])

    def predict(self, context, model_input: list[str], 
                params=None)-> list[dict]:
        results=self.model.predict(
            source=model_input,
            verbose=False,
        )
        predictions=[]

        for result in results:
            detections=[]
            if result.boxes is not None:
                boxes=result.boxes.xyxy.cpu().tolist()
                confidences=result.boxes.conf.cpu().tolist()
                classes =result.boxes.cls.cpu().tolist()

                for box, confidence, class_id in zip(
                    boxes, confidences, classes
                ):
                    class_id=int(class_id)

                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": result.names[class_id],
                            "confidence": float(confidence),
                            "bbox": box
                        }
                    )
            predictions.append({
            "detections": detections
        })
    
        return predictions

def register_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    model_path=(
        OUTPUT_DIR
        /TRAIN_NAME
        /"weights"
        /"best.pt"
    )

    with open(TEST_METRICS, "r") as f:
        test_metrics=json.load(f)

    with mlflow.start_run(
        run_name=RUN_NAME
    ):
        # log final test metrics
        mlflow.log_metrics(test_metrics)

        # package best.pt as an MLflow model
        model_info=mlflow.pyfunc.log_model(
            name="model",
            python_model=YOLOModel(),
            artifacts={
                "model": str(model_path)
            },
        )

        # Register a new model version
        model_version=mlflow.register_model(
            model_uri=model_info.model_uri,
            name=REGISTERED_MODEL_NAME
        )

        # Give new version the candidate alias
        client=MlflowClient()
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias=CANDIDATE_ALIAS,
            version=model_version.version
        )

        print(f"Registered {REGISTERED_MODEL_NAME}")
        print(f"version {model_version.version}: alias {CANDIDATE_ALIAS}")
if __name__=="__main__":
    register_model()