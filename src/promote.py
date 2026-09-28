import mlflow
from mlflow import MlflowClient
from src.config import (
    REGISTERED_MODEL_NAME,
    CANDIDATE_ALIAS, 
    CHAMPION_ALIAS,
    TRACKING_URI
)

def promote_model():

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    client=MlflowClient()

    # find the candidate model version
    candidate=client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CANDIDATE_ALIAS,
    )

    # promote candidate to champion
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=candidate.version
    )

    print(f"Promted {REGISTERED_MODEL_NAME} with model version of {candidate.version} from {CANDIDATE_ALIAS} to {CHAMPION_ALIAS}.")

if __name__=="__main__":
    promote_model()