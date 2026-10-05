## Railway Defects Object Identification with yolo26s

### Project in a nutshell
* App in a glance: An **end-to-end railway defect object identification ML app powered by yolo26s**. The app receives an image of railway and be able to identify up to seven types of railway defects including **Corrugation**, **Cracks**, **Flaking**, **Shelling**, **Spalling**, **Squats**, **Wheel-Burn**. Upon detecting defects on the image of railways, it will draw a small box around the defect area with names and probabilty on that image.

* Tech stacks: **python**, **yolo26s**, **Streamlit**, **FastAPI**, **Docker**, **DVC**, **MLflow**, **AWS ECR/ECS** and **GitHub action**; 

* ML Pipelines: The first is **mlflow-dvc pipeline**, which allows the retraining of the model by committing in CLI a line: `dvc repro`. Second, **CI/CD pipeline**, which allows integrating new codes and redeployment of the app by committing `git push` in bash.

* Future Improvements. The accuracies are low. It could be improved by more datasets.

### Setup and Usage
#### Clone repo
```bash
git clone https://github.com/kun-2023/Rail_YOLO_Defect_Object_Identification.git

cd rail_yolo_object_identification
```

#### Create and activate a virtual enviroment first
```bash
python -m venv .rail_venv
source .rail_venv/Scripts/activate
```
#### Install CUDA-enabled version of PyTorch and dependencies
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```
```bash
pip install -r requirements.txt
```

#### Run apps locally with docker while keep Docker desktop open
```bash
dvc pull

docker compose up --build

# Then open http://localhost:8501 at your browser
```
#### Run apps without dockers
```bash
git clone <repo-url>
cd rail_yolo_object_idnetification
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
dvc pull

# backend
python -m uvicorn api.main:app --reload
# front end in another terminal bash
streamlit run streamlit/app.py
# then open http://localhost:8501
```

#### Manually run ML training pipeline
```bash
python -m src.train
python -m src.evaluate
python -m src.frontend_data
python -m src.register
python -m src.promote
```

#### Start Mlflow Server

```bash
mlflow server \
 --backend-store-uri sqlite:///mlflow.db \
 --artifacts-destination ./mlartifacts \
 --host 127.0.0.1 \
 --port 5000 \
 --workers 1
```
The MLflow UI is available at:
```text
http://127.0.0.1:5000
```

#### Train model with pipeline and Manually register and promte model
```bash
dvc repro
# register and promote the model
python -m src.register
python -m src.promote
# push dvc artifacts to S3
dvc push
# push dvc metadata to github
git add dvc.yaml dvc.lock
git commit -m "update trained model"
git push
```

#### AWS authentication
```bash
aws sts get-caller-identity # verify AWS authentication
dvc remote add -d storage s3://rail-yolo-detection # configure DVC s3 remote
dvc remote list # verify remote s3
dvc push
aws s3 ls s3://rail-yolo-detection # verify objects in S3 bucket
git add .dvc/config
git commit -m "configure DVC S3 remote"
```

#### RUN API Server
```bash
python -m uvicorn api.main:app --reload
# api runs at http://127.0.0.1:8000
# fastAPI docs run at http://127.0.0.1:8000/docs
```

#### Run with Compose
```bash
docker compose up --build
docker compose down # stop and remove containers image
```
#### Run streamlit locally
```bash
streamlit run streamlit/app.py
# streamlit app available at http://localhost:8501
```


#### Run tests
```bash
pytest -v
```

#### CI/CD
A push to the `main` branch triggers the pipeline.
The CI/CD workflow is available at .github/workflows/ci_cd.yml

