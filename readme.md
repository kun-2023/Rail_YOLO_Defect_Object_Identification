#### Install library with requirements
1. install PyTOrch first
2. Install requirements.txt
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```
```bash
pip install -r requirements.txt
```

#### Manually run ml training pipeline
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

#### Train model with pipeline and Manually register and promte model
```bash
dvc repro
python -m src.register
python -m src.promote
dvc push
git add dvc.yaml dvc.lock
git commit -m "update trained model"
```

#### AWS authentication
```bash
aws sts get-caller-identity
dvc remote add -d storage s3://rail-yolo-detection
dvc remote list
dvc push
aws s3 ls s3://rail-yolo-detection
git add .dvc/config
git commit -m "configure DVC S3 remote
```

#### RUN API Server
```bash
python -m uvicorn api.main:app --reload
```

#### Run dockerfile
```bash
docker compose build
docker compose up
# docker compose up --build
docker compose down
```

#### Run apps locally with docker; open Docker desktop
```bash
git clone ...
cd rail_yolo_object_identification
dvc pull
docker compose up --build
```
<p>Then open http://localhost:8501</p>

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

#### Run tests
```bash
pytest -v
```