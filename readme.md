## Railway Defects Object Identification with yolo26s

### Project in a nutshell
* App in a glance: An **end-to-end railway defect object identification ML app powered by yolo26s**. The app receives an uploaded image of railway and be able to identify up to seven types of railway defects including **Corrugation**, **Cracks**, **Flaking**, **Shelling**, **Spalling**, **Squats**, **Wheel-Burn** by drawing a small box around the defect area with names and probabilty on that defects.

* Tech stacks: **python**, **yolo26s**, **Streamlit**, **FastAPI**, **Docker**, **pytest**, **DVC**, **MLflow**, **AWS ECR/ECS** and **GitHub action**; 


* ML Pipelines: The first is **dvc pipeline**, which allows the retraining of the model by committing in CLI a line: `dvc repro`. Second, **CI/CD pipeline**, which allows integrating new codes and redeployment of the app by committing `git push` in bash.

* Future Improvements. The accuracies are low. It could be improved by more datasets.

#### Model Performance
| Metrics | Scores |
|---------|--------|
| Precision | 42.29% |
| Recall | 43.25% |
| mAP50 | 37.92% |
| mAP50-95 | 13.84% |


### How to Run Railway Defect Detection App locally
#### Option 1: Run with Docker(Recommended)
```bash
# 1 clone repo
git clone https://github.com/kun-2023/Rail_YOLO_Defect_Object_Identification.git

cd rail_yolo_object_identification

# 2 create and activate virtual environment
python -m venv .rail_venv
source .rail_venv/Scripts/activate

# 3 Install Dependencies
# Install CUDA-enabled version of PyTorch and dependencies
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
# Install libraries
pip install -r requirements.txt

# 4 clone .env file
cp .env.example .env

# 5 Download Data, Models and other outputs from AWS s3
dvc pull

# 6 Build Docker Image
# Turn on your Docker Desktop before proceeding
docker compose up --build -d

# 7 Then type http://localhost:8501 at Chrome browser or any other browser of your choice

# 8 After finishing inspecting the app, remove docker containers and networks
docker compose down
```
#### Option 2: Run apps without dockers
```bash
# 1 clone repo
git clone https://github.com/kun-2023/Rail_YOLO_Defect_Object_Identification.git

cd rail_yolo_object_identification

# 2 create and activate virtual environment
python -m venv .rail_venv
source .rail_venv/Scripts/activate

# 3 Install Dependencies
# Install CUDA-enabled version of PyTorch and dependencies
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
# Install libraries
pip install -r requirements.txt

# 4 clone .env file
cp .env.example .env

# 5 Download Data, Models and other outputs from AWS s3
dvc pull

# 6 Turn on backend server
python -m uvicorn api.main:app --reload

# 7 front end in another terminal bash
streamlit run streamlit/app.py
# then click open http://localhost:8501
```

### How to retrain Models with DVC pipeline
```bash
# Assuming The first 5 steps from either option 1 or option 2 are finished

# 1 Turn on MLflow server
mlflow server \
 --backend-store-uri sqlite:///mlflow.db \
 --artifacts-destination ./mlartifacts \
 --host 127.0.0.1 \
 --port 5000 \
 --workers 1

# The MLflow UI is available at: http://127.0.0.1:5000

# 2 Retrain the model 
dvc repro

# 3 register and promote the model Manually
python -m src.register
python -m src.promote

```
`Thanks for Reading`
