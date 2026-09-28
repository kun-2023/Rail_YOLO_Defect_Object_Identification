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
```