from fastapi import FastAPI
from api.routers import router


app=FastAPI(
    title="Railway Defect Detection API",
    description="Railway Yolo object identification",
    version="1.0.0"
)

app.include_router(router)