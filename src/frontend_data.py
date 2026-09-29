import random
from src.config import PROJECT_ROOT, OUTPUT_DIR, TRAIN_NAME, DEVICE
from src.inference import RailDefectDetector



test_dir=PROJECT_ROOT/"data_7class/test/images"
test_images=[
    image for image in test_dir.iterdir() 
    if image.suffix.lower()==".jpg"]

random.seed(42)

sample_images=random.sample(
    test_images,
    min(5, len(test_images))
)

frontend_image_dir=(
    PROJECT_ROOT
    /"outputs"
    /"frontend_data"
    /"images"
)

frontend_image_dir.mkdir(
    parents=True,
    exist_ok=True
)

#### Make predition with model
model_path=OUTPUT_DIR/TRAIN_NAME/"weights"/"best.pt"
model=RailDefectDetector(model_path, DEVICE)

# predict 5 images
for image in sample_images:

    results=model.predict(
        image,
        confidence=0.25
    )
    results[0].save(
        filename=str(
            frontend_image_dir/image.name
        )
    )

