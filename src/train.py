from ultralytics import YOLO
from src import config, utils
import os

def train_model():
    # Set seed for reproducibility
    utils.set_seed(config.SEED)
    
    # Load the pre-trained YOLOv11 model
    model = YOLO(config.YOLO_WEIGHTS)
    
    print(f"Using device: {config.DEVICE}")
    
    # Train the model
    results = model.train(
        data=config.YAML_OUTPUT_PATH,
        epochs=config.EPOCHS,
        imgsz=config.IMAGE_SIZE,
        device=config.DEVICE,
        seed=config.SEED,
        augment=False  # Disable augmentations as dataset is pre-augmented
    )
    
    print("Training completed.")
    return model

if __name__ == "__main__":
    train_model()
