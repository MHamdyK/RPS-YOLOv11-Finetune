# src/config.py

import os

# Set seed for reproducibility
SEED = 42

# Device configuration
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or __import__("torch").cuda.is_available() else "cpu"

# Dataset paths (update these if you change your directory structure)
DATASET_ZIP_PATH = '/content/drive/MyDrive/yolodataset/rock-paper-scissors.v14i.yolov11.zip'
EXTRACTED_DATA_DIR = 'rock_paper_scissors'
YAML_INPUT_PATH = os.path.join(EXTRACTED_DATA_DIR, 'data.yaml')
YAML_OUTPUT_PATH = 'data.yaml'

# Training hyperparameters
EPOCHS = 10
IMAGE_SIZE = 640

# Pre-trained model weight file
YOLO_WEIGHTS = 'yolo11m.pt'
