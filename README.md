# RPS-YOLOv11-Finetune
YOLOv11 fine-tuning on rock-paper-scissors dataset from roboflow, this was an assignment for my Image Processing and Pattern Recognition (CC516) course at university.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training the Model](#training-the-model)
  - [Evaluation and Inference](#evaluation-and-inference)
- [Package Structure](#package-structure)
- [Submission Notes](#submission-notes)
- [License](#license)

## Overview

- Unzipping and preparing the dataset.
- Updating dataset YAML paths for correct data loading.
- Fine-tuning the model for 10 epochs.
- Plotting the confusion matrix and loss curves.
- Running inference on a few test samples and displaying predictions.

## Directory Structure

RPS-YOLOv11-Finetune/ ├── data/ │ └── README.md # Instructions for downloading/preparing the dataset ├── notebooks/ │ └── yolov11_finetune.ipynb # Jupyter/Colab notebook for interactive exploration ├── src/ │ ├── init.py # Package initializer (imports key modules/functions) │ ├── config.py # Configuration parameters (paths, hyperparameters, etc.) │ ├── data_utils.py # Data preparation (unzipping dataset, updating YAML paths) │ ├── train.py # Training script for YOLOv11 fine-tuning │ ├── predict.py # Evaluation/inference script (plots and predictions) │ └── utils.py # Utility functions (seeding, plotting, etc.) ├── requirements.txt # List of dependencies ├── README.md # This file └── .gitignore # Files and directories to ignore in Git

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your_username/RPS-YOLOv11-Finetune.git
   cd RPS-YOLOv11-Finetune

2. **Install Dependencies:**
pip install -r requirements.txt

3. Training the model
`python src/train.py`

4. Evaluation and Inference
`python src/predict.py`
this script will 
. Display the confusion matrix and loss curve (stored in the `runs` folder).
. Run inference on three random test images and display the predicted outputs.

## Package Structure
from .config import (
    SEED,
    DEVICE,
    DATASET_ZIP_PATH,
    EXTRACTED_DATA_DIR,
    YAML_INPUT_PATH,
    YAML_OUTPUT_PATH,
    EPOCHS,
    IMAGE_SIZE,
    YOLO_WEIGHTS,
)

from .data_utils import (
    unzip_dataset,
    update_yaml_paths,
)

from .train import train_model

from .predict import (
    display_evaluation_images,
    predict_samples,
)

from .utils import (
    set_seed,
    plot_image,
)
