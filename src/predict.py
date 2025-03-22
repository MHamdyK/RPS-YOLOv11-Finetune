import os
import random
import matplotlib.pyplot as plt
from ultralytics import YOLO
from src import config, utils

def display_evaluation_images(model):
    # Display confusion matrix and loss curve images from runs folder
    confusion_matrix_path = os.path.join('runs', 'detect', 'train', 'confusion_matrix.png')
    loss_curve_path = os.path.join('runs', 'detect', 'train', 'results.png')
    
    # Check and plot if files exist
    if os.path.exists(confusion_matrix_path):
        utils.plot_image(plt.imread(confusion_matrix_path), title="Confusion Matrix")
    else:
        print("Confusion matrix not found.")
        
    if os.path.exists(loss_curve_path):
        utils.plot_image(plt.imread(loss_curve_path), title="Loss Curve")
    else:
        print("Loss curve image not found.")

def predict_samples(model, num_samples=3):
    test_path = os.path.join(config.EXTRACTED_DATA_DIR, 'test', 'images')
    test_images = os.listdir(test_path)
    sample_images = random.sample(test_images, num_samples)
    
    for image_name in sample_images:
        img_path = os.path.join(test_path, image_name)
        results = model.predict(img_path)
        # Assuming results[0].plot() returns an image array
        output_image = results[0].plot()
        utils.plot_image(output_image, title=f"Prediction: {image_name}")

if __name__ == "__main__":
    # Load the trained model (assumes training was just done)
    model = YOLO(config.YOLO_WEIGHTS)
    display_evaluation_images(model)
    predict_samples(model)
