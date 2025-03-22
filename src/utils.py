import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_image(image, title=""):
    """Display an image with matplotlib."""
    plt.figure(figsize=(8,8))
    plt.title(title)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
