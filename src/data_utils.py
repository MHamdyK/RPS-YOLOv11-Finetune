import os
import shutil
import zipfile

def unzip_dataset(zip_path, extract_dir):
    """Unzip the dataset if not already unzipped."""
    if not os.path.exists(extract_dir):
        print("Unzipping dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Dataset unzipped successfully.")
    else:
        print("Dataset already unzipped.")

def update_yaml_paths(input_yaml_path, output_yaml_path):
    """Update the dataset YAML file paths to absolute paths."""
    with open(input_yaml_path, 'r') as file:
        data_yaml = file.read()
    
    # Adjust paths (modify these as needed)
    data_yaml = data_yaml.replace('../train/images', f'/content/{os.path.join("rock_paper_scissors", "train", "images")}')
    data_yaml = data_yaml.replace('../valid/images', f'/content/{os.path.join("rock_paper_scissors", "valid", "images")}')
    data_yaml = data_yaml.replace('../test/images', f'/content/{os.path.join("rock_paper_scissors", "test", "images")}')
    
    with open(output_yaml_path, 'w') as file:
        file.write(data_yaml)
    print("YAML paths updated.")
