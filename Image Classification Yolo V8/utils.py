import os
import cv2 as cv
import numpy as np
import shutil
import random
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import matplotlib.pyplot as plt
def create_folders(dataset_path, labels = ['shine', 'cloudy', 'sunrise', 'rain']):
    """Function that split the entire dataset to different folders according to the different image labels.

    Args:
        dataset_path (__str__): Path to the dataset
        labels (__List__): List of all lables desired for the dataset
    """
    #Create the folders
    try:
        for label in labels : os.mkdir(label)
    except:
        print("An error occured during the creation of the folder")
    finally:
        print("Process terminated")
    
    # Start putting images on the right folder
    images_list = os.listdir(dataset_path)
    for label in labels:
        for image in images_list:
            if image.find(label)!=-1:
                shutil.copyfile('dataset2/'+image, label+'/'+image)
    print('Folders created')


def create_train_test(labels=['shine', 'cloudy', 'sunrise', 'rain']):
    # Define folders
    train_folder = "train"
    val_folder = "val"
    dataset = 'weather_dataset'
    dataset_train = os.path.join(dataset, train_folder)
    dataset_val = os.path.join(dataset, val_folder)
    os.makedirs(dataset, exist_ok=True)
    # Create train and validation folders if not already present
    os.makedirs(dataset_train, exist_ok=True)
    os.makedirs(dataset_val, exist_ok=True)
    
    for label in labels:
        # Ensure label-specific subdirectories exist in train and val
        os.makedirs(os.path.join(dataset_train, label), exist_ok=True)
        os.makedirs(os.path.join(dataset_val, label), exist_ok=True)
        
        # List files in the label directory and filter for files only
        list_files = [f for f in os.listdir(label) if os.path.isfile(os.path.join(label, f))]
        list_files = np.array(list_files)
        
        print(f"Processing {label}: {list_files.shape[0]} files found.")
        
        # Shuffle files
        random.shuffle(list_files)
        
        # Split into train and validation sets
        split_index = max(0, len(list_files) - 60)  # Avoid negative index
        train_files = list_files[:split_index]
        val_files = list_files[split_index:]
        
        # Move files to train/val folders
        for file in train_files:
            print(dataset_train)
            shutil.move(os.path.join(label, file), os.path.join(dataset_train, label, file))
        for file in val_files:
            shutil.move(os.path.join(label, file), os.path.join(dataset_val, label, file))

        print(f"{label}: {len(train_files)} training files, {len(val_files)} validation files.")
    for label in labels:
        os.removedirs(label)
        print(f"Directory {label} removed")
    # Make verification
    validate_dataset(dataset)



def validate_dataset(folder):
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            filepath = os.path.join(subdir, file)
            image = cv.imread(filepath) # Read the image
            if image is None:
                print(f"Invalid image file: {filepath}")
                os.remove(filepath)  

def YOLO_train():
    model = YOLO("yolov8n-cls.pt")
    model.train(data="G:/Computer Vision/Image Classification/Image Classification Yolo V8/weather_dataset",epochs=20,imgsz=64)

def  YOLO_prediction(best_model: str, image_test_path:str):
    model  = YOLO(best_model)
    test_images = [os.path.join(image_test_path, f) for f in os.listdir(image_test_path)]
    results = model(test_images, show_labels=True)
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        image_test = cv.cvtColor(result.orig_img, cv.COLOR_BGR2RGB)
        plt.imshow(image_test)
        plt.title(result.names[result.probs.top1])
        plt.show()
        
        