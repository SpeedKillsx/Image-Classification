import os
import shutil
import random
import cv2 as cv
from ultralytics import YOLO
import matplotlib.pyplot as plt
def create_dataset(name="FER", folder="train"):
    """Create a facial emotion dataset

    Args:
        name (str, optional): Desired name given to the dataset. Defaults to "FER".
    """
    # Create the Dataset directories
    os.makedirs(name, exist_ok=True)
    os.makedirs(os.path.join(name, 'train'), exist_ok=True)
    os.makedirs(os.path.join(name, 'val'), exist_ok=True)

    # Load all classes
    classes = [f for f in os.listdir(folder)]
    for label in classes:
        # Load all images for the label
        list_images = [os.path.join(label, image) for image in os.listdir(os.path.join(folder, label))]
        list_image_folder = []

        for image_path in list_images:
            image = cv.imread(os.path.join(folder, image_path))
            if image is None:
                print(f"Error: Cannot read the image: {image_path}")
                os.remove(os.path.join(folder, image_path))  # Remove invalid images
            else:
                list_image_folder.append(os.path.join(folder, image_path))

        # Shuffle and split into train and validation sets
        random.shuffle(list_image_folder)
        train_list = list_image_folder[:len(list_image_folder) - 100]
        val_list = list_image_folder[len(list_image_folder) - 100:]

        # Move files in train_list
        for image_train in train_list:
            train_folder = os.path.join(name, 'train', label)
            os.makedirs(train_folder, exist_ok=True)
            destination = os.path.join(train_folder, os.path.basename(image_train))
            if os.path.exists(destination):
                print(f"Conflict: {destination} already exists. Skipping.")
            else:
                shutil.move(image_train, destination)
                print(f'{image_train} moved to {destination}')

        # Move files in val_list
        for image_val in val_list:
            val_folder = os.path.join(name, 'val', label)
            os.makedirs(val_folder, exist_ok=True)
            destination = os.path.join(val_folder, os.path.basename(image_val))
            if os.path.exists(destination):
                print(f"Conflict: {destination} already exists. Skipping.")
            else:
                shutil.move(image_val, destination)
                print(f'{image_val} moved to {destination}')
def YOLO_train():
    model = YOLO("yolov8n-cls.pt")
    model.train(data="G:/Computer Vision/Image Classification/Face Emotion Recognition/FER",epochs=200,imgsz=64)

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
        