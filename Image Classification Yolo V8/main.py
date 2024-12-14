from utils import *
# Create the label folders
# create_folders('dataset2')
# # Create the train and validation directories
# create_train_test()
# validate_dataset('weather_dataset')

# YOLO_train()
YOLO_prediction('./runs/classify/train2/weights/best.pt', './test_folder')
