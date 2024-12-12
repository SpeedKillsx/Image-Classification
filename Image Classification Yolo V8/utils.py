import os
import cv2 as cv
import numpy as np
import shutil


def create_folders(dataset_path, labels = ['sunshine', 'cloudy', 'sunrise', 'rain']):
    """Function that split the entire dataset to different folders according to the different image labels.

    Args:
        dataset_path (__str__): Path to the dataset
    """
    #Create the folders
    try:
        for label in labels : os.mkdir(label)
    except:
        OSError('Erruer')
        print("An error occured during the creation of the")
    finally:
        print("Process terminated")
    
    # Start putting images on the right folder
    images_list = os.listdir(dataset_path)
    for label in labels:
        for image in images_list:
            if image.find(label)!=-1:
                shutil.copyfile('dataset2/'+image, label+'/'+image)
        