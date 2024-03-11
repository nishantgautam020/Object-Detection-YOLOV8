import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from shutil import copyfile
from shutil import move
import matplotlib.pyplot as plt
import data_processing
import model_training
import video_prediction


def object_detection():
    # Function calls to read images and display images with labelss
    data_processing.read_images()

    # MAPPING LABELS OVER IMAGES / PLOTTING FOR VISUALIZATIONS

    # Identifying Annotation Classess
    print("                                          CLASSES", )
    print("                                       Class 0: Door", )
    print("                                       Class 1: Window", )
    print("                                       Class 2: Zone/Room", )


    # Example usage:
    image_number = 521
    data_processing.display_image_with_labels(image_number)


    # Model Training(yolov8)
    model_training.model_train()

    # Pretrained Model Prediction on a video
    video_prediction.model_predict()

# Object Detection Start
object_detection()
