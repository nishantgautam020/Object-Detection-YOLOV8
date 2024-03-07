import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from shutil import copyfile
from shutil import move


# READING THE ORIGINAL DATA ( IMAGES AND ASSOCIATED LABELS)

# Specify the folders containing the images and labels
input_image_folder = "/Users/nishantgautam/Desktop/buro/training/images/" # folder location containing images 
input_label_folder = "/Users/nishantgautam/Desktop/buro/training/labels/" # folder location containing labels


# Initializing an Empty Array
data = []

# Get a list of image files in the input folder
image_files = [f for f in os.listdir(input_image_folder) if f.endswith((".jpg", ".jpeg", ".png", ".gif", '.bmp'))]

# sort the filenames based on the embedded numerical values in the file name.
image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
print("------------------------------------------------")
print("Total Number of Images in the Dataset:", len(image_files))

# Loop through all images in the input folder
for image_file in image_files:
    image_path = os.path.join(input_image_folder, image_file)
    image = Image.open(image_path)
    
    # Get the size (width and height) of each image
    width, height = image.size
    
    # Read data from the corresponding label file
    label_file = os.path.join(input_label_folder, os.path.splitext(image_file)[0] + '.txt')
    with open(label_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Split each line into class_id, x_center, y_center, bbox_width, bbox_height
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.split())
            
            # Append label data to the list
            data.append({'Image_File': image_file, 'Width': width, 'Height': height,
                         'class_id': int(class_id), 'x_center': x_center,
                         'y_center': y_center, 'bbox_width': bbox_width, 'bbox_height': bbox_height})

# Create a DataFrame from the collected data
df_combine = pd.DataFrame(data)

print(df_combine)

