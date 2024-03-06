import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from shutil import copyfile
from shutil import move

# READING THE ORIGINAL DATA (ONLY IMAGES)
# Specify the folder containing the images
input_folder = "/Users/nishantgautam/Desktop/buro/training/images/"

# Create output folders for different blur categories
# output_folder_base = input('Output Folder Location:')


data = []

# Get a list of image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith((".jpg", ".jpeg", ".png", ".gif", '.bmp'))]
image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
print("------------------------------------------------")
print("Total Number of Images in the Dataset:", len(image_files))

# Loop through all images in the input folder
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = Image.open(image_path)
    
    # Get the size of each image
    width, height = image.size
    
    # Save image size data in the list
    data.append({'Image_File': image_file, 'Width': width, 'Height': height})

# Create a DataFrame from the collected data
image_data_df = pd.DataFrame(data)

print(image_data_df)