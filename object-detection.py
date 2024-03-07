import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from shutil import copyfile
from shutil import move


# Set paths
input_image_folder = "/Users/nishantgautam/Desktop/buro/training/images" # folder location containing original images
input_label_folder = "/Users/nishantgautam/Desktop/buro/training/labels/" # folder location containing labels
output_image_folder = "/Users/nishantgautam/Desktop/buro/training/re_images" # folder location containing resized images
target_size = 300  # Choose your target size


# Initializing an empty array
data = []

# Get a list of image files in the input folder
image_files = [f for f in os.listdir(input_image_folder) if f.endswith((".jpg", ".jpeg", ".png", ".gif", '.bmp'))]
image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
print("------------------------------------------------")
print("Total Number of Images in the Dataset:", len(image_files))


# Create the output folder if it doesn't exist
if not os.path.exists(output_image_folder):
    os.makedirs(output_image_folder)

# Iterate through each image
for image_file in image_files:
    image_path = os.path.join(input_image_folder, image_file)
    
    # Resize the image while maintaining aspect ratio
    image = Image.open(image_path)
    width, height = image.size
    aspect_ratio = width / height
    new_width = target_size
    new_height = int(new_width / aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Read data from the corresponding label file
    label_file = os.path.join(input_label_folder, os.path.splitext(image_file)[0] + '.txt')
    with open(label_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Split each line into class_id, x_center, y_center, bbox_width, bbox_height
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.split())
            
            # Append label data to the list
            data.append({'Image_File': image_file, 'Width': new_width, 'Height': new_height,
                         'class_id': int(class_id), 'x_center': x_center,
                         'y_center': y_center, 'bbox_width': bbox_width, 'bbox_height': bbox_height})

    # Save the resized image to the output folder
    output_image_path = os.path.join(output_image_folder, image_file)
    resized_image.save(output_image_path)


# Create DataFrame
df_img_resize = pd.DataFrame(data)

# # Save DataFrame to a CSV file
# df.to_csv("/Users/nishantgautam/Desktop/buro/training/data.csv", index=False)

print("Dataframe",df_img_resize)
print("------------------------------------------------")

# gives the shape of object types of our data
print("Dataframe Information: ", df_img_resize.info())
print("------------------------------------------------")

# Columns and Row Stats
print("Shape of the Dataframe: ",df_img_resize.shape)
print("Number of Columns:",len(df_img_resize.columns))
print("Number of Rows:",len(df_img_resize))
print("------------------------------------------------")

#provides a summary of the central tendencies, dispersion, and shape of a dataset's distribution.
print("Statistical Summary of Columns:",df_img_resize.describe())