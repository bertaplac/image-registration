# Load the required libraries
import os
from itertools import groupby
import numpy as np
from IPython.display import display, Image
import cv2
from scipy.optimize import differential_evolution
import torch.nn.functional as F
import torch


folder_path = "path/to/folder/with/all/images"

# Get a list of all files in the folder
all_files = os.listdir(folder_path)

# Sort the files to ensure they are grouped together
all_files.sort()

# Define a function to extract the common part of the filenames
def extract_common_part(filename):
    return filename.split('_')[0]

# Group files based on the common part of the filenames
grouped_files = {key: [os.path.join(folder_path, file) for file in group] for key, group in groupby(all_files, key=extract_common_part)}

# Print the grouped files
for group, files in grouped_files.items():
    print(f"Group: {group}")
    for file_path in files:
        print(f"  {file_path}")
    print("\n")

# Function to calculate mean squared error for shifted images
def calculate_mse_shifted_2(shift_x, shift_y):
    idx_x = shift_x.int()
    idx_y = shift_y.int()

    # Crop overlapping regions of images based on shift values
    overlap_base_image = base_image[idx_y:, max(0, idx_x):base_image.shape[1] + min(0, idx_x)]
    overlap_shift_image = shift_image[:shift_image.shape[0] - idx_y, max(0, -idx_x):shift_image.shape[1] - max(0, idx_x)]

    # Compute MSE only if the overlap is significant
    if overlap_base_image.numel() >= (base_image.numel() * 0.3):
        mse = F.mse_loss(overlap_base_image, overlap_shift_image)
        return mse

    return torch.tensor(float('inf'))

# Objective function for optimization
def objective(params):
    shift_x.data = torch.tensor(params[0])
    shift_y.data = torch.tensor(params[1])
    return calculate_mse_shifted_2(shift_x, shift_y).item()

# Function to refine the minimum MSE after optimization
def find_minimum_mse(shift_x_min, shift_y_min):
    minimum_mse = float("inf")
    x_min = 0
    y_min = 0

    # Search in a small range around the found minimum
    for x in range(int(shift_x_min)-5, int(shift_x_min)+5):
        for y in range(int(shift_y_min)-5, int(shift_y_min)+5):
            x_tensor = torch.tensor(x)
            y_tensor = torch.tensor(y)

            curr_mse = calculate_mse_shifted_2(x_tensor, y_tensor).item()

            # Update if a lower MSE is found
            if curr_mse < minimum_mse:
                minimum_mse = curr_mse
                x_min = x
                y_min = y

    return minimum_mse, x_min, y_min

# Define bounds for the shift values
bounds = [(-100, 101), (500, 2001)]

# Process each group of files
for group_key, group_files in grouped_files.items():
    print(f"Group: {group_key}")
    translation = [[0,0]]
    
    # Load and process the base image
    base_image = cv2.imread(group_files[0])
    base_image = cv2.resize(base_image, None, fx=0.5, fy=0.5)
    base_image = torch.as_tensor(base_image, dtype=torch.float32)
    
    # Iterate over consecutive pairs of images in the group
    for i in range(len(group_files) - 1):
        shift_image = cv2.imread(group_files[i + 1])
        shift_image = cv2.resize(shift_image, None, fx=0.5, fy=0.5)
        shift_image = torch.as_tensor(shift_image, dtype=torch.float32)

        shift_x = torch.tensor(0.0, requires_grad=True)
        shift_y = torch.tensor(0.0, requires_grad=True)

        # Perform differential evolution optimization
        result = differential_evolution(objective, bounds)
        shift_x_min, shift_y_min = result.x

        # Refine the result by a local search
        result = find_minimum_mse(shift_x_min, shift_y_min)
        translation.append([translation[-1][0] + result[1], translation[-1][1] + result[2]])
        
        # Update the base image for the next iteration
        base_image = shift_image
            
    print(translation)
      
    # Calculate the dimensions of the stitched image
    lowest_x = min(pair[0] for pair in translation)
    highest_x = max(pair[0] for pair in translation)
    highest_y = translation[-1][1]
    stitched_height = base_image.shape[0] + abs(highest_y)
    stitched_width = base_image.shape[1] + abs(lowest_x) + abs(highest_x)
    
    # Save the resulting stitched image using the best shift combinations
    stitched_image = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)
    
    # Iterate over images and place them in the stitched image
    for i in range(len(group_files)):
        image_path = group_files[i]
        image = cv2.imread(image_path)
        image = cv2.resize(image, None, fx=0.5, fy=0.5)

        # Calculate the position based on translation
        x_pos = translation[i][0] - lowest_x
        y_pos = translation[i][1]

        # Place the image in the stitched image at the calculated position
        stitched_image[y_pos:y_pos + image.shape[0], x_pos:x_pos + image.shape[1]] = image

    cv2.imwrite(f'best_stitched_image_{group_key}.jpg', stitched_image)

    # Display the resulting stitched image in the notebook
    display(Image(filename=f'best_stitched_image_{group_key}.jpg'))
       
    print("\n")