# Load the required libraries
import cv2
import numpy as np
from sklearn.metrics import mean_squared_error
from IPython.display import display, Image

# Load a pair of images and resizes them to half their original size
base_image = cv2.imread('path/to/img/1.jpg')
base_image = cv2.resize(base_image, None, fx=0.5, fy=0.5)

shift_image = cv2.imread('path/to/img/2.jpg')
shift_image = cv2.resize(shift_image, None, fx=0.5, fy=0.5)

# Define the increment for shift_x and shift_y
shift_x_increment = 10
shift_y_increment = 10

# Define the range of shifts
shift_x_range = range(-100, 100, shift_x_increment)
shift_y_range = range(500, 2000, shift_y_increment)

# Define a minimum required overlap size (in pixels)
min_overlap_size = (base_image.shape[0] * base_image.shape[1]) * 0.3

# Initialize a dictionary to store MSE values for each combination
mse_values = {}

# Initialize lists to store shift_x, shift_y, and MSE values
shift_x_values = []
shift_y_values = []
mse_values_list = []

for shift_x in shift_x_range:
    for shift_y in shift_y_range:
        # Calculate the dimensions of the stitched image
        stitched_height = base_image.shape[0] + abs(shift_y)
        stitched_width = base_image.shape[1] + abs(shift_x)

        # Create an empty canvas to stitch the images
        stitched_image = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)

        # Copy base_image to the left side of the canvas
        stitched_image[:base_image.shape[0], max(0, -shift_x):base_image.shape[1] + max(0, -shift_x)] = base_image

        # Copy shift_image shifted by the specified amount to the right side of the canvas
        stitched_image[shift_y:shift_y + shift_image.shape[0], max(0, shift_x): max(0, shift_x) + shift_image.shape[1]] = shift_image

        # Crop the overlapping region from both images
        overlap_base_image = base_image[shift_y:, max(0, shift_x):base_image.shape[1] + min(0, shift_x)]
        overlap_shift_image = shift_image[:shift_image.shape[0] - shift_y, max(0, -shift_x):shift_image.shape[1] - max(0, shift_x)]

        # Check if both overlapping regions have non-zero dimensions
        if overlap_base_image.shape[0] * overlap_base_image.shape[1] >= min_overlap_size:
            # Compute the Mean Square Error (MSE) between the overlapping regions
            mse = mean_squared_error(overlap_base_image.ravel(), overlap_shift_image.ravel())

            # Store the shift values and MSE in the lists
            shift_x_values.append(shift_x)
            shift_y_values.append(shift_y)
            mse_values_list.append(mse)

            # Store the MSE value in the dictionary with shift_x and shift_y as keys
            mse_values[(shift_x, shift_y)] = mse

# Find the shift combination with the minimum MSE
best_shift = min(mse_values, key=mse_values.get)
best_mse = mse_values[best_shift]

# Calculate the dimensions of the stitched image
stitched_height = base_image.shape[0] + abs(best_shift[1])
stitched_width = base_image.shape[1] + abs(best_shift[0])

# Save the resulting stitched image using the best shift combination
stitched_image = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)
stitched_image[:base_image.shape[0], max(0, -best_shift[0]):base_image.shape[1] + max(0, -best_shift[0])] = base_image
stitched_image[best_shift[1]:best_shift[1] + shift_image.shape[0], max(0, best_shift[0]): max(0, best_shift[0]) + shift_image.shape[1]] = shift_image
cv2.imwrite('best_stitched_image.jpg', stitched_image)

# Display the resulting stitched image in the notebook
display(Image(filename='best_stitched_image.jpg'))

# Display the best shift combination and its MSE
print(f"Best Shift (shift_x, shift_y): {best_shift}")
print(f"Minimum Mean Square Error (MSE): {best_mse:.2f}")