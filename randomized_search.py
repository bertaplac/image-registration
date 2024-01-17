# Load the required libraries
import cv2
from sklearn.metrics import mean_squared_error
import random

# Load a pair of images and resizes them to half their original size
base_image = cv2.imread('path/to/img/1.jpg')
base_image = cv2.resize(base_image, None, fx=0.5, fy=0.5)

shift_image = cv2.imread('path/to/img/2.jpg')
shift_image = cv2.resize(shift_image, None, fx=0.5, fy=0.5)

# Function to calculate mean squared error for shifted images
def calculate_mse_shifted(shift_x, shift_y):
    # Crop overlapping regions of images based on shift values
    overlap_base_image = base_image[shift_y:, max(0, shift_x):base_image.shape[1] + min(0, shift_x)]
    overlap_shift_image = shift_image[:shift_image.shape[0] - shift_y, max(0, -shift_x):shift_image.shape[1] - max(0, shift_x)]

    # Compute mean squared error between the overlapping regions
    mse = mean_squared_error(overlap_base_image.ravel(), overlap_shift_image.ravel())
    return mse

# Initialize variables for the optimization process
min_x, min_y, min_z = 0, 1500, 1e6
previous_min_z = min_z
iteration = 0
convergence_threshold = 1e-6

# Iterative process to find the shift that minimizes MSE
while True:
    if iteration % 2 == 0:
        # In odd iterations, vary y while keeping x fixed
        y_values = [min_y + random.randint(-50, 50) for _ in range(10)]
        new_data = [(min_x, y, calculate_mse_shifted(min_x, y)) for y in y_values]
    else:
        # In even iterations, vary x while keeping y fixed
        x_values = [min_x + random.randint(-10, 10) for _ in range(10)]
        new_data = [(x, min_y, calculate_mse_shifted(x, min_y)) for x in x_values]

    # Find the minimum MSE in the new set of data
    new_min_x, new_min_y, new_min_z = min(new_data, key=lambda item: item[2])

    # Check if the change in MSE is below the convergence threshold
    if abs(new_min_z - previous_min_z) < convergence_threshold:
        break

    # Update minimum values for the next iteration
    min_x, min_y, min_z = new_min_x, new_min_y, new_min_z
    previous_min_z = new_min_z
    iteration += 1

# Display the best shift combination and its MSE
print(f"Best Shift (shift_x, shift_y): ({min_x}, {min_y})")
print(f"Minimum Mean Square Error (MSE): {min_z:.2f}")
