# Load the required libraries
import cv2
from scipy.optimize import differential_evolution
import torch
import torch.nn.functional as F

# Load a pair of images and resizes them to half their original size
base_image = cv2.imread('path/to/img/1.jpg')
base_image = cv2.resize(base_image, None, fx=0.5, fy=0.5)
# Convert to PyTorch tensor for processing
base_image = torch.as_tensor(base_image, dtype=torch.float32)

shift_image = cv2.imread('path/to/img/2.jpg')
shift_image = cv2.resize(shift_image, None, fx=0.5, fy=0.5)
# Convert to PyTorch tensor for processing
shift_image = torch.as_tensor(shift_image, dtype=torch.float32)

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

# Initialize variables for differential evolution algorithm
shift_x = torch.tensor(0.0, requires_grad=True)
shift_y = torch.tensor(0.0, requires_grad=True)

# Define bounds for the shift values
bounds = [(-100, 101), (500, 2001)]

# Perform differential evolution optimization
result = differential_evolution(objective, bounds)
shift_x_min, shift_y_min = result.x

# Refine the result by a local search
result = find_minimum_mse(shift_x_min, shift_y_min)

# Display the best shift combination and its MSE
print(f"Best Shift (shift_x, shift_y): ({result[1]}, {result[2]})")
print(f"Minimum Mean Square Error (MSE): {result[0]:.2f}")