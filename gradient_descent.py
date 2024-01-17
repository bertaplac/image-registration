# Load the required libraries
import cv2
import torch
from torch.optim import SGD
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


def calculate_mse_shifted_2(shift_x, shift_y, base_image, shift_image):
    # Reshape to [1, C, H, W]. It converts images to 4D tensors for processing
    base_image = base_image.permute(2, 0, 1).unsqueeze(0).float()
    shift_image = shift_image.permute(2, 0, 1).unsqueeze(0).float()

    # Normalize shifts to be in the range [-1, 1]
    shift_x_normalized = 2 * shift_x / base_image.size(3)  # Normalize by width
    shift_y_normalized = 2 * shift_y / base_image.size(2)  # Normalize by height

    # Create the affine transformation matrix
    theta = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float).unsqueeze(0)
    theta[:, 0, 2] = -shift_x_normalized
    theta[:, 1, 2] = -shift_y_normalized
    
    # Apply the affine transformation
    grid = F.affine_grid(theta, shift_image.size(), align_corners=False)
    shifted_image = F.grid_sample(shift_image, grid, align_corners=False)

    # Create masks for overlapping regions
    base_mask = torch.ones_like(base_image)
    shifted_mask = F.grid_sample(base_mask, grid, align_corners=False)

    # Apply the masks to the images
    base_image_masked = base_image * shifted_mask
    shifted_image_masked = shifted_image * shifted_mask

    # Compute MSE on the masked (overlapping) regions
    mse = F.mse_loss(base_image_masked, shifted_image_masked)
    return mse

# Initialize shift variables with gradient tracking
shift_x = torch.tensor(0.0, requires_grad=True)
shift_y = torch.tensor(1500.0, requires_grad=True)

# Set up optimizer w/different learning rates for shift_x and shift_y
optimizer = SGD([shift_x, shift_y], lr=100)
optimizer = SGD([{'params': [shift_x], 'lr': 5},
                  {'params': [shift_y], 'lr': 50}])

# Initialize variables for monitoring optimization
iteration_data = []
threshold = 0.1  
previous_loss = None

# Perform optimization for a maximum of 100 iterations
for i in range(100):  
    optimizer.zero_grad()
    loss = calculate_mse_shifted_2(shift_x, shift_y, base_image, shift_image)
    loss.backward()
    optimizer.step()
    
    # Applying constraints to shift_x and shift_y
    with torch.no_grad():
        shift_x.clamp_(-100, 100)
        shift_y.clamp_(500, 2000)
    
    current_loss = loss.item()
    print(f"Iteration {i}, Loss: {current_loss}, Shift x: {shift_x.item()}, Shift y: {shift_y.item()}")
    iteration_data.append((shift_x.item(), shift_y.item(), current_loss))
    
    # Check for convergence and adjust learning rate if needed
    if previous_loss is not None and abs(previous_loss - current_loss) < threshold:
        if threshold > 0.01:
            print("Convergence reached. Reducing learning rate for fine-tuning.")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                threshold = 0.01 
        else: break

    previous_loss = current_loss

# Output the final optimal shift values
shift_x_min = shift_x.item()
shift_y_min = shift_y.item()

print(f"Optimal shift_x: {shift_x_min}, Optimal shift_y: {shift_y_min}")