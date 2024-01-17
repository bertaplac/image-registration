# Image Processing Scripts for Alignment and Stitching

This repository contains a set of Python scripts designed for image alignment and stitching using various optimization techniques. The scripts demonstrate different approaches, including exhaustive search, randomized search, differential evolution, and gradient descent, to minimize the mean squared error (MSE) between images and to stitch them together.

## Scripts

### Script 1: Exhaustive Search for Image Alignment
`exhaustive_search.py`
- Performs an exhaustive search over a predefined range of shift values to find the best alignment of two images.
- Minimizes MSE by trying every combination of shifts within the range.
- Saves and displays the stitched image with the best shift combination.

### Script 2: Randomized Search for Image Alignment
`randomized_search.py`
- Aligns two images by sampling shift values randomly to minimize MSE.
- Uses an iterative approach with a convergence threshold for optimization.

### Script 3: Differential Evolution for Image Alignment
`differential_evolution.py`
- Employs differential evolution for global optimization to align two images.
- Includes a function to refine the minimum MSE after optimization.

### Script 4: Complete Blade Stitching with Differential Evolution
`differential_evolution_Complete_blade.py`
- Groups and stitches images based on a common part of their filenames.
- Optimizes the alignment of each pair of images using differential evolution and local search.
- Constructs and saves the final stitched image for each group.

### Script 5: Gradient Descent for Image Alignment
`gradient_descent.py`
- Utilizes gradient descent with the SGD optimizer to align two images.
- Applies affine transformations and calculates MSE on masked overlapping regions.
- Monitors loss for convergence and adjusts learning rates for fine-tuning.

## Dependencies
- Python 3
- OpenCV (cv2)
- NumPy
- PyTorch
- SciPy
- scikit-learn
- IPython (for display in Jupyter notebooks)
