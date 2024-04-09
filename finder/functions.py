import os
from pathlib import Path
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from finder.region import ArrayRegion

def load_h5(images_dir:Path) -> tuple:
    print("Loading images from:", images_dir)
    # images with "processed" for background demonstration
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.h5') and f.startswith('img')]
    if not image_files:
        raise FileNotFoundError("No processed image files found in the directory.")
    random_image = np.random.choice(image_files) # random image
    image_path = os.path.join(images_dir, random_image)
    print("Loading image:", random_image)
    try:
        with h5.File(image_path, 'r') as file:
            data = file['entry/data/data'][:]
        return data, image_path
    except Exception as e:
        raise OSError(f"Failed to read {image_path}: {e}")

def find_dir(base_path:str, dir_name:str) -> Path:
    for path in base_path.rglob('*'):
        if path.name == dir_name and path.is_dir():
            return path
    raise FileNotFoundError(f"{dir_name} directory not found.")   

def display_peaks_3d(image, peaks, threshold, img_threshold=0.005):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # grid 
    x, y = np.arange(0, image.shape[1]), np.arange(0, image.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = np.ma.masked_less_equal(image, img_threshold)
    
    # plot surface/image data 
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False, alpha=0.6)
    ax.set_title('3D View of Image with Detected Peaks')
    
    valid_peaks = [(px, py) for px, py in peaks if px < image.shape[0] and py < image.shape[1]]
    peak_intensities = np.array([image[px, py] for px, py in valid_peaks])
    above_threshold = peak_intensities > threshold
    
    if np.any(above_threshold):
        p_x, p_y = np.array(valid_peaks)[above_threshold].T
        p_z = peak_intensities[above_threshold]
        ax.scatter(p_y, p_x, p_z, color='r', s=50, marker='x', label='Peaks')
    
    # labels 
    ax.set_title('3D View of Image with Detected Peaks')
    ax.set_xlabel('X-axis (ss)')
    ax.set_ylabel('Y-axis (fs)')
    ax.set_zlabel('Intensity')
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Intensity')

    plt.legend()
    plt.show()
    
def display_peaks_3d_beamstop(image, threshold_value):
    # Assuming center is at the middle of the image
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    
    # Create exclusion mask around the center
    region_handler = ArrayRegion(image)
    region_handler.set_peak_coordinate(center_x, center_y)
    region_handler.set_region_size(8)
    exclusion_mask = region_handler.get_exclusion_mask()
    
    # Apply mask
    masked_image = np.ma.array(image, mask=~exclusion_mask)
    
    # Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    X, Y = np.meshgrid(x, y)
    
    # Surface plot
    surf = ax.plot_surface(X, Y, masked_image, cmap='viridis', edgecolor='none', alpha=0.5)
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Intensity')
    
    # Find and plot peaks outside the excluded region
    coordinates = np.argwhere(masked_image > threshold_value)
    if coordinates.size > 0:
        p_x, p_y = coordinates[:, 1], coordinates[:, 0]
        p_z = masked_image[p_y, p_x]
        ax.scatter(p_x, p_y, p_z, color='r', s=20, marker='o', label='Peaks')

    ax.set_title('3D View of Water Ring with Excluded Center')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Intensity')
    plt.legend()
    plt.show()