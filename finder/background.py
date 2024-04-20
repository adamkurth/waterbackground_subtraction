import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
import pandas as pd
from typing import Tuple, List, Union
from pathlib import Path
import torch
from .threshold import PeakThresholdProcessor
from .region import ArrayRegion
from .functions import load_h5

class BackgroundSubtraction:
    def __init__(self, threshold: int = 1000) -> None:
        self.threshold = threshold
        self.radii = [1, 2, 3, 4]
            
    """torch.tensor below"""
    def process_tensor(self, tensors:torch.Tensor) -> pd.DataFrame:
        """Process a list of tensors."""
        # provide tensor, as input and cast to numpy array.
        # loaded_image = input_tensor.cpu().numpy()     
        all_data = [self.process_single_tensor(input_tensor=tensor) for tensor in tensors]
        return pd.concat(all_data, ignore_index=True)
    
    def process_single_tensor(self, input_tensor: torch.Tensor) -> pd.DataFrame:
        """Process a single tensor."""
        # loaded_image = input_tensor.cpu().numpy()
        
        print(f"----- pre-loaded image size : {input_tensor.size()} -----")   
        loaded_image = input_tensor.squeeze(0).cpu().numpy()
        print(f"----- loaded image size : {loaded_image.shape} -----") 
        
        print(f'Tensor -> Numpy: {loaded_image}')
        
        p = PeakThresholdProcessor(image=loaded_image, threshold_value=self.threshold)
        coordinates = p.get_coordinates_above_threshold()    
        if coordinates.any():
            data = [self.analyze_region(loaded_image, coord, r) for coord in coordinates for r in self.radii]
            return pd.DataFrame(data)
        else:
            print("No coordinates found above threshold. Returning empty DataFrame.")
            return pd.DataFrame()

    """numpy below"""
    def process_single_image(self, image_path: str) -> pd.DataFrame:
        loaded_image, _ = load_h5(file_path=image_path)
        # print(f"Image loaded with shape: {loaded_image.shape}")  # Debugging line

        p = PeakThresholdProcessor(image=loaded_image, threshold_value=self.threshold)
        coordinates = p.get_coordinates_above_threshold()
        # print(f"Found {len(coordinates)} coordinates above threshold")  # Debugging line
        
        if coordinates.any():
            data = [self.analyze_region(loaded_image, coord, r) for coord in coordinates for r in self.radii]
            return pd.DataFrame(data)
        else:
            # print("No coordinates found above threshold. Returning empty DataFrame.")
            return pd.DataFrame()

    def analyze_region(self, image: np.ndarray, coord: Tuple[int, int], r: int) -> dict:
        x, y = coord[0:2] # added [0:2] to avoid error, as coord is a tuple of 3 elements
        a = ArrayRegion(array=image)
        region = a.extract_region(x_center=x, y_center=y, region_size=r)
        
        if region.size > 0:
            sum_excluding_center = np.sum(region) - region[r][r] if r < region.shape[0] else 0
            count_excluding_center = region.size - 1
            avg_intensity = sum_excluding_center / count_excluding_center if count_excluding_center > 0 else 0
            peak_intensity_estimate = region[r][r] - avg_intensity if r < region.shape[0] else 0
        else:
            
            # Add BC 
            
            print(f"No data in region for coordinates {coord} with radius {r}")
            return {}

        return {
            'coordinate': (x, y),
            'radius': r,
            'average_intensity': avg_intensity,
            'center_pixel_intensity': region[r][r] if r < region.shape[0] else 0,
            'peak_intensity_estimate': peak_intensity_estimate
        }

    def process_overlay_images(self, overlay_files: List[str]) -> pd.DataFrame:
        """Process a list of overlay image files."""
        all_data = [self.process_single_image(path) for path in overlay_files]
        return pd.concat(all_data, ignore_index=True)
    

    def main(self, inputs: List[str] | torch.Tensor):   
        if isinstance(inputs, list):
            # Example processing a single file (first file for demonstration)
            # single_image_data = self.process_single_image(image_path=overlay_files[0])
            # print("Single Image Data:")
            # print(single_image_data)
            
            # self.visualize_peaks(image_path=overlay_files[0], data=single_image_data)

            # Processing all overlay files  
            batch_data = self.process_overlay_images(overlay_files=inputs)
            print("Batch Overlay Files Data:")
            print(batch_data)
            
        elif isinstance(inputs, torch.Tensor):
            batch_data = self.process_tensor(tensors=inputs)
            print("Batch Overlay Files Data:")
            print(batch_data)
        
        return batch_data

    def visualize_peaks(self, input: str | torch.Tensor, data:pd.DataFrame):
        if isinstance(input, str):
            loaded_image, _ = load_h5(file_path=input)
        elif isinstance(input, torch.Tensor):
            loaded_image = input.cpu().numpy()
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Convert coordinates from strings to tuples
        data['coordinate'] = data['coordinate'].apply(lambda coord: coord if isinstance(coord, tuple) else eval(coord))

        x_coords = data['coordinate'].apply(lambda coord: coord[0])
        y_coords = data['coordinate'].apply(lambda coord: coord[1])
        z_coords = data['peak_intensity_estimate']

        # Plotting the original image at low opacity
        x_img, y_img = np.meshgrid(range(loaded_image.shape[1]), range(loaded_image.shape[0]))
        z_img = np.zeros(loaded_image.shape)

        ax.scatter(x_img, y_img, z_img, c='gray', alpha=0.1)  # Plot image at low opacity

        # Plotting the estimated peak intensities as scatter points
        scatter = ax.scatter(y_coords, x_coords, z_coords, c=z_coords, cmap='viridis', marker='o', depthshade=False)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Estimated Peak Intensity')

        # Adding a color bar to show the scale of peak intensities
        fig.colorbar(scatter, ax=ax, label='Peak Intensity Estimate')

        plt.show()

    

    
    # def visualize_peaks(self, image_path:str, data:pd.DataFrame):
    #     loaded_image, _ = load_h5(file_path=image_path)
        
    #     fig = plt.figure(figsize=(10, 7))
    #     ax = fig.add_subplot(111, projection='3d')

    #     # Convert coordinates from strings to tuples
    #     data['coordinate'] = data['coordinate'].apply(lambda coord: coord if isinstance(coord, tuple) else eval(coord))

    #     x_coords = data['coordinate'].apply(lambda coord: coord[0])
    #     y_coords = data['coordinate'].apply(lambda coord: coord[1])
    #     z_coords = data['peak_intensity_estimate']

    #     # Plotting the original image at low opacity
    #     x_img, y_img = np.meshgrid(range(loaded_image.shape[1]), range(loaded_image.shape[0]))
    #     z_img = np.zeros(loaded_image.shape)

    #     ax.scatter(x_img, y_img, z_img, c='gray', alpha=0.1)  # Plot image at low opacity

    #     # Plotting the estimated peak intensities as scatter points
    #     scatter = ax.scatter(y_coords, x_coords, z_coords, c=z_coords, cmap='viridis', marker='o', depthshade=False)

    #     ax.set_xlabel('X Label')
    #     ax.set_ylabel('Y Label')
    #     ax.set_zlabel('Estimated Peak Intensity')

    #     # Adding a color bar to show the scale of peak intensities
    #     fig.colorbar(scatter, ax=ax, label='Peak Intensity Estimate')

    #     plt.show()

    
