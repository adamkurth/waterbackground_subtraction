import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
import pandas as pd
from matplotlib.patches import Circle
from typing import Tuple, List, Union
from pathlib import Path
from scipy.interpolate import griddata
import random 
import torch
from .threshold import PeakThresholdProcessor
from .region import ArrayRegion
from .functions import load_h5, apply_waterbackground



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
        p = PeakThresholdProcessor(image=loaded_image, threshold_value=self.threshold)
        coordinates = p.get_coordinates_above_threshold()
        data = []
        
        for coord in coordinates:
            for r in self.radii:
                result = self.analyze_region(image=loaded_image, coord=coord, r=r)
                if result:
                    data.append(result)
                
        return pd.DataFrame(data)
    
    def analyze_region(self, image: np.ndarray, coord: Tuple[int, int], r: int) -> dict:
        x, y = coord
        half_size = r
        x_start, x_end = max(x - half_size, 0), min(x + half_size + 1, image.shape[0])
        y_start, y_end = max(y - half_size, 0), min(y + half_size + 1, image.shape[1])

        # Extract the region around the coordinate, ensuring we stay within image bounds
        region = image[x_start:x_end, y_start:y_end]

        # If the extracted region is smaller than the intended size, skip processing
        if region.shape[0] < 2*r + 1 or region.shape[1] < 2*r + 1:
            print(f"Region too small for coordinate {coord} with radius {r}")
            return {}

        # Calculate the sum and average intensity excluding the peak at the center
        center_value = region[half_size, half_size]
        sum_excluding_center = np.sum(region) - center_value
        count_excluding_center = region.size - 1
        avg_intensity = sum_excluding_center / count_excluding_center
        peak_intensity_estimate = center_value - avg_intensity

        return {
            'coordinate': (x, y),
            'radius': r,
            'average_intensity': avg_intensity,
            'center_pixel_intensity': center_value,
            'peak_intensity_estimate': peak_intensity_estimate
        }


    def process_overlay_images(self, overlay_files: List[str]) -> pd.DataFrame:
        """Process a list of overlay image files."""
        all_data = [self.process_single_image(path) for path in overlay_files]
        return pd.concat(all_data, ignore_index=True)
    
    """ demo below """
    def to_overlay(self, image_path: Path, background:Path, output_path: Path):
        """Demo function to show how to use the BackgroundSubtraction class."""
        peak_image, _ = load_h5(file_path=image_path)
        background, _ = load_h5(file_path=background)
        overlay_image = apply_waterbackground(peak_image=peak_image, background=background)

        with h5.File(output_path, 'w') as f:
            f.create_dataset('entry/data/data', data=overlay_image)

    def test_demo(self): 
        """Test function to show how to use the BackgroundSubtraction class."""
        images_directory : Path = Path('../../../../../images/test').resolve()
        background : Path = Path(images_directory, 'water-test.h5').resolve()
        
        # Collect all images in the directory
        image_paths = [str(f) for f in images_directory.glob('test_*.h5')]
        overlay_files = [str(f) for f in images_directory.glob('overlay_test_*_8keV.h5')]

        # Sort files to ensure proper pairing
        image_paths.sort(), overlay_files.sort() 

        # Combine the peak and overlay images with correct pairing
        tuple_files = [(x, y) for x,y in zip(image_paths, overlay_files)]
        print(tuple_files)

        # Randomly choose a tuple from the list
        
        choice_tuple = random.choice(tuple_files)
        peak_path, overlay_path = choice_tuple

        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.max_rows', None)

        # Assuming a method to process images and visualize coordinates
        dataframe = self.process_single_image(image_path=peak_path)
        print(dataframe)

        self.visualize_intensity(image_path=peak_path, df=dataframe)
        self.visualize_intensity_2d(image_path=peak_path, df=dataframe)
        self.visualize_intensity_3d(df=dataframe)
        self.visualize_intensity_dist(df=dataframe)

    def visualize_intensity(self, image_path: str, df: pd.DataFrame):
        # # Load the image from .h5 file
        img,_ = load_h5(image_path)

        # Adjust the contrast of the image by scaling the intensities
        img = img.astype(float)
        p90, p98 = np.percentile(img, (90, 98))
        img = (img - p90) / (p98 - p90)  # Stretching the contrast

        # Cap values at 1.0
        img[img > 1.0] = 1.0

        # Create a plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, cmap='gray', aspect='equal')

        # Logarithmic color normalization
        intensity = df['peak_intensity_estimate']
        log_norm = plt.Normalize(vmin=np.log10(intensity.min() + 1), vmax=np.log10(intensity.max() + 1))
        cmap = plt.cm.plasma  # Using a different colormap for better visibility

        # Draw circles for each peak
        for idx, row in df.iterrows():
            x, y = row['coordinate']
            radius = row['radius']*2 # Adjust the radius as needed
            color = cmap(log_norm(np.log10(row['peak_intensity_estimate'] + 1)))
            circle = Circle((y, x), radius, fill=True, color=color, alpha=0.7)  # Filled circles
            ax.add_patch(circle)

        plt.axis('off')

        # Create color bar
        scalar_mappable = plt.cm.ScalarMappable(norm=log_norm, cmap=cmap)
        scalar_mappable.set_array(intensity)
        cbar = plt.colorbar(scalar_mappable, ax=ax)
        cbar.set_label('Log10(Peak Intensity)')

        plt.show()

    def visualize_intensity_2d(self, image_path: str, df: pd.DataFrame):
        img,_ = load_h5(image_path)

        # Enhance the contrast of the image if needed
        img = img.astype(float) / img.max()

        # Create a plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, cmap='gray', aspect='equal')

        # Logarithmic color normalization
        intensity = np.log10(df['peak_intensity_estimate'] + 1)  # Avoid log(0)
        norm = plt.Normalize(vmin=intensity.min(), vmax=intensity.max())
        cmap = plt.cm.jet  # Use a high-contrast colormap

        # Draw circles for each peak
        for idx, row in df.iterrows():
            x, y = row['coordinate']
            radius = 5  # Increase the size of the marker for better visibility
            color = cmap(norm(np.log10(row['peak_intensity_estimate'] + 1)))
            circle = Circle((y, x), radius, fill=True, color=color, edgecolor='white', linewidth=1)
            ax.add_patch(circle)

        plt.axis('off')
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label('Log10(Peak Intensity)')

        plt.show()


    def visualize_intensity_3d(self, df: pd.DataFrame):
        # Create a 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Logarithmically scale the peak intensities for visualization
        # Adding 1 to the intensity before taking the log to avoid log(0) issues
        df['log_intensity'] = np.log10(df['peak_intensity_estimate'] + 1)

        # Set a colormap
        cmap = plt.get_cmap('viridis')

        # Plotting each coordinate with its logarithmically scaled peak intensity
        sc = ax.scatter(
            df['coordinate'].apply(lambda c: c[0]),  # X coordinates
            df['coordinate'].apply(lambda c: c[1]),  # Y coordinates
            df['log_intensity'],  # Log intensity as Z value
            c=df['log_intensity'],  # Color by log intensity
            cmap=cmap,
            marker='o',
            depthshade=True
        )

        # Set labels
        ax.set_xlabel('Detector X Coordinate')
        ax.set_ylabel('Detector Y Coordinate')
        ax.set_zlabel('Logarithmic Peak Intensity')

        # Adding a color bar to show intensity scale
        color_bar = plt.colorbar(sc, ax=ax)
        color_bar.set_label('Logarithmic Peak Intensity')

        # Show the plot
        plt.show()

    def visualize_intensity_dist(self, df: pd.DataFrame):
        # Remove any rows with non-positive intensity values
        df = df[df['peak_intensity_estimate'] > 0].copy()
        
        # Extract coordinates and intensities
        x = df['coordinate'].apply(lambda c: c[0]).astype(np.float64)
        y = df['coordinate'].apply(lambda c: c[1]).astype(np.float64)
        z = df['peak_intensity_estimate'] # actual
        positive_z = z[z > 0] # positive values only
        
        # Logarithmic scaling for coloring
        log_z = np.log10(positive_z)

        # Interpolation for the Z values on a grid
        xi = np.linspace(x.min(), x.max(), num=100)
        yi = np.linspace(y.min(), y.max(), num=100)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi, yi), method='linear')

        # Check for NaNs in the interpolated values and replace them with the minimum positive z value
        zi = np.nan_to_num(zi, nan=np.nanmin(positive_z))

        # Normalize log_z to match zi's shape using interpolation
        log_zi = griddata((x, y), log_z, (xi, yi), method='linear')
        log_zi = np.nan_to_num(log_zi, nan=np.nanmin(log_z))

        # Create a 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create the surface plot with actual Z values and color by logarithmic Z
        norm = plt.Normalize(log_z.min(), log_z.max())
        surf = ax.plot_surface(xi, yi, zi, facecolors=plt.cm.viridis(norm(log_zi)), alpha=0.7, edgecolor='none')

        # Set labels
        ax.set_xlabel('Detector X Coordinate')
        ax.set_ylabel('Detector Y Coordinate')
        ax.set_zlabel('Peak Intensity')
        
        # Adding a color bar to show intensity scale
        mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        mappable.set_array(log_zi)
        color_bar = plt.colorbar(mappable, ax=ax)
        color_bar.set_label('Logarithmic Peak Intensity')

        # Show the plot
        plt.show()




    def main(self, inputs):   
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

    def visualize_peaks(self, input, data:pd.DataFrame):
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
