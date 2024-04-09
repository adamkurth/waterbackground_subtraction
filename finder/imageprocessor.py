import os
from pathlib import Path
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from finder.threshold import PeakThresholdProcessor


class ImageProcessor: 
    def __init__(self, image, threshold):
        self.image = image
        self.dim = image.shape
        self.coordinates = self.find_peaks()
        self.p = PeakThresholdProcessor(self.image, threshold_value=threshold)
        self.threshold = self.p.threshold_value

    def find_peaks(self, min_distance=10, threshold_abs=250):
        coordinates = peak_local_max(self.image, min_distance=min_distance, threshold_abs=threshold_abs)
        return coordinates
    
    def _display_peaks_2d(self, img_threshold=0.005):
        # for visualization exclusion diameter is okay
        image, peaks = self.loaded_image, self.peaks
        plt.figure(figsize=(10, 10))
        masked_image = np.ma.masked_less_equal(image, img_threshold) # mask values less than threshold (for loading speed)
        plt.imshow(masked_image, cmap='viridis')
        
        # filter peaks by threshold
        flt_peaks = [coord for coord in peaks if image[coord] > img_threshold]
        for x,y in flt_peaks: 
            plt.scatter(y, x, color='r', s=50, marker='x') 
            
        plt.title('Image with Detected Peaks')            
        plt.xlabel('X-axis (ss)')
        plt.ylabel('Y-axis (fs)')
        plt.show()

    def visualize_peaks(self):
        coordinates = self.coordinates
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image, cmap='inferno')
        plt.scatter(coordinates[:, 1], coordinates[:, 0], color='cyan', s=100, edgecolor='white', marker='o', label='Detected Peaks')
        plt.title("Detected Peaks in Image")
        plt.colorbar(label='Intensity')
        plt.legend()
        plt.axis('on')
        plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
        plt.show()

    def visualize_image_3d(self, coordinates, threshold, img_threshold=0.005):
        image = self.image
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Prepare data
        x = np.arange(0, image.shape[1])
        y = np.arange(0, image.shape[0])
        X, Y = np.meshgrid(x, y)
        Z = image

        mask = Z > img_threshold
        Xmasked, Ymasked, Zmasked = X[mask], Y[mask], Z[mask]

        print(f"Number of points above img_threshold: {np.sum(mask)}")
        print(f"Number of points in the image: {image.size}")
        print(f"Threshold value for highlighting: {threshold}")

        # Plotting all pixels as a scatter plot
        pixel_scatter = ax.scatter(Xmasked.flatten(), Ymasked.flatten(), Zmasked.flatten(),
                                c=Zmasked.flatten(), cmap='viridis', alpha=0.7, marker='.')

        # Highlighting coordinates above the threshold
        flt_peaks = [(y, x) for x, y in coordinates if image[y, x] > threshold]
        print(f"Number of peaks above threshold: {len(flt_peaks)}")

        if flt_peaks:
            p_x, p_y = zip(*flt_peaks)
            p_z = [image[y, x] for y, x in flt_peaks]
            ax.scatter(p_y, p_x, p_z, color='red', s=100, marker='^', label='Highlighted Peaks', edgecolor='white')

        # Colorbar and labels
        fig.colorbar(pixel_scatter, shrink=0.5, aspect=5, label='Intensity')
        ax.set_title('3D Scatter Plot of Image Intensity with Highlighted Peaks')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Intensity')
        plt.legend()
        plt.show()
            