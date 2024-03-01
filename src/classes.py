import os
from pathlib import Path
import re
import shutil
import h5py as h5
import argparse  
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import peak_local_max
from scipy.signal import find_peaks

class DataHandler:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.waterbackground_dir = self.find_dir("waterbackground_subtraction")
        self.high_low_stream_dir = self.find_dir("high_low_stream")
        
    def find_dir(self, dir_name):
        for path in self.base_path.rglob('*'):
            if path.name == dir_name and path.is_dir():
                return path
        raise FileNotFoundError(f"{dir_name} directory not found.")   

    def load_h5_image(self, image_name):
        image_path = self.waterbackground_dir / "images" / image_name
        with h5.File(image_path, 'r') as file:
            data = file['entry/data/data'][:]
        return data
    
    def load_stream_data(self, stream_name):
        stream_path = self.high_low_stream_dir / stream_name
        print(f"\nLoading file: {stream_name}")
        with open(stream_path, 'r') as stream:
            data_columns = {'h': [], 'k': [], 'l': [], 'I': [], 'sigmaI': [], 'peak': [], 'background': [], 'fs': [], 'ss': [], 'panel': []}
            x, y, z = [], [], []
            reading_peaks = False
            reading_geometry = False
            reading_chunks = True 
            for line in stream:
                if reading_chunks:
                    if line.startswith('End of peak list'):
                        reading_peaks = False
                    elif line.startswith("   h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel"):
                        reading_peaks = True
                        continue
                    elif reading_peaks:
                        try:
                            elements = line.split()
                            for key, element in zip(data_columns.keys(), elements):
                                data_columns[key].append(float(element) if key not in {'h', 'k', 'l', 'panel'} else element)
                        except:
                            pass
                elif line.startswith('----- End geometry file -----'):
                    reading_geometry = False
                elif reading_geometry:   
                    try:
                        par, val = line.split('=')
                        if par.split('/')[-1].strip() == 'max_fs' and int(val) > max_fs:
                            max_fs = int(val)
                        elif par.split('/')[-1].strip() == 'max_ss' and int(val) > max_ss:
                            max_ss = int(val)
                    except ValueError:
                        pass
                elif line.startswith('----- Begin geometry file -----'):
                    reading_geometry = True
                elif line.startswith('----- Begin chunk -----'):
                    reading_chunks = True   
            print(f"File {stream_name} loaded successfully.\n")
            return self.process_stream_data(data_columns, stream_path)
          
    def process_stream_data(self, data_columns, stream_path):
        """Process loaded stream data to calculate intensity matrix."""
        x, y, z = data_columns['fs'], data_columns['ss'], data_columns['I']
        if not x:
            raise ValueError("Stream file contains no data.")
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        intensity = np.zeros((int(xmax - xmin + 1), int(ymax - ymin + 1)))
        for X, Y, Z in zip(x, y, z):
            row, col = int(X - xmin), int(Y - ymin)
            intensity[row, col] = Z
        return data_columns, intensity, stream_path
   
    
class ImageProcessor: 
    def __init__(self, image):
        self.image = image
        self.dim = image.shape
        self.coordinates = self.find_peaks()

    def find_peaks(self, min_distance=10, threshold_abs=250):
        coordinates = peak_local_max(self.image, min_distance=min_distance, threshold_abs=threshold_abs)
        return coordinates

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

    def visualize_image_3d(self, img_threshold=0.005):
        image, coordinates = self.image, self.coordinates
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Grid setup
        x, y = np.arange(0, image.shape[1], 1), np.arange(0, image.shape[0], 1)
        X, Y = np.meshgrid(x, y)
        Z = np.ma.masked_less_equal(image, img_threshold)

        # Surface plot with enhanced visual settings
        surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=True, alpha=0.7)
        fig.colorbar(surf, shrink=0.5, aspect=5, label='Intensity')
        
        # Filtering peaks above the threshold
        flt_peaks = [(x, y) for x, y in coordinates if image[x, y] > img_threshold]
        if flt_peaks:
            p_x, p_y = zip(*[(y, x) for x, y in flt_peaks])  # Note the inversion of x and y for plotting
            p_z = np.array([image[x, y] for x, y in flt_peaks])
            ax.scatter(p_x, p_y, p_z, color='lime', s=100, marker='^', label='Peaks')
        else: 
            print(f"No peaks above the threshold {img_threshold} were found.")

        ax.set_title('3D View of Image')
        ax.set_xlabel('X-axis (ss)')
        ax.set_ylabel('Y-axis (fs)')
        ax.set_zlabel('Intensity')

        plt.legend()
        plt.show()                
          
          
