import os
from pathlib import Path
import h5py as h5
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
                
class PeakThresholdProcessor: 
    def __init__(self, image, threshold_value=0):
        self.image = image
        self.threshold_value = threshold_value
    
    def _set_threshold_value(self, new):
        self.threshold_value = new
    
    def _get_coordinates_above_threshold(self):  
        coordinates = np.argwhere(self.image > self.threshold_value)
        return coordinates
    
    def _get_local_maxima(self):
        image_1d = self.image.flatten()
        peaks, _ = find_peaks(image_1d, height=self.threshold_value)
        coordinates = [self.flat_to_2d(idx) for idx in peaks]
        return coordinates
        
    def _flat_to_2d(self, index):
        shape = self.image.shape
        rows, cols = shape
        return (index // cols, index % cols) 
    

