import os
import re
import h5py as h5
import argparse  
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import peak_local_max
from scipy.signal import find_peaks, peak_prominences, peak_widths
from skimage import filters
from skimage.filters import median
from skimage.morphology import disk
from skimage.util import img_as_float
from skimage.exposure import rescale_intensity
from collections import namedtuple

class BackgroundSubtraction:
    def __init__(self):
        self.cxfel_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.images_dir = self._walk()
        self.args = self._args()
        self.radii = [1, 2, 3, 4]
        self.loaded_image, self.image_path = self._load_h5(self.images_dir)
        # self.loaded_image, self.image_path = self._load_test()
        self.p = PeakThresholdProcessor(self.loaded_image, 1000)
        self.coordinates = self.p._get_coordinates_above_threshold()
        self.peaks = self._find_peaks(use_1d=False)
        
    def _walk(self):
        # returns images/ directory
        start = self.cxfel_root
        for root, dirs, files in os.walk(start):
            if "images" in dirs:
                return os.path.join(root, "images")
        raise Exception("Could not find the 'images' directory starting from", start)
    
    def _load_test(self):
        image_path = os.path.join(os.getcwd(), "9_18_23_high_intensity_3e8keV.h5")
        try:
            with h5.File(image_path, 'r') as file:
                data = file['entry/data/data'][:]
            return data, image_path
        except Exception as e:
            raise OSError(f"Failed to read {image_path}: {e}")
        
    def _load_h5(self, image_dir):
        # choose images with "processed" in the name for better visuals
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.h5') and 'processed' in f]
        if not image_files:
            raise FileNotFoundError("No processed images found in the directory.")
        # choose a random image to load
        random_image = np.random.choice(image_files)
        image_path = os.path.join(image_dir, random_image)
        print("Loading image:", random_image)
        try:
            with h5.File(image_path, 'r') as file:
                data = file['entry/data/data'][:]
            return data, image_path
        except Exception as e:
            raise OSError(f"Failed to read {image_path}: {e}")
    
    def _coordinate_menu(self, r):
        print(f"\nCoordinates above given threshold: with radius: {r}")
        coordinates = self.coordinates
        for i, (x, y) in enumerate(coordinates):
            print(f"{i+1}. ({x}, {y})")
        
        choice = input("\nWhich coordinate do you want to process? (or 'q' to quit)\n")
        if choice.lower() == 'q':
            print("Quitting...")
            return
        
        try:
            index = int(choice)-1 
            if 0 <= index < len(coordinates):
                x, y = coordinates[index]
                self._process_region(coord=(x, y), r=r)
            else:
                print("Invalid index.")
        except ValueError:
            print("Invalid input. Enter a number or 'q' to quit.")
    
    def _process_region(self, coord, r):
        (x, y), sum_ = coord, 0
        print(f"\nProcessing coordinate: ({x}, {y})")
        print(f"Radius: {r}")
        
        a = ArrayRegion(self.loaded_image)
        region = a._extract_region(x_center=x, y_center=y, region_size=r)
        neighborhood = a._extract_region(x_center=x, y_center=y, region_size=5)
    
        np.set_printoptions(precision=4, suppress=True)
        print('Neighborhood:')
        print(neighborhood)
        print(f"\nRegion with radius {r} extracted for coordinate ({x}, {y}):\n")
        print(region)
        print("\n")
        
        # calc average intensity excluding the center pixel
        for i in range(len(region)):
            for j in range(len(region[i])):
                if i != r or j != r:  # Skip the center element
                    element = region[i][j]
                    print(f"Processing element at ({i}, {j}): {element}")
                    sum_ += element
        count = len(region) * len(region[0]) - 1  # Exclude the center pixel
        avg = sum_ / count if count > 0 else 0
        print("\n\n")
        print(f"Average intensity of region: {avg}")
        print(f"Center pixel intensity: {region[r][r]}")
        print(f"Peak intensity Estimate: {region[r][r] - avg}")
        print("\n")

    def _display(self, img_threshold=0.05):
        y_vals, x_vals = np.where(self.loaded_image > img_threshold)
        z_vals = self.loaded_image[y_vals, x_vals] 
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='hot', marker='o')

        # Highlight coordinates above the set threshold in the class
        for x, y in self.coordinates:
            if self.loaded_image[y, x] > img_threshold:
                ax.scatter(x, y, self.loaded_image[y, x], color='blue', s=100, edgecolor='black', marker='^', label='Highlighted')

        # Enhancements for better visualization
        ax.set_xlabel('X Coordinate (ss)')
        ax.set_ylabel('Y Coordinate (fs)')
        ax.set_zlabel('Intensity (I)')
        ax.set_title('Scatter Plot of Intensity Values Above 0.05')
        plt.colorbar(scatter, label='Intensity')
        
        # Only add legend for highlighted points once
        handles, labels = plt.gca().get_legend_handles_labels()
        if 'Highlighted' in labels:
            ax.legend()

        plt.show()

    def _display_peaks_3d(self, img_threshold=0.005):
        image, peaks = self.loaded_image, self.peaks
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # grid 
        x, y = np.arange(0, image.shape[1], 1), np.arange(0, image.shape[0], 1)
        X, Y = np.meshgrid(x, y)
        Z = np.ma.masked_less_equal(image, img_threshold)
        # plot surface/image data 
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False, alpha=0.6)
        fig.colorbar(surf, shrink=0.5, aspect=5, label='Intensity')
        flt_peaks = [coord for coord in peaks if image[coord] > img_threshold]
        if flt_peaks:
            p_x, p_y = zip(*flt_peaks)
            p_z = np.array([image[px, py] for px, py in flt_peaks])
            ax.scatter(p_y, p_x, p_z, color='r', s=50, marker='x', label='Peaks')
        else: 
            print(f"No peaks above the threshold {self.args.threshold_value} were found.")

        # labels 
        ax.set_title('3D View of Image with Detected Peaks')
        ax.set_xlabel('X-axis (ss)')
        ax.set_ylabel('Y-axis (fs)')
        ax.set_zlabel('Intensity')
        fig.colorbar(surf, shrink=0.5, aspect=5, label='Intensity')

        plt.legend()
        plt.show()
    
    
    def main(self):
        b = BackgroundSubtraction()
        for r in b.radii:
            b._coordinate_menu(r)
        # b._display()
        b._display_peaks_3d()
        
    
    @staticmethod
    def _args():
        parser = argparse.ArgumentParser(description='Apply a Gaussian mask to an HDF5 image and detect peaks with noise reduction')
        # Noise reduction parameter
        parser.add_argument('--median_filter_size', type=int, default=3, help='Size of the median filter for noise reduction', required=False)
        # Peak detection parameters
        parser.add_argument('--min_distance', type=int, default=10, help='Minimum number of pixels separating peaks', required=False)
        parser.add_argument('--prominence', type=float, default=1.0, help='Required prominence of peaks', required=False)
        parser.add_argument('--width', type=float, default=5.0, help='Required width of peaks', required=False)
        parser.add_argument('--min_prominence', type=float, default=0.1, help='Minimum prominence to consider a peak', required=False)
        parser.add_argument('--min_width', type=float, default=1.0, help='Minimum width to consider a peak', required=False)
        parser.add_argument('--threshold_value', type=float, default=500, help='Threshold value for peak detection', required=False)
        # Region of interest parameters for peak analysis
        parser.add_argument('--region_size', type=int, default=9, help='Size of the region to extract around each peak for analysis', required=False)
        return parser.parse_args()
    
    def _find_peaks(self, use_1d=False):
        """
        This function processes the loaded image to find and refine peaks.
        It first reduces noise using a median filter, then applies a Gaussian mask.
        After initial peak detection, it refines the peaks based on prominence and width criteria.
        """
        # assuming: self.loaded_image is the image to be processed
        # Noise reduction
        denoised_image = median(self.loaded_image, disk(self.args.median_filter_size)) # disk(3) is a 3x3 circular mask
        # Gaussian mask application
        # masked_image = self._apply(denoised_image)
        # Initial peak detection
        #   coordinates output from peak_local_max (not necissarily peaks)
        coordinates = peak_local_max(denoised_image, min_distance=self.args.min_distance) 
        # Peak refinement
        refined_peaks = self._refine_peaks(denoised_image, coordinates, use_1d)
        return refined_peaks

    def _refine_peaks(self, image, coordinates, use_1d=False):
        """
        Unified function to refine detected peaks using either 1D or 2D criteria.
        Extracts a region around each peak and analyzes it to determine the true peaks.
        """
        if use_1d:
            return self._refine_peaks_1d(image)
        else:
            return self._refine_peaks_2d(image, coordinates)
    
    def _refine_peaks_1d(self, image, axis=0):
        """
        Applies 1D peak refinement to each row or column of the image.
        axis=0 means each column is treated as a separate 1D signal; axis=1 means each row.
        """
        refined_peaks = []
        num_rows, num_columns = image.shape
        for index in range(num_columns if axis == 0 else num_rows):
            # Extract a row or column based on the specified axis
            signal = image[:, index] if axis == 0 else image[index, :]
            
            # Find peaks in this 1D signal
            peaks, _ = find_peaks(signal, prominence=self.args.prominence, width=self.args.width)
            
            # Store refined peaks with their original coordinates
            for peak in peaks:
                if axis == 0:
                    refined_peaks.append((peak, index))  # For columns
                else:
                    refined_peaks.append((index, peak))  # For rows
        return refined_peaks

    def _refine_peaks_2d(self, image, coordinates):
        """
        Refines detected peaks in a 2D image based on custom criteria.
        """
        exclusion_radius = 10  # pixels (avoid beam stop)
        x_center, y_center = image.shape[0] // 2, image.shape[1] // 2
        
        refined_peaks = []
        threshold = self.args.threshold_value
        for x, y in coordinates:
            dist_from_beamstop = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
            
            # extract small region around peak and apply custom criterion
            if dist_from_beamstop > exclusion_radius:
                region = image[max(0, x-10):x+10, max(0, y-10):y+10]
                
                # check if peak is significantly brighter than median of its surrounding
                if image[x, y] > np.median(region) + threshold:
                    refined_peaks.append((x, y))
            return refined_peaks



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
    
class ArrayRegion:
    def __init__(self, array):
        self.array = array
        self.x_center = 0
        self.y_center = 0
        self.region_size = 9 
    
    def _set_peak_coordinate(self, x, y):
        self.x_center = x
        self.y_center = y
    
    def _set_region_size(self, size):
        #limit that is printable in terminal
        self.region_size = size
        max_printable_region = min(self.array.shape[0], self.array.shape[1]) //2
        self.region_size = min(size, max_printable_region)
    
    def _get_region(self):
        x_range = slice(self.x_center - self.region_size, self.x_center + self.region_size+1)
        y_range = slice(self.y_center - self.region_size, self.y_center + self.region_size+1)
        region = self.array[x_range, y_range]
        return region

    def _extract_region(self, x_center, y_center, region_size):
        self._set_peak_coordinate(x_center, y_center)
        self._set_region_size(region_size)
        region = self._get_region()
        # Set print options for better readability
        np.set_printoptions(precision=8, suppress=True, linewidth=120, edgeitems=7)
        return region
    
# if __name__ == "__main__":
#     b = BackgroundSubtraction()
#     b.main()



