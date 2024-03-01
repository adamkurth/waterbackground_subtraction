import os
import re
import shutil
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
        self.waterbackground_dir = self._walk_waterbackground()
        self.args = self._args()
        self.radii = [1, 2, 3, 4]
        self.loaded_image, self.image_path = self._load_h5(self.images_dir)
        self.p = PeakThresholdProcessor(self.loaded_image, 1000)
        self.coordinates = self.p._get_coordinates_above_threshold()
        # self.peaks = self._find_peaks(use_1d=False)
        self.high_low_stream_dir = self._walk_highlow()
        self.high_data, _, self.high_data_path = self._load_stream('test_high.stream')
        self.low_data, _, self.low_data_path = self._load_stream('test_low.stream')

    # path management
    def _walk(self):
        # returns images/ directory
        start = self.cxfel_root
        for root, dirs, files in os.walk(start):
            if "images" in dirs:
                return os.path.join(root, "images")
        raise Exception("Could not find the 'images' directory starting from", start)

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
    
    # default background subtraction
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

    # stream adaptation     
    @staticmethod
    def _walk_waterbackground():
        """Dynamically find the project root directory based on script location."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        while current_dir:
            if 'waterbackground_subtraction' in os.listdir(current_dir):
                return os.path.join(current_dir, 'waterbackground_subtraction')
            parent_dir = os.path.dirname(current_dir)
            if current_dir == parent_dir:  # If we've reached the root of the filesystem
                break
            current_dir = parent_dir
        raise FileNotFoundError("waterbackground_subtraction directory not found.")

    def _walk_highlow(self):
        """Find the high_low_stream directory starting from the project root directory."""
        for root, dirs, _ in os.walk(self.cxfel_root):
            if 'high_low_stream' in dirs:
                return os.path.join(root, 'high_low_stream')
        raise FileNotFoundError("high_low_stream directory not found.")
    
    def _load_stream(self, file):
        """Load and process data from a specified stream file."""
        path = os.path.join(self.high_low_stream_dir, file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{file} not found in high_low_stream directory.")
        
        print(f"\nLoading file: {file}")
        with open(path, 'r') as stream:
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
                                # data_columns['h'].append(int(elements[0]))
                                # data_columns['k'].append(int(elements[1]))
                                # data_columns['l'].append(int(elements[2]))
                                # data_columns['I'].append(float(elements[3]))
                                # data_columns['sigmaI'].append(float(elements[4]))
                                # data_columns['peak'].append(float(elements[5]))
                                # data_columns['background'].append(float(elements[6]))
                                # data_columns['fs'].append(float(elements[7]))
                                # data_columns['ss'].append(float(elements[8]))
                                # data_columns['panel'].append(str(elements[9]))
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
            
            print(f"File {file} loaded successfully.\n")
            return self.process_stream_data(data_columns, path)
  
    def process_stream_data(self, data_columns, path):
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
        return data_columns, intensity, path
    
    
    # overwrite implementation
    def _duplicate_before(self, file):
        """Create a copy of the file before overwriting."""
        overwrite_out_dir = os.path.join(self.waterbackground_dir, "overwrite_out")
        os.makedirs(overwrite_out_dir, exist_ok=True)
        filename = os.path.basename(file)
        new_file = os.path.join(overwrite_out_dir, f"{os.path.splitext(filename)[0]}_copy{os.path.splitext(filename)[1]}")
        shutil.copyfile(file, new_file)
        print(f"Duplicate created: {new_file}")
        return new_file
        
    def _compare(self, high, low, *columns):
        """Compare high/low data and return the compared data for quick debugging."""
        compare = {}
        for col in columns:
            if col in high and col in low:
                print(f'High: {high[col]} \nLow: {low[col]} \n')
                compare[col] = (high[col], low[col])
        return compare
    
    def _retrieve(self, data_columns, *columns):
        """Retrieve specified columns from data_columns."""
        return {col: data_columns[col] for col in columns if col in data_columns}
    
        
    def _overwrite(self, overwrite_data):
        """Overwrite the LOW data in the HIGH stream file."""
        file = self.high_data_path
        self._duplicate_before(file)
    
        overwritten = False
        print(f"\nStarting to overwrite data in file: {file}")
        
        with open(file, 'r') as f:
            lines = f.readlines()
            
        # track if any has been overwritten 
        overwritten = False
        
        # Open the file for writing
        with open(file, 'w') as f:
            for line in lines:
                if line.startswith("   h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel"):
                    # write the header line
                    f.write(line)
                    # avoid IndexError
                    expected_length = len(overwrite_data['h'])
                    if all(len(overwrite_data[key]) == expected_length for key in overwrite_data):
                        # write new data
                        for i in range(expected_length):
                            formatted_row = '{:>4} {:>4} {:>4} {:>9.3f} {:>12.3f} {:>12.3f} {:>12.3f} {:>6.2f} {:>6.2f} {:>6}\n'.format(
                                overwrite_data['h'][i],
                                overwrite_data['k'][i],
                                overwrite_data['l'][i],
                                overwrite_data['I'][i],
                                overwrite_data['sigmaI'][i],
                                overwrite_data['peak'][i],
                                overwrite_data['background'][i],
                                overwrite_data['fs'][i],
                                overwrite_data['ss'][i],
                                overwrite_data['panel'][i]
                            )
                            f.write(formatted_row)
                    overwritten = True
                else:
                    f.write(line) # unmodified
        # log
        if overwritten:
            print(f"Data overwritten successfully in file: {file}")
        else:
            print(f"No data was overwritten in file: {file}")
            
            
    # main
    def main(self):
        # no if __name__ == "main""

        # for r in self.radii:
        #     self._coordinate_menu(r)
        # self._display()
        # self._display_peaks_3d()

        high_data, high_intensity, high_path = self._load_stream('test_high.stream')
        low_data, low_intensity, low_path = self._load_stream('test_low.stream')
        self._overwrite(self.high_data)    
        


    # extra development 
    
    # peak detection using sklearn
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
    b = BackgroundSubtraction()
    b.main()


