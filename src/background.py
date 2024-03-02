import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks


class BackgroundSubtraction:
    def __init__(self):
        self.cxfel_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.images_dir = self._walk()
        self.radii = [1, 2, 3, 4]
        self.loaded_image, self.image_path = self._load_h5(self.images_dir)
        self.p = PeakThresholdProcessor(self.loaded_image, 1000)
        self.coordinates = self.p._get_coordinates_above_threshold()
        # self.peaks = self._find_peaks(use_1d=False)
        
    def _walk(self):
        # returns path to cxfel/images
        start = self.cxfel_root
        for root, dirs, files in os.walk(start):
            if "images" in dirs:
                return os.path.join(root, "images")
        raise Exception("Could not find the 'images' directory starting from", start)

    def _load_h5(self, image_dir):
        # images with "processed" for background demonstration
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.h5') and 'processed' in f]
        if not image_files:
            raise FileNotFoundError("No processed images found in the directory.")
        random_image = np.random.choice(image_files) # random image
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


class ArrayRegion:
    def __init__(self, array):
        self.array = array
        self.x_center = array.shape[0] // 2
        self.y_center = array.shape[1] // 2
        self.region_size = 9 
    
    def _set_peak_coordinate(self, x, y):
        self.x_center = x
        self.y_center = y

    def _set_region_size_terminal(self, size):
        #limit that is printable in terminal
        self.region_size = size
        max_printable_region = min(self.array.shape[0], self.array.shape[1]) // 2
        self.region_size = min(size, max_printable_region)
    
    def _set_region_size(self, size):
        self.region_size = size
        
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
                
    def get_exclusion_mask(self):
        mask = np.ones(self.array.shape, dtype=bool)
        x_min = max(0, self.x_center - self.region_size)
        x_max = min(self.array.shape[1], self.x_center + self.region_size + 1)
        y_min = max(0, self.y_center - self.region_size)
        y_max = min(self.array.shape[0], self.y_center + self.region_size + 1)
        mask[y_min:y_max, x_min:x_max] = False
        return mask
                
                
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

# display functions
def display(image, coorinates, img_threshold=0.05):
    mask = image > img_threshold
    y_vals, x_vals = np.nonzero(mask)
    z_vals = image[mask]
    
    # Setup the figure and axis for 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot of all points above img_threshold
    scatter = ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='hot', marker='o')
    
    # Mask for highlighted points
    highlight_mask = np.zeros_like(image, dtype=bool)
    highlight_mask[coorinates[:, 1], coorinates[:, 0]] = True
    highlight_mask &= mask
    
    # Extract highlighted points
    y_highlight, x_highlight = np.nonzero(highlight_mask)
    z_highlight = image[highlight_mask]
    
    # Scatter plot for highlighted points
    if z_highlight.size > 0:
        ax.scatter(x_highlight, y_highlight, z_highlight, color='blue', s=100, edgecolor='black', marker='^', label='Highlighted')
    
    # Set labels and title
    ax.set_xlabel('X Coordinate (ss)')
    ax.set_ylabel('Y Coordinate (fs)')
    ax.set_zlabel('Intensity (I)')
    ax.set_title(f'Scatter Plot of Intensity Values Above {img_threshold}')
    
    # Colorbar and legend
    plt.colorbar(scatter, label='Intensity')
    ax.legend()
    
    # Show plot
    plt.show()

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
    region_handler._set_peak_coordinate(center_x, center_y)
    region_handler._set_region_size(8)
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
    
if __name__ == "__main__":
    b = BackgroundSubtraction()
    # for r in b.radii:
        # b._coordinate_menu(r)
    # b._display()
    display_peaks_3d(b.loaded_image, b.coordinates, b.p.threshold_value)
    display_peaks_3d_beamstop(b.loaded_image, b.p.threshold_value)
    