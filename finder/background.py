import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
from pathlib import Path
from finder.threshold import PeakThresholdProcessor
from finder.region import ArrayRegion
from finder.functions import load_h5

class BackgroundSubtraction:
    def __init__(self):
        self.cxfel_root = Path('../').resolve()
        self.images_dir = Path('../images').resolve()
        # self.cxfel_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # self.images_dir = self.walk()
        self.radii = [1, 2, 3, 4]
        self.loaded_image, self.image_path = load_h5(images_dir=self.images_dir)
        self.p = PeakThresholdProcessor(self.loaded_image, 1000)
        self.coordinates = self.p.get_coordinates_above_threshold()
        # self.peaks = self._find_peaks(use_1d=False)
        
    def walk(self):
        # returns path to cxfel/images
        start = self.cxfel_root
        for root, dirs, files in os.walk(start):
            if "images" in dirs:
                return os.path.join(root, "images")
        raise Exception("Could not find the 'images' directory starting from", start)
    
    # default background subtraction
    def coordinate_menu(self, r):
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
                self.process_region(coord=(x, y), r=r)
            else:
                print("Invalid index.")
        except ValueError:
            print("Invalid input. Enter a number or 'q' to quit.")
    
    def process_region(self, coord, r):
        (x, y), sum_ = coord, 0
        print(f"\nProcessing coordinate: ({x}, {y})")
        print(f"Radius: {r}")
        
        a = ArrayRegion(self.loaded_image)
        region = a.extract_region(x_center=x, y_center=y, region_size=r)
        neighborhood = a.extract_region(x_center=x, y_center=y, region_size=5)
    
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