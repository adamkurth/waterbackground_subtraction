import os
import re
import shutil
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import namedtuple

class Stream:
    def __init__(self):
        self.water_background_dir = self._walk_water_background()
        self.high_low_stream_dir = self._walk_highlow()
        self.high_data, self.high_intensity, _ = None, None, None
        self.low_data, self.low_intensity, _ = None, None, None

    @staticmethod
    def _walk_water_background():
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
    
    
def main(self):
    self.high_data, self.high_intensity, _ = self._load_stream('test_high.stream')
    self.low_data, self.low_intensity, _ = self._load_stream('test_low.stream')
    self._overwrite(self.high_data)
    
    
    