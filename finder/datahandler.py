import os
import shutil
from pathlib import Path
import h5py as h5
import numpy as np
from .functions import load_h5, find_dir

class DataHandler:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.waterbackground_dir = self.find_dir(base_path=base_path, dir_name="waterbackground_subtraction")
        self.high_low_stream_dir = self.find_dir(base_path=base_path, dir_name="high_low_stream")
        
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
    
    # overwrite implementation
    def duplicate_before(self, file):
        """Create a copy of the file before overwriting."""
        overwrite_out_dir = os.path.join(self.waterbackground_dir, "overwrite_out")
        os.makedirs(overwrite_out_dir, exist_ok=True)
        filename = os.path.basename(file)
        new_file = os.path.join(overwrite_out_dir, f"{os.path.splitext(filename)[0]}_copy{os.path.splitext(filename)[1]}")
        shutil.copyfile(file, new_file)
        print(f"Duplicate created: {new_file}")
        return new_file

    def compare_debug(self, high, low, *columns):
        """Compare high/low data and return the compared data for quick debugging."""
        compare = {}
        for col in columns:
            if col in high and col in low:
                print(f'High: {high[col]} \nLow: {low[col]} \n')
                compare[col] = (high[col], low[col])
        return compare
    
    def retrieve_debug(self, data_columns, *columns):
        """Retrieve specified columns from data_columns."""
        return {col: data_columns[col] for col in columns if col in data_columns}

    def overwrite(self, overwrite_data):
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
   
   