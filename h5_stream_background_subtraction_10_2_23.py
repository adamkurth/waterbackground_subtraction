#!/usr/bin/env python
import os
import sys
import numpy as np
import h5py as h5
from os.path import basename, splitext
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import overwrite_10_2_23 as overwrite

# #argument runnig script through term
# if sys.argv[1] == '-':
#     stream = sys.stdin
# else:
#     stream = open(sys.argv[1], 'r')

################################    
def load_file_h5():
    global filename
    filename =  "test_manipulate2HDF5.h5"
    #if filename is not within working directory
    if not os.path.exists(filename):
        print("File not found within working directory.")
        return
    try:
        with h5.File(filename, "r") as f: 
            print("\nLoaded file successfully.", filename)
    except Exception as e:
        print("\nAn error has occurred:", str(e))
 
        
class PeakThresholdProcessor: 
    #self method
    def __init__(self, image_array, threshold_value=0):
        self.image_array = image_array
        self.threshold_value = threshold_value
    #setter for threshold value
    def set_threshold_value(self, new_threshold_value):
        self.threshold_value = new_threshold_value
    #getter for for above threshold
    def get_coordinates_above_threshold(self):  
        coordinates = np.argwhere(self.image_array > self.threshold_value)
        return coordinates
 
def extract_region(image_array, region_size, x_center, y_center):
    extract = ArrayRegion(image_array)
    extract.set_peak_coordinate(x_center,y_center)
    extract.set_region_size(region_size)
    np.set_printoptions(floatmode='fixed', precision=10)
    np.set_printoptions(edgeitems=3, suppress=True, linewidth=200)
    region = extract.get_region()
    return region      
    
class ArrayRegion:
    def __init__(self, array):
        self.array = array
        self.x_center = 0
        self.y_center = 0
        self.region_size = 0
    def set_peak_coordinate(self, x, y):
            self.x_center = x
            self.y_center = y

    def set_region_size(self, size):
        #limit that is printable in terminal
        self.region_size = size
        max_printable_region = min(self.array.shape[0], self.array.shape[1]) //2
        self.region_size = min(size, max_printable_region)
    def get_region(self):
        x_range = slice(self.x_center - self.region_size, self.x_center + self.region_size+1)
        y_range = slice(self.y_center - self.region_size, self.y_center + self.region_size+1)
        region = self.array[x_range, y_range]
        return region

def coordinate_menu(image_array, threshold_value, coordinates, radius): 
    print("\nCoordinates above given threshold:", threshold_value, 'with radius: ', radius)
    for i, (x, y) in enumerate(coordinates):
        print(f"{i + 1}. ({x}, {y})")
        
    while True:
        choice = input("\nWhich coordinate do you want to process? (or 'q' to quit)\n")
        if choice == "q":
            print("Exiting")
            break
        
        try: 
            count = int(choice)-1
            if 0 <= count < len(coordinates):
                x,y = coordinates[count]
                print(f"\nProcessing - ({x}, {y})")
                print('Printing 9x9 two-dimensional array\n')
                
                #creates visualization if the array, of chosen peak
                print(x,y)
                display_region = extract_region(image_array, region_size=4, x_center=x, y_center=y)
                
                print('DISPLAY REGION \n', display_region, '\n')
                
                #segment is the area with the given radius that's passed through the function.
                segment = extract_region(image_array, region_size=radius, x_center=x, y_center=y)
                print ('SEGMENT \n', segment, '\n')
                
                # COMMENTED OUT FOR NOW, EASE OF TESTING

                #returns boolean array of traversed values.
                bool_square = np.zeros_like(segment, dtype=bool)
                print('BOOLEAN: before traversing.', '\n', bool_square, '\n') 
            
                ######start 3 ring integration
                values_array = extract_region(image_array, region_size=radius, x_center=x, y_center=y)
                
                #traverses through (i = row) , (j = column)         

                global avg_values, intensity_peak
                total_sum = 0; skipped_point = None; count = 0; intensity_peak = 0
                #traverses through (i = row) , (j = column)
                for col_index in range(values_array.shape[0]):
                    for row_index in range(values_array.shape[1]):
                        if values_array[row_index, col_index] >= 0:
                            count += 1
                            bool_square[row_index, col_index] = True
                            if row_index == radius and col_index == radius:
                                skipped_point = (row_index, col_index)  
                                intensity_peak = values_array[row_index, col_index]
                                print(f'Peak point to be skipped: ({row_index}, {col_index}) ', values_array[radius,radius])
                            elif abs(row_index - radius) <= 1 and abs(col_index - radius) <=1:
                                print(f'Passed (row, col) ({row_index}, {col_index})', values_array[row_index,col_index])
                                pass
                            else:
                                print(f'(row,col) ({row_index}, {col_index}) with a value of ', values_array[row_index, col_index])
                                total_sum += values_array[row_index, col_index]
                                
                print('\n######################')
                print(bool_square)
                print('Number of traversed cells', count)
                print('Peak point to be skipped:', skipped_point)
                print('Total sum:',total_sum)
                if count > 0:
                    avg_values = total_sum / count
                else: 
                    avg_values = "Could not divide by 0."
                print('Average surrounding peak:',avg_values)
                print('Peak point:', intensity_peak)
                return avg_values,intensity_peak
                break
            else: 
                print("Invalid coordinate idex.")
        except ValueError:
            print("Invalid input. Enter a number of 'q' to quit.")

################################## 

def load_stream(stream_path):
    global stream_coord
    global result_x, result_y, result_z #for building intensity array
    
    stream_name = os.path.basename(stream_path)
    # directory_path = f'{stream_path}'
    full_path = os.path.join(stream_path)
    
    try:
        
        stream = open(full_path, 'r') 
        print("\nLoaded file successfully.", stream_name, '\n')
    except Exception as e: 
        print("\nAn error has occurred:", str(e),'\n')
    
    reading_peaks = False
    reading_geometry = False
    reading_chunks = True 
    global data_columns
    data_columns = {
        'h':[], 'k':[], 'l':[],
        'I':[], 'sigmaI':[], 'peak':[], 'background':[],
        'fs':[],'ss':[], 'panel':[]
    }
    
    for line in stream:
        if reading_chunks:
           if line.startswith('End of peak list'):
               reading_peaks = False
           elif line.startswith("   h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel"):
               reading_peaks = True
           elif reading_peaks:
                try:
                    elements = line.split()
                    data_columns['h'].append(int(elements[0]))
                    data_columns['k'].append(int(elements[1]))
                    data_columns['l'].append(int(elements[2]))
                    data_columns['I'].append(float(elements[3]))
                    data_columns['sigmaI'].append(float(elements[4]))
                    data_columns['peak'].append(float(elements[5]))
                    data_columns['background'].append(float(elements[6]))
                    data_columns['fs'].append(float(elements[7]))
                    data_columns['ss'].append(float(elements[8]))
                    data_columns['panel'].append(str(elements[9]))
                    
                    # result_x.append(fs)
                    # result_y.append(ss)
                    # result_z.append(intensity)
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
    result_x = data_columns['fs']; result_y = data_columns['ss']; result_z = data_columns['I']
    return data_columns, result_x, result_y, result_z
    
def build_coord_intensity_array():
    coordinates_and_intensities = data_columns['fs','ss','I']
    return coordinates_and_intensities

# def create_scatter(x, y, z, highlight_x=None, highlight_y=None):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     # data_columns = build_coord_intensity_array()
#     scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], z, c=z, cmap='viridis', marker='o')

#     highlight_z = intensity_array[highlight_x,highlight_y]
#     print("Intensity value", highlight_z, "\n")
#     # Highlight the specific point if provided
#     if highlight_x is not None and highlight_y is not None:
#         ax.scatter([highlight_x], [highlight_y], [highlight_z], c='red', marker='x', s=100, label='Highlighted Point')

#     cbar = plt.colorbar(scatter)
#     cbar.set_label('Intensity')
    
#     # Set labels and title
#     ax.set_xlabel('X Coordinate')
#     ax.set_ylabel('Y Coordinate')
#     ax.set_zlabel('Intensity')
#     plt.title('3D Scatter Plot of (X, Y, Intensity)')
#     plt.show()
#     return None

def read_hdf5(filename, location):
    try:
        with h5.File(filename, "r") as f:
            if '/data/data/' + location in f: 
                dset = f['/data/data/'+location][()]
                global image_array
                image_array = np.array(dset)
                image_array_size = image_array.shape
                print("Shape of the array:", image_array_size)
                print(image_array)
                f.close()
                return image_array
            else:
                print("Dataset not found in location", location)
                return None
    except FileNotFoundError as e:
        print("Error HDF5 file not found.", e)
    except Exception as e:
        print("An error has occured.", e)
   
def main(stream_path):
    # read the stream
    load_stream(stream_path)
    xmin, xmax = np.min(result_x), np.max(result_x)
    ymin, ymax = np.min(result_y), np.max(result_y)

    num_rows = int(xmax-xmin+1)
    num_cols = int(ymax-ymin+1)
    print(num_rows, num_cols)
    intensity_array = np.zeros((num_rows,num_cols))

    for x,y,z in zip(result_x,result_y,result_z):
        row = int(x - xmin)
        col = int(y - ymin)
        intensity_array[row,col] = z
        
    #do not need to constantly write the same intensity values to the same array.
    # write_hdf5(filename=stream_name, data=intensity_array)

    location = "intensity_array"
    # read_hdf5(filename=stream_name, location=location)
    threshold = PeakThresholdProcessor(intensity_array, threshold_value=10000)
    print ("Original threshold value: ", threshold.threshold_value, "\n")
    global coordinates
    coordinates = threshold.get_coordinates_above_threshold()

    radius0 = 1; radius1=2; radius2=3; radius3=4
    completed = False
    while not completed:
        threshold = PeakThresholdProcessor(intensity_array, threshold_value=9000)
        coordinate_menu(intensity_array, threshold_value=threshold.threshold_value, coordinates=coordinates, radius=radius0)
        intensity = intensity_peak; avg = avg_values
        spot_estimate_peak = intensity - avg    
        coordinate_menu(intensity_array, threshold_value=threshold.threshold_value, coordinates=coordinates, radius=radius1)
        intensity = intensity_peak; avg = avg_values
        spot_estimate_peak = intensity - avg    
        print("Peak Estimate for ring 1:", spot_estimate_peak, 'with radius of', radius1)
        coordinate_menu(intensity_array, threshold_value=threshold.threshold_value, coordinates=coordinates, radius=radius2)
        intensity = intensity_peak; avg = avg_values
        spot_estimate_peak = intensity - avg    
        print("Peak Estimate for ring 2:", spot_estimate_peak, 'with radius of', radius2)    
        coordinate_menu(intensity_array, threshold_value=threshold.threshold_value, coordinates=coordinates, radius=radius2)
        intensity = intensity_peak; avg = avg_values
        spot_estimate_peak = intensity - avg    
        print("Peak Estimate for ring 3:", spot_estimate_peak, 'with radius of', radius3)
        completed = True   

######################################
if __name__ == "__main__":
    print(os.getcwd())
    stream_path = os.path.join(os.getcwd(), "high_low_stream", "test_high.stream")
    # call for high stream 
    main(stream_path)
    
    # call for low stream
    stream_path = os.path.join(os.getcwd(),"high_low_stream", "test_low.stream")
    # call for high stream 
    main(stream_path)







