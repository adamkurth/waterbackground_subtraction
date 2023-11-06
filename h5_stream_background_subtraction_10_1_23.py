#!/usr/bin/env python
import os
import sys
import numpy as np
import h5py as h5
from os.path import basename, splitext
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


##argument runnig script through term
# if sys.argv[1] == '-':
#     stream = sys.stdin
# else:
#     stream = open(sys.argv[1], 'r')

################################    
def load_file_h5():
    global filename         #scope outside of this method. 
    #filename = input("Please enter a filename to load: ")
    #FOR NOW
    #filename = "DATASET1_8_16_23-1.h5"
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
                
                create_scatter(result_x, result_y, result_z, highlight_x=x, highlight_y=y)
                #create_scatter_test(x_highlight=x, y_highlight=y, z_values=result_z)
                # #creating boolean array of segment within the defined circle, so see which index is in the circle (to then be summed)
                # bool_array = in_circle(segment, radius,x_center=x, y_center=y)
                
                #returns boolean array of traversed values.
                bool_square = np.zeros_like(segment, dtype=bool)
                print('BOOLEAN: before traversing.', '\n', bool_square, '\n') 
            
                ######start 3 ring integration
                values_array = extract_region(image_array, region_size=radius, x_center=x, y_center=y)
                
                global avg_values, intensity_peak
                total_sum = 0; skipped_point = None; count = 0; intensity_peak= 0
                #traverses through (i = row) , (j = column)
                for col_index in range(values_array.shape[0]):
                    for row_index in range(values_array.shape[1]):
                        if values_array[row_index, col_index] >= 0:
                            count += 1
                            bool_square[row_index, col_index] = True
                            if col_index == radius and row_index == radius:
                                skipped_point = (row_index, col_index)  
                                intensity_peak = values_array[row_index, col_index]
                                print(f'Peak point to be skipped: ({row_index}, {col_index}) ', values_array[radius,radius])
                            else:
                                print(f'(row,col) ({row_index}, {col_index}) with a value of ', values_array[row_index, col_index])
                                total_sum += values_array[row_index, col_index]
                                
                print('\n######################')
                print(bool_square)
                # print(square)
                print('Number of traversed cells', count)
                print('Peak point to be skipped:', skipped_point)
                print('Total sum:',total_sum)
                if count > 0:
                    avg_values = total_sum / count
                else: 
                    avg_values = "Could not divide by 0."
                print('Average surrounding peak:',avg_values)
                return avg_values,intensity_peak
                break
            else: 
                print("Invalid coordinate idex.")
        except ValueError:
            print("Invalid input. Enter a number of 'q' to quit.")

################################## 

        
def load_stream():
    global stream_name; global stream_coord
    global result_x, result_y, result_z
    #streamfile = input("Please enter a STREAM filename to load: ")    
    stream_name = "test_manipulate2.stream"

    try:
        stream = open(stream_name, "r")
        print("\nLoaded file successfully.", stream_name, '\n')
    except Exception as e:
        print("\nAn error has occurred:", str(e),'\n')
    
    reading_peaks = False
    reading_geometry = False
    reading_chunks = True 

    result_x = []; result_y = []; result_z = []
    for line in stream:
        if reading_chunks:
           if line.startswith('End of peak list'):
               reading_peaks = False
           elif line.startswith("   h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel"):
               reading_peaks = True
           elif reading_peaks:
                try:
                    elements = line.split()
                    fs, ss = [float(i) for i in line.split()[7:9]]
                    intensity = float(elements[3])
                    result_x.append(fs)
                    result_y.append(ss)
                    result_z.append(intensity)
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
    build_coord_intensity_array(result_x, result_y,result_z)
    
def build_coord_intensity_array(x,y,z):
    global coordinates_and_intensities, coordinates
    coordinates = np.transpose(np.array((x,y), dtype=np.float32))
    coordinates_and_intensities = np.column_stack((coordinates,z))
    print(coordinates_and_intensities)
    z = coordinates_and_intensities[:,2]
    return coordinates_and_intensities

def create_scatter(x, y, z, highlight_x=None, highlight_y=None):
    global coordinates_and_intensities, coordinates
    coordinates = np.transpose(np.array((x,y), dtype=np.float32))
    coordinates_and_intensities = np.column_stack((coordinates,z))
    print(coordinates_and_intensities)
    z = coordinates_and_intensities[:,2]
    
    ####
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], z, c=z, cmap='viridis', marker='o')

    highlight_z = intensity_array[highlight_x,highlight_y]
    print("Intensity value", highlight_z, "\n")
    # Highlight the specific point if provided
    if highlight_x is not None and highlight_y is not None:
        ax.scatter([highlight_x], [highlight_y], [highlight_z], c='red', marker='x', s=100, label='Highlighted Point')

    cbar = plt.colorbar(scatter)
    cbar.set_label('Intensity')
    
    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Intensity')
    plt.title('3D Scatter Plot of (X, Y, Intensity)')
    plt.show()
    return None

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
   

def write_hdf5(filename, data, dataset_name="/data/data/intensity_data"):
    # WRITES HDF5 FILE
    with h5.File(filename+'-HDF5.h5', 'w') as f:
        dset = f.create_dataset(dataset_name, data=data)
        
        
######################################
if __name__ == "__main__":
    load_stream()
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
    read_hdf5(filename=stream_name, location=location)
    threshold = PeakThresholdProcessor(intensity_array, threshold_value=7000)
    print ("Original threshold value: ", threshold.threshold_value, "\n")
    global coordinates
    coordinates = threshold.get_coordinates_above_threshold()



    radius1=2; radius2=3; radius3=4
    completed = False
    while not completed:
        threshold = PeakThresholdProcessor(intensity_array, threshold_value=9000)
        coordinate_menu(intensity_array, threshold_value=threshold.threshold_value, coordinates=coordinates, radius=radius1)
        intensity = intensity_peak; avg = avg_values
        spot_estimate_peak = intensity - avg    
        print("Peak Estimate for ring 1:", spot_estimate_peak, 'with radius of', radius1)
        # coordinate_menu(intensity_array, threshold_value=threshold.threshold_value, coordinates=coordinates, radius=radius2)
        # intensity = intensity_peak; avg = avg_values
        # spot_estimate_peak = intensity - avg    
        # print("Peak Estimate for ring 2:", spot_estimate_peak, 'with radius of', radius2)    
        # coordinate_menu(intensity_array, threshold_value=threshold.threshold_value, coordinates=coordinates, radius=radius2)
        # intensity = intensity_peak; avg = avg_values
        # spot_estimate_peak = intensity - avg    
        # print("Peak Estimate for ring 3:", spot_estimate_peak, 'with radius of', radius3)
            # SEEMS TO IGNORE THRESHOLD
            # COORDINATE LIST IS NOT SHORTENED, 
            # BECOMES AS LONG AS THE NUMBER OF TOTAL VALUES IN INTENSITY ARRAY
        completed = True   
        
    # coordinate_menu(intensity_array, 9000, coordinates, radius2)
    # coordinate_menu(intensity_array, 9000, coordinates, radius3)

    # SCATTER FUNCTION DOES NOT SEEM TO BE ACCURATELY SETTING THE PEAK VALUES. 



    # coordinate_menu(image_array, 7000, coordinates, radius2)
    # coordinate_menu(image_array, 7000, coordinates, radius3)


    # filename = "test_manipulate2.stream-HDF5.h5"
    # location = "intensity_array"
    # read_hdf5(filename, location="/data/data"+location)
        

























# # ## READ
# filename =  "test_manipulate2HDF5.h5"
# image_array = None

# image = h5.File(filename, "r") 
# #image_array = None
# with h5.File(filename, "r") as f:
#     #prints <HDF5 dataset "data": shape (4371, 4150), type "<f4">
#     dset = image['/z_values'][()]
#     #dset = image["entry/data/data"][()]     #returns np array of (4371,4150) of 0's
#     image_array = np.array(dset)
#     image_array_size = dset.shape
#     print(image_array)
#     image.close




