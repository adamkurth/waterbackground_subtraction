import os
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

class PeakThresholdProcessor: 
    def __init__(self, image_array, threshold_value=0):
        self.image_array = image_array
        self.threshold_value = threshold_value
    def set_threshold_value(self, new_threshold_value):
        self.threshold_value = new_threshold_value
    def get_coordinates_above_threshold(self):  
        coordinates = np.argwhere(self.image_array > self.threshold_value)
        return coordinates
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
    
def load_file_h5(filename):
    if not os.path.exists(filename):
        print("File not found within working directory.")
        return
    try:
        with h5.File(filename, "r") as f: 
            print("\nLoaded file successfully.", filename)
    except Exception as e:
        print("\nAn error has occurred:", str(e))
               
def extract_region(image_array, region_size, x_center, y_center):
    extract = ArrayRegion(image_array)
    extract.set_peak_coordinate(x_center,y_center)
    extract.set_region_size(region_size)
    np.set_printoptions(floatmode='fixed', precision=10)
    np.set_printoptions(edgeitems=3, suppress=True, linewidth=200)
    region = extract.get_region()
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
                display_region = extract_region(image_array, region_size=4, x_center=x, y_center=y)
                print('DISPLAY REGION \n', display_region, '\n')
                
                #segment is the area with the given radius that's passed through the function.
                segment = extract_region(image_array, region_size=radius, x_center=x, y_center=y)
                print ('SEGMENT \n', segment, '\n')
                
                #returns boolean array of traversed values.
                bool_square = np.zeros_like(segment, dtype=bool)
                print('BOOLEAN', '\n', bool_square, '\n') 

                values_array = extract_region(image_array, region_size=radius, x_center=x, y_center=y)
                
                global avg_values, intensity_peak
                total_sum = 0; skipped_point = None; count = 0; intensity_peak = 0
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
                avg_values = total_sum / count
                print('Average surrounding peak:',avg_values)
                
                build_coord_intensity()
                
                create_scatter(result_x, result_y, result_z, highlight_x=x, highlight_y=y)
                return avg_values,intensity_peak
                break
            else: 
                print("Invalid coordinate idex.")
        except ValueError:
            print("Invalid input. Enter a number of 'q' to quit.")

def build_coord_intensity():
    global result_x, result_y, result_z, coordinates_and_intensities
    result_z = []
    threshold = PeakThresholdProcessor(image_array, threshold_value=.01)
    coord_above_threshold = threshold.get_coordinates_above_threshold()
    coord_above_threshold = np.array(coord_above_threshold)
    
    for i in coord_above_threshold: 
        result_x = coord_above_threshold[:,0]
        result_y = coord_above_threshold[:,1]
    
    result_x = np.array(result_x)
    result_y = np.array(result_y)
    
    for i in range(len(coord_above_threshold)):
        x = result_x[i]
        y = result_y[i]
        z = image_array[x,y]
        result_z.append(z)
    # creating a coordinate and intensity array to store the values we want to plot.
    coordinates_and_intensities = np.column_stack((result_x, result_y, result_z))
    return result_x, result_y, result_z, coordinates_and_intensities

def create_scatter(x, y, z, highlight_x=None, highlight_y=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(coordinates_and_intensities[:, 0], coordinates_and_intensities[:, 1], coordinates_and_intensities[:,2], c=z, cmap='viridis', marker='o')

    highlight_z = image_array[highlight_x,highlight_y]
    print("Intensity value", highlight_z, "\n")
    # Highlight the specific point if provided
    if highlight_x is not None and highlight_y is not None:
        ax.scatter([highlight_x], [highlight_y], [highlight_z], c='red', marker='x', s=100, label='Highlighted Point')

    cbar = plt.colorbar(scatter)
    cbar.set_label('Intensity')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Intensity')
    plt.title('3D Scatter Plot of (X, Y, Intensity)')
    plt.show()
    return None

def main(filename):
    load_file_h5(filename)
    global image_array
    image_array = None
    image = h5.File(filename, "r") 
    image_array = None
    with h5.File(filename, "r") as f:
        dset = image["entry/data/data"][()]     #returns np array of (4371,4150) of 0's
        image_array = np.array(dset)
        image_array_size = dset.shape
        image.close()
        
    """ 3 RING INTEGRATION """
        
    threshold = PeakThresholdProcessor(image_array, threshold_value=1000)
    print ("Original threshold value: ", threshold.threshold_value, "\n")
    global coordinates
    coordinates = threshold.get_coordinates_above_threshold()

    radius = [1,2,3,4]
    completed = False
    build_coord_intensity()
    while not completed:
        for r in radius:
            coordinate_menu(image_array, 1000, coordinates, r)
            intensity = intensity_peak; avg = avg_values
            spot_estimate_peak = intensity - avg
            print("Peak Estimate for ring", r, ":", spot_estimate_peak, 'with radius of', r)
        completed = True
    
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(script_dir)
    images_dir = os.path.join(parent_dir, "images")
    print("Working directory:", images_dir)
    image_path1 = os.path.join(images_dir, "9_18_23_high_intensity_3e8keV-2.h5")
    print(image_path1)
    # image_path2 = os.path.join(images_dir, "9_18_23_low_intensity_3e7keV-2.h5")
    # main(image_path1)
    # main(image_path2)
    
