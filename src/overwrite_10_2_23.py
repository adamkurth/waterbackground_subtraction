import os
import shutil
import numpy as np
import h5py as h5
import h5_stream_background_subtraction_10_2_23 as streampy

def load_stream(stream_name):
    global data_columns
    
    data_columns = {
    'h':[], 'k':[], 'l':[],
    'I':[], 'sigmaI':[], 'peak':[], 'background':[],
    'fs':[],'ss':[], 'panel':[]}      
    
    reading_peaks = False
    reading_geometry = False
    reading_chunks = True 
    
    try:
        stream = open(stream_name, 'r') 
        print("\nLoaded file successfully.", stream_name, '\n')
    except Exception as e: 
        print("\nAn error has occurred in load method:", str(e),'\n')
   
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
    return data_columns


#############################################

def duplicate_before_overwrite(filename):
    # taking filename and adding copy extension to it.
    base_file, extension = filename.rsplit('.',1)
    new_base_file = f'{base_file}_copy'
    new_filename = f'{new_base_file}.{extension}'
    duplicate = shutil.copyfile(filename, new_filename)
    return duplicate

#############################################

def compare_high_low(high_data, low_data, *columns):
    """
    Compare the high and low data and return the compared data.
    """
    compared_data = {}
    for col in columns:
        if col in high_data and col in low_data:
            print(f'High: {high_data[col]} \n')
            print(f'Low: {low_data[col]} \n')
            print()
            compared_data[col] = (high_data[col], low_data[col])
            retrieve(list(high_data), list(low_data), *columns)
    return compared_data

def retrieve(data_columns, *args):
    result = []
    try:
        # taking in data_columns and selecting the desired columns to retrieve
        result = [data_columns[col] for col in args if col in data_columns]
    except Exception as e:
        pass
    return result
#############################################


def overwrite_low_in_high(filename, overwrite_data):
    """
    Overwrite the low data in the high stream file with the given overwrite data.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    with open(filename, 'r+') as f:
        for line in lines:
            if line.startswith("   h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel"):
                f.write(line)
                for i in range(len(overwrite_data['h'])):
                    formatted_row = '{:>4} {:>4} {:>4} {:>9} {:>12} {:>12} {:>12} {:>6} {:>6} {:>6}\n'.format(
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
            else:
                # Write the unmodified line to the file
                f.write(line)


def intensity_finder(x_coords, y_coords, image_name):
    """
    Retrieve the intensity values for every x,y coordinate in the image.
    """
    with h5.File(image_name, "r") as f:
        intensities = f['/entry/data/data'][()]
    intensities = np.array(intensities)
    found_intensities = []
    for x, y in zip(x_coords, y_coords):
        if x < intensities.shape[0] and y < intensities.shape[1]:
            found_intensities.append(intensities[int(x), int(y)])
    return found_intensities


def populate_intensity_array(data_columns, image_name):
    """
    Populate the intensity array with the intensity values for each x,y coordinate.
    """
    # reads the h5 image
    with h5.File(image_name, "r") as f:
        intensities = f['/entry/data/data'][()]
    intensities = np.array(intensities)
    # generates a new array of zeros with the same shape as the image
    new_intensities = np.zeros((intensities.shape[0], intensities.shape[1]))
    # for each x,y coordinate in the data_columns, set the value in the new array to the intensity value
    # populate the intensity array with corresponding (fs,ss) coordinates
    for i in range(len(data_columns['fs'])):
        x = int(data_columns['fs'][i])
        y = int(data_columns['ss'][i])
        if x < intensities.shape[0] and y < intensities.shape[1]:
            new_intensities[x][y] = intensities[x][y]
    return new_intensities

def main():
    print("Current working directory:", os.getcwd())
    src_path = os.getcwd()
    stream_dir = os.path.join(src_path, "high_low_stream")
    image_dir = os.path.join(src_path, "images")
    
    intensities_array = None
    high_stream_name = 'test_high.stream'
    low_stream_name = 'test_low.stream'

    high_stream_path = "high_low_stream/test_high.stream"
    low_stream_path = "high_low_stream/test_low.stream"

    if not os.path.exists(high_stream_path):
        print(f"File {high_stream_path} does not exist.")
        return
    elif os.path.exists(low_stream_path) and os.path.exists(high_stream_path):
        print(f"Files {low_stream_path} and {high_stream_path} exist.")

    if not os.path.exists(high_stream_path):
        print(f"File {high_stream_path} does not exist.")
        return
    elif os.path.exists(low_stream_path) and os.path.exists(high_stream_path):
        print(f"Files {low_stream_path} and {high_stream_path} exist.")

    # compare_high_low(high_data, low_data)
    high_data = load_stream(high_stream_path)
    low_data = load_stream(low_stream_path)
    compare_high_low(high_data, low_data)

    # Took low data from low_stream and put in high_stream file.
    overwrite_data = low_data
    overwrite_low_in_high(high_stream_path, overwrite_data)
    
    # compare any columns in data_columns
    # compare_high_low(high_data, low_data, "h")

    # now high_stream has data from low_stream
    
    image_name = '9_18_23_high_intensity_3e8keV-1_test.h5'
    image_path = os.path.join(image_dir, image_name)

    # retrieved from stream coordinate menu
    intensities = intensity_finder(high_data['fs'], high_data['ss'], image_path)

    # populate_inteneity_array is not correctly working
    intensities_array = populate_intensity_array(high_data, image_path)

    print("Number of non-zero values in intensity array\t", np.count_nonzero(intensities_array))  # 1251 10/13/23

    # for debugging
    # intensities_array = np.array(intensities_array)
    # print(intensities_array)
    # compare_high_low(high_data, low_data, "I")

    threshold_stream = streampy.PeakThresholdProcessor(intensities_array, threshold_value=1e-5) # very low!
    print("Original threshold value: ", threshold_stream.threshold_value, "\n")
    coordinate_list_stream = threshold_stream.get_coordinates_above_threshold()
    
    completed = False
    radius = [1,2,3,4]

    threshold = streampy.PeakThresholdProcessor(intensities_array, threshold_value=9000)
    for r in [1, 2, 3, 4]:
        print(f"Threshold value for radius {r}: {threshold.threshold_value}")
        streampy.coordinate_menu(intensities_array, threshold_value=threshold.threshold_value, coordinates=coordinate_list_stream, radius=r)
        print(f"Completed coordinate menu for radius {r}")
        completed = True
        
    
if __name__ == '__main__':           
    main() 

