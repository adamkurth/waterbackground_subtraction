# Water Background Subtraction for Crystallography Image Analysis 

This repository is dedicated to the implementation of a project focused on background subtraction for X-ray crystallography. It contains several directories and files, each serving a specific purpose in the project.

## Directories

- **`/crystfel_scripts`**: This directory is a crucial part of the project as it contains all the scripts from CrystFEL. These scripts are integral to the project's implementation and have been adapted to perform manual background subtraction.

- **`/src`**: This is where the source code for the project resides. It includes Python scripts for various functionalities, Jupyter notebooks for detailed review and analysis, and other code files necessary for the project.

- **`/high_low_stream`**: This directory contails the stream files in question.

- **`/stream_files`**: This directory includes various copies of test stream files to analyze and change for the various python scripts. 

- **`/docs`**: This directory is dedicated to the documentation of the project. It includes design documents that outline the architecture and design decisions of the project, user manuals that guide users on how to use the project, and other documentation files that provide additional information about the project. *This will not be fully complete at this time.*

- **`/images`**: This directory is for the test images of the project. These will be labeled with the date of creation, a name to remember them by, and the attribute under consideration.

## Files in /src
- **`/src/h5_background_subtraction_10_2_23.py`**:
  - Imports: The script imports necessary libraries such as os, numpy, h5py, and matplotlib.pyplot.
  - Classes:
    - `PeakThresholdProcessor`: This class processes an image array and a threshold value. It has methods to set a new threshold value and get coordinates above the threshold.
    - `ArrayRegion`: This class represents a region of an array. It has methods to set the peak coordinate, set the region size, and get the region.    
  - Functions:
    - `load_file_h5(filename)`: This function loads an h5 file and handles exceptions if the file does not exist or cannot be opened.
    - `extract_region(image_array, region_size, x_center, y_center)`: This function extracts a region from an image array based on the given center coordinates and region size. 
    - `coordinate_menu(image_array, threshold_value, coordinates, radius)`: This function displays a menu for the user to select a coordinate to process. It then processes the selected coordinate and calculates the average surrounding peak and peak point.
    - `build_coord_intensity()`: This function builds the coordinates and intensities for a scatter plot.
    - `create_scatter(x, y, z, highlight_x=None, highlight_y=None)`: This function creates a 3D scatter plot of the coordinates and intensities.
    - `main(filename)`: This is the main function of the script. It loads the h5 file, creates an intensity array, and performs a 3-ring integration on the data. 
  - Execution: If the script is run as the main program, it sets the working directory to the parent directory of the script's location, then calls the main function with two different h5 files.

- **`/src/h5_stream_background_subtraction_10_2_23.py`**: Much the same as the previous file, but adapted for reading the `.stream` files.
    - `Imports`: The script imports necessary libraries such as os, sys, numpy, h5py, and matplotlib.pyplot.
    - Classes:
      - `PeakThresholdProcessor`: This class processes an image array and a threshold value. It has methods to set a new threshold value and get coordinates above the threshold.
      - `ArrayRegion`: This class represents a region of an array. It has methods to set the peak coordinate, set the region size, and get the region.
    - Functions:
      - `load_file_h5(filename)`: This function loads an h5 file and handles exceptions if the file does not exist or cannot be opened.
      - `extract_region(image_array, region_size, x_center, y_center)`: This function extracts a region from an image array based on the given center coordinates and region size.
      - `coordinate_menu(image_array, threshold_value, coordinates, radius)`: This function displays a menu for the user to select a coordinate to process. It then processes the selected coordinate and calculates the average surrounding peak and peak point.
      - `load_stream(stream_path)`: This function loads a stream file and reads its contents. It stores the data in a dictionary.
      - `main(stream_path)`: This is the main function of the script. It loads the stream file, creates an intensity array, and performs a 3-ring integration on the data.

    - Execution: If the script is run as the main program, it sets the working directory to the parent directory of the script's location, then calls the main function with two different stream files.

- **`/src/overwrite_10_2_23.py`**: This script requires the os, shutil, numpy, h5py, and h5_stream_background_subtraction_10_2_23 (as streampy) modules. Designed to process and manipulate stream data files. It includes functionality for loading stream data, comparing high and low data, overwriting low data in high stream file, finding intensity values for each x,y coordinate in an image, and populating an intensity array with these values.
  
- Functions: many of the same functions are used from the previous two scripts for overwrite script.
    - `load_stream(stream_name)`: This function loads a stream file and parses the data into a dictionary.
    - `duplicate_before_overwrite(filename)`: This function creates a copy of a file before it is overwritten.
    - `compare_high_low(high_data, low_data, *columns)`: This function compares high and low data for specified columns.
    - `retrieve(data_columns, *args)`: This function retrieves specified columns from the data.
    - `overwrite_low_in_high(filename, overwrite_data)`: This function overwrites the low data in the high stream file with the given overwrite data.
    - `intensity_finder(x_coords, y_coords, image_name)`: This function retrieves the intensity values for every x,y coordinate in the image.
    - `populate_intensity_array(data_columns, image_name)`: This function populates an intensity array with the intensity values for each x,y coordinate.
  - Main Function
  
The `main()` function orchestrates the execution of the script. It loads high and low stream data, compares them, overwrites the low data in the high stream file, finds intensity values for each x,y coordinate in an image, populates an intensity array with these values, and performs threshold processing on the intensity array.

Usage
To use this script, you need to have two stream files (high and low) and an image file in the '.h5' format. The script will process these files and output the results. The main function can be modified to suit your specific needs.

```{python}
    if __name__ == '__main__':           
        main() 
```

- **`project_2.ipynb`**: This Jupyter notebook is a further explanation of the project. It contains the main analysis for the project, including data loading where the data necessary for the project is loaded, data cleaning where the loaded data is cleaned and preprocessed, exploratory data analysis where the cleaned data is analyzed to gain insights, model building where a model is built based on the insights gained, and evaluation where the built model is evaluated to measure its performance.
