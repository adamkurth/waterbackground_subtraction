# Background Subtraction

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





- **`project_2.ipynb`**: This Jupyter notebook is the heart of the project. It contains the main analysis for the project, including data loading where the data necessary for the project is loaded, data cleaning where the loaded data is cleaned and preprocessed, exploratory data analysis where the cleaned data is analyzed to gain insights, model building where a model is built based on the insights gained, and evaluation where the built model is evaluated to measure its performance.