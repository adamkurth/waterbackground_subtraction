import numpy as np

class ArrayRegion:
    def __init__(self, array):
        self.array = array
        self.x_center = array.shape[0] // 2
        self.y_center = array.shape[1] // 2
        self.region_size = 9 
    
    def set_peak_coordinate(self, x, y):
        self.x_center = x
        self.y_center = y

    def set_region_size_terminal(self, size):
        #limit that is printable in terminal
        self.region_size = size
        max_printable_region = min(self.array.shape[0], self.array.shape[1]) // 2
        self.region_size = min(size, max_printable_region)
    
    def set_region_size(self, size):
        self.region_size = size
        
    def get_region(self):
        x_range = slice(self.x_center - self.region_size, self.x_center + self.region_size+1)
        y_range = slice(self.y_center - self.region_size, self.y_center + self.region_size+1)
        region = self.array[x_range, y_range]
        return region

    def extract_region(self, x_center, y_center, region_size):
        self.set_peak_coordinate(x_center, y_center)
        self.set_region_size(region_size)
        region = self.get_region()
        # Set print options for better readability
        np.set_printoptions(precision=8, suppress=True, linewidth=120, edgeitems=7)
        return region
                
    def get_exclusion_mask(self):
        mask = np.ones(self.array.shape, dtype=bool)
        x_min = min(self.array.shape[1], self.x_center + self.region_size + 1)
        x_max = max(0, self.x_center - self.region_size)
        y_min = min(self.array.shape[0], self.y_center + self.region_size + 1)
        y_max = max(0, self.y_center - self.region_size)
        mask[y_min:y_max, x_min:x_max] = False
        return mask
                
        