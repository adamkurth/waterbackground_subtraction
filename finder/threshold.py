import numpy as np
from scipy.signal import find_peaks
        
class PeakThresholdProcessor: 
    def __init__(self, image, threshold_value=0):
        self.image = image
        self.threshold_value = threshold_value
    
    def set_threshold_value(self, new):
        self.threshold_value = new
    
    def get_coordinates_above_threshold(self):  
        coordinates = np.argwhere(self.image > self.threshold_value)
        return coordinates
    
    def get_local_maxima(self):
        image_1d = self.image.flatten()
        peaks, _ = find_peaks(image_1d, height=self.threshold_value)
        coordinates = [self.flat_to_2d(idx) for idx in peaks]
        return coordinates
        
    def flat_to_2d(self, index):
        shape = self.image.shape
        rows, cols = shape
        return (index // cols, index % cols) 
