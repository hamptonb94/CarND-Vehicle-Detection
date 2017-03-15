
from UtilWindows import *
import UtilClassifiers

y_start_stop = [400, 700] # Min and max in y to search in slide_window()

class CarDetector:
    def __init__(self):
        self.windows = []
        self.ready = False
        self.classifier = UtilClassifiers.CarClassifier()
    
    def setup(self, image):
        self.windows = slide_window(image, y_start_stop=y_start_stop, xy_window=(96, 96), xy_overlap=(0.5, 0.5))
        self.ready = True
    
    def update(self, image):
        if not self.ready:
            self.setup(image)
        
        hot_windows = search_windows(image, self.windows, self.classifier)
        
        ## TODO: generate heat map
        
        ## Merge heat map over several frames
        
        ## Threshold heatmap
        
        
        draw_image = np.copy(image)
        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
        return window_img
    


