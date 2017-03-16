
from UtilWindows import *
import UtilClassifiers

y_start_stop = [400, 700] # Min and max in y to search in slide_window()

class CarDetector:
    def __init__(self):
        self.windows = []
        self.ready = False
        self.classifier = UtilClassifiers.CarClassifier()
    
    def setup(self, image):
        self.windows1 = slide_window(image, y_start_stop=[400, None], xy_window=(128, 128), xy_overlap=(0.8, 0.8))
        self.windows2 = slide_window(image, y_start_stop=[400, 592],  xy_window=( 64,  64), xy_overlap=(0.6, 0.6))
        self.windows3 = slide_window(image, y_start_stop=[400, 502],  xy_window=( 32,  32), xy_overlap=(0.6, 0.6))
        self.ready = True
    
    def update(self, image):
        if not self.ready:
            self.setup(image)
        
        hot_windows1 = search_windows(image, self.windows1, self.classifier)
        hot_windows2 = search_windows(image, self.windows2, self.classifier)
        hot_windows3 = search_windows(image, self.windows3, self.classifier)
        hot_windows = hot_windows1 + hot_windows2 + hot_windows3
        
        ## TODO: generate heat map
        
        
        ## Merge heat map over several frames
        
        ## Threshold heatmap
        
        
        draw_image = np.copy(image)
        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
        return window_img
    


