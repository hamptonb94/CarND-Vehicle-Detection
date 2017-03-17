from collections import deque
from scipy.ndimage.measurements import label

from UtilWindows import *
import UtilClassifiers

y_start_stop = [400, 700] # Min and max in y to search in slide_window()

class CarDetector:
    def __init__(self, history=True):
        self.windows = []
        self.ready = False
        self.classifier = UtilClassifiers.CarClassifier()
        self.threshold  = 1
        self.histLength = 1
        if history:
            self.threshold  = 12
            self.histLength = 25
        self.heatmaps = deque(maxlen=self.histLength)

    
    def setup(self, image):
        self.windows1 = slide_window(image, y_start_stop=[380, 636], xy_window=(128, 128), xy_overlap=(0.8, 0.8))
        self.windows2 = slide_window(image, y_start_stop=[400, 560],  xy_window=( 64,  64), xy_overlap=(0.8, 0.8))
        self.ready = True
    
    def update(self, image):
        if not self.ready:
            self.setup(image)
        
        hot_windows1 = search_windows(image, self.windows1, self.classifier)
        hot_windows2 = search_windows(image, self.windows2, self.classifier)
        hot_windows = hot_windows1 + hot_windows2
        
        # visualize raw boxes found around cars
        draw_image = np.copy(image)
        self.windowImg = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
        
        # generate heat map
        self.applyHeatmap(image, hot_windows)
        
        return self.finalImg
    
    def applyHeatmap(self, image, bbox_list):
        # add empty heat map for current frame
        heatmap = np.zeros_like(image[:,:,0]).astype(np.uint8)
        
        # add heat for every hot window
        for box in bbox_list:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        
        # Need a minimum of 2 to be counted
        heatmap[heatmap < 2] = 0
        
        # Add latest map to our deck
        self.heatmaps.appendleft(heatmap)
        
        # join all frames of heat maps
        totalHeatmap = np.zeros_like(image[:,:,0]).astype(np.uint8)
        for heatmap in self.heatmaps:
            totalHeatmap += heatmap
        
        # Zero out pixels below the threshold
        totalHeatmap[totalHeatmap <= self.threshold] = 0
        
        # Visualize the heatmap when displaying 
        totalHeatmap *= 25   
        self.heatmap  = np.clip(totalHeatmap, 0, 255)
        self.heatMask = np.dstack((self.heatmap, np.zeros_like(self.heatmap), np.zeros_like(self.heatmap))).astype(np.uint8)
        self.heatImg  = self.heatMask #cv2.addWeighted(image, 1.0, self.heatMask, 0.8, 0.0)

        # Find final boxes from heatmap using label function
        labels = label(self.heatmap)
        self.finalImg = self.draw_labeled_bboxes(image, labels)
        
    def draw_labeled_bboxes(self, image, labels):
        labelImage = np.copy(image)
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            if abs(bbox[0][0] - bbox[1][0]) <= 4 or abs(bbox[0][1] - bbox[1][1]) <= 4:
                continue
            # Draw the box on the image
            cv2.rectangle(labelImage, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return labelImage

