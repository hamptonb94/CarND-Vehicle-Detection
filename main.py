import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from Detectors import CarDetector
import UtilWindows

detector = CarDetector()

def imagePipeline(image, fileName=None):
    """Complete process for each frame image.  If 'fileName' is given then each stage
        of the pipeline will write out an image for debugging.
    """
    if fileName:
        mpimg.imsave(os.path.join("test_images/outputs/", fileName), image)
        
    imgFinal = detector.update(image)
    
    if fileName:
        boxImg = UtilWindows.draw_boxes(image, detector.windows1, color=(0, 0, 255), thick=3)
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+",1,windows.jpg"), boxImg)
        boxImg = UtilWindows.draw_boxes(image, detector.windows2, color=(0, 255, 0), thick=2)
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+",2,windows.jpg"), boxImg)
        boxImg = UtilWindows.draw_boxes(image, detector.windows3, color=(255, 0, 0), thick=1)
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+",3,windows.jpg"), boxImg)
        
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+",4,boxes.jpg"), detector.windowImg)
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+",5,heat.jpg"), detector.heatImg)
        
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+",9,final.jpg"), imgFinal)
    
    return imgFinal
    

def makeNewDir(dir):
    if not os.path.isdir(dir):
        try:
            os.mkdir(dir)
        except:
            raise Exception("Error, %s is not a valid output directory\n" %dir)
    if not os.path.isdir(dir):
        raise Exception("Error, could not make run directory %s, exiting" %dir)


def processImages():
    global detector
    makeNewDir("test_images/outputs/")
    fileNames = os.listdir("test_images/")
    for fileName in fileNames:
        if 'jpg' not in fileName:
            continue
        print("Processing: ", fileName)
        fullName = os.path.join("test_images",fileName)
        image = mpimg.imread(fullName)
        detector = CarDetector(history=False) # reset detectors for test images
        imagePipeline(image, fileName)
        
from moviepy.editor import VideoFileClip

def processMovie(movieName):
    outputName = 'out-'+movieName
    clip1 = VideoFileClip(movieName)
    out_clip = clip1.fl_image(imagePipeline) #NOTE: this function expects color images!!
    out_clip.write_videofile(outputName, audio=False)

import sys
if __name__ == '__main__':
    """Main processing script.
        If no arguments given, it will process all images in the test_images folder.  
        A single argument is the name of a movie to process.
    """
    if len(sys.argv) > 1:
        processMovie(sys.argv[1])
    else:
        processImages()
