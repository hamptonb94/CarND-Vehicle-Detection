import numpy as np
import os.path
import pickle
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

from UtilFeatures import *

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off


color_spaces = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
orients = [7,9,11]
pix_per_cells = [4,8,16]
cell_per_blocks = [2,4]
hog_channels = [0,1,2,'ALL']
spatial_sizes = [(16, 16), (32, 32)]
hist_binss = [16,32]
totalCount = len(color_spaces) * len(orients) * len(pix_per_cells) * len(cell_per_blocks) * len(hog_channels) * len(spatial_sizes) * len(hist_binss)

print("Testing classifiers, spatial: ", spatial_feat, "hist features: ", hist_feat)
fileName = "out-traintest,%d,%d.dat"%(spatial_feat,hist_feat)
outFile = open(fileName, 'w')

# Read in cars and notcars
cars    = glob.glob('../image-data/vehicles_smallset/**/*.jpeg', recursive=True)
notcars = glob.glob('../image-data/non-vehicles_smallset/**/*.jpeg', recursive=True)

# experiment with different combinations
count = 0
for color_space in color_spaces:
    for orient in orients:
        for pix_per_cell in pix_per_cells:
            for cell_per_block in cell_per_blocks:
                for hog_channel in hog_channels:
                    for spatial_size in spatial_sizes:
                        for hist_bins in hist_binss:
                            # Check the training time for the SVC
                            t=time.time()
                            count += 1
                            #print("Iteration: ", count, " out of ", totalCount)
                            opts = "\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins)
                            
                            car_features = extract_features(cars, color_space=color_space, 
                                                    spatial_size=spatial_size, hist_bins=hist_bins, 
                                                    orient=orient, pix_per_cell=pix_per_cell, 
                                                    cell_per_block=cell_per_block, 
                                                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                                    hist_feat=hist_feat, hog_feat=hog_feat)
                            notcar_features = extract_features(notcars, color_space=color_space, 
                                                    spatial_size=spatial_size, hist_bins=hist_bins, 
                                                    orient=orient, pix_per_cell=pix_per_cell, 
                                                    cell_per_block=cell_per_block, 
                                                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                                    hist_feat=hist_feat, hog_feat=hog_feat)

                            X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
                            # Fit a per-column scaler
                            scaler = StandardScaler().fit(X)
                            # Apply the scaler to X
                            scaled_X = scaler.transform(X)

                            # Define the labels vector
                            y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

                            # Split up data into randomized training and test sets
                            X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=12)

                            #print('Feature vector length:', len(X_train[0]))
                            # Use a linear SVC 
                            svc = LinearSVC()
    
                            svc.fit(X_train, y_train)
                            t2 = time.time()
                            #print(round(t2-t, 2), 'Seconds to train SVC...')
                            
                            
                            # Check the score of the SVC
                            line = "%6.5f  "%(round(svc.score(X_test, y_test), 5)) + opts + "\t t={} {} out of {}".format(round(t2-t, 2), count, totalCount)
                            #print(line)
                            outFile.write(line + "\n")
                            outFile.flush()



