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

CLASSIFIER_FILE = 'classifier_calibration.p'
SCALER_FILE     = 'scaler_calibration.p'

class CarClassifier:
    def __init__(self):
        if os.path.isfile(CLASSIFIER_FILE):
            # load calibration
            self.svc    = pickle.load(open(CLASSIFIER_FILE, "rb"))
            self.scaler = pickle.load(open(SCALER_FILE, "rb"))
        else:
            self.train()
    
    def predict(self, features):
        return self.svc.predict(features)
    
    def train(self):
        print("Training the classifier")
        
        # Read in cars and notcars
        images  = glob.glob('./vehicles_smallset/**/*.jpeg', recursive=True)
        images += glob.glob('./non-vehicles_smallset/**/*.jpeg', recursive=True)
        cars = []
        notcars = []
        for image in images:
            if 'image' in image or 'extra' in image:
                notcars.append(image)
            else:
                cars.append(image)        
        
        car_features    = extract_features(cars)
        notcar_features = extract_features(notcars)

        X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        self.scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = self.scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:',orient,'orientations',pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC 
        self.svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        self.svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))
        
        # save trained models
        f = open(CLASSIFIER_FILE, "wb")
        pickle.dump(self.svc,  f)
        f = open(SCALER_FILE, "wb")
        pickle.dump(self.scaler,  f)



import sys
if __name__ == '__main__':
    carclass = CarClassifier()
    print("Loaded classifier ok")
    if len(sys.argv) > 1 and sys.argv[1] == 'retrain':
        carclass.train()



