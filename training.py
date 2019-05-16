
# Import math and CV libs
from helper_functions import get_hog_features, bin_spatial, color_hist, slide_window, draw_boxes, convert_image ,get_dataset,extract_features
from heatmap import add_heat, apply_threshold, draw_labeled_bboxes, heat_threshold
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import LinearSVC
import os


class Parameters():

    def __init__(self, color_space='YCrCb', spatial_size=(16, 16),
                 hist_bins=32, orient=8, 
                 pix_per_cell=8, cell_per_block=2, hog_channel="ALL", scale = 1.5,hist_range = (0, 256),
                 spatial_feat=True, hist_feat=True, hog_feat=True):
        # HOG parameters
        self.color_space = color_space
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.scale = scale
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        self.hist_range = hist_range

parameter = Parameters()





path = './dataset/vehicles'
cars = get_dataset(path)
path = './dataset/non-vehicles'
notcars = get_dataset(path)

car_features =[]
notcar_features=[]
car_features = list(map(lambda img: extract_features(img, parameter), cars))
notcar_features = list(map(lambda img: extract_features(img, parameter), notcars))


X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',parameter.orient,'orientations',parameter.pix_per_cell,
    'pixels per cell and', parameter.cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))


def hog_subsample(img, ystart, ystop, svc, scaler, parameter):
    draw_img = np.copy(img)
    cspace = parameter.color_space
    cells_per_step = 1
    img_tosearch = img[ystart:ystop,:,:]

    if cspace != 'RGB':
        if cspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(img_tosearch)  
    
    if parameter.scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/parameter.scale), np.int(imshape[0]/parameter.scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // parameter.pix_per_cell) - parameter.cell_per_block + 1
    nyblocks = (ch1.shape[0] // parameter.pix_per_cell) - parameter.cell_per_block + 1 
    nfeat_per_block = parameter.orient*parameter.cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // parameter.pix_per_cell) - parameter.cell_per_block + 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, parameter.orient, parameter.pix_per_cell, parameter.cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, parameter.orient, parameter.pix_per_cell, parameter.cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, parameter.orient, parameter.pix_per_cell, parameter.cell_per_block, feature_vec=False)
    car_windows = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            xleft = xpos*parameter.pix_per_cell
            ytop = ypos*parameter.pix_per_cell
            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=parameter.spatial_size)
            hist_features = color_hist(subimg, nbins=parameter.hist_bins, bins_range=parameter.hist_range)
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*parameter.scale )
                ytop_draw = np.int(ytop*parameter.scale )
                win_draw = np.int(window*parameter.scale )
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                car_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return car_windows




def pipeline(img):
    ystart = 350
    ystop = 656
    threshold = 1 
    car_windows = hog_subsample(img,ystart, ystop, svc, 1.5, parameter)
    draw_img, heat_map = heat_threshold(img, threshold, svc, X_scaler, car_windows)
    
    return draw_img

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(img):
    return pipeline(img)

white_output = 'output_video/project_video2.mp4'
clip1 = VideoFileClip("test_video.mp4")
white_clip = clip1.fl_image(process_image)

