import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

dir = "1mX_5mDepthRender"

images = glob.glob(f"{dir}/*.png")

orb = cv2.ORB_create()
img = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)

# find the keypoints with ORB
kp = orb.detect(img,None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
cv2.imshow('ORB', img2)
cv2.waitKey()

sift = cv2.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(img,None)
img3 = cv2.drawKeypoints(img, kp_sift, None, color=(0,255,0), flags=0)
cv2.imshow('SIFT',img3)
cv2.waitKey()

imgH = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
imgH_draw = cv2.imread(images[0])
harrisCorners = cv2.cornerHarris(img, blockSize=2, ksize=3, k=.04)
harrisCorners = cv2.dilate(harrisCorners,None)
ret, dst = cv2.threshold(harrisCorners,0.01*harrisCorners.max(),255,0)
dst = np.uint8(dst)
# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
for item in centroids:
    x, y = item 
    x = int(x) 
    y = int(y) 
    cv2.circle(imgH_draw, (x, y), 4, (0, 255, 0), 2) 
cv2.imshow('harrisCorner',imgH_draw) 
cv2.waitKey()

# Shi-Tomasi
imgST_draw = cv2.imread(images[0])
imgST = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
corners = cv2.goodFeaturesToTrack( 
    imgST, maxCorners=50, qualityLevel=0.02, minDistance=20) 
corners = np.float32(corners) 

for item in corners: 
    x, y = item[0] 
    x = int(x) 
    y = int(y) 
    cv2.circle(imgST_draw, (x, y), 4, (0, 255, 0), 2) 
  
# Showing the image 
cv2.imshow('good_features', imgST_draw) 
cv2.waitKey()



fast = cv2.FastFeatureDetector_create() 
fast.setNonmaxSuppression(False) 

# Drawing the keypoints 
kp = fast.detect(img, None) 
kp_image = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0)) 
  
cv2.imshow('FAST', kp_image) 
cv2.waitKey() 