import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# I think I am doing this right. This is bad!!!

dir = "1mX1mY_5mDepthRender"

images = glob.glob(f"{dir}/*.png")
images = ['thyme_cropped1.jpg', 'thyme_cropped2.jpg']

orb = cv2.ORB_create()
img1 = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
img1_draw = cv2.imread(images[0])
img2 = cv2.imread(images[1], cv2.IMREAD_GRAYSCALE)
img2_draw = cv2.imread(images[1])
cv2.imshow('Original', img1_draw)
cv2.waitKey()

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Match descriptors.
matches = bf.match(des1,des2)
 
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

list_kp1 = [kp1[mat.queryIdx].pt for mat in matches] 
list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]

print(list_kp1[0])

for i, kp in enumerate(list_kp1[:20]):
    cv2.line(img1_draw, (int(kp[1]), int(kp[0])), (int(list_kp2[i][1]), int(list_kp2[i][0])), (255,0,0), 2)
    cv2.line(img2_draw, (int(kp[1]), int(kp[0])), (int(list_kp2[i][1]), int(list_kp2[i][0])), (255,0,0), 2)
 
# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
cv2.imshow('ORB matches 1', img1_draw)
cv2.waitKey()
cv2.imshow('ORB matches 2', img2_draw)
cv2.waitKey()
cv2.imshow('ORB matches 3', img3)
cv2.waitKey()