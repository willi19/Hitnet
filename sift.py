import cv2
import numpy as np

filename = '/home/vclab/Desktop/Hitnet/videos/samples3/obj/rectified'
left_img = cv2.imread(filename+"/left4000.jpg")
right_img = cv2.imread(filename+"/right4000.jpg")

left_gray= cv2.cvtColor(left_img,cv2.COLOR_BGR2GRAY)
right_gray= cv2.cvtColor(right_img,cv2.COLOR_BGR2GRAY)
sift = cv2.ORB_create()

kp_left, des_left = sift.detectAndCompute(left_gray, None)
kp_right, des_right = sift.detectAndCompute(right_gray, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des_left,des_right)

matches = sorted(matches, key=lambda x:x.distance)

res = cv2.drawMatches(left_img,kp_left,right_img,kp_right,matches,None,flags=2)
res = cv2.resize(res,(res.shape[1]//4,res.shape[0]//4))
cv2.imwrite("SIFT.jpg",res)