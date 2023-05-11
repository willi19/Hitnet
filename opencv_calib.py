import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((10*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:10].T.reshape(-1,2)*50
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('videos/samples3/right/*.jpg')

first = True
if first:
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,10), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            print(fname)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (7,10), corners2, ret)
            img = cv.resize(img,(960,540))
            cv.imshow('img', img)
            cv.waitKey(500)
        else: 
            cv.drawChessboardCorners(img, (7,10), corners2, ret)
            img = cv.resize(img,(960,540))
            cv.imshow('img', img)
    cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("ret = ", ret)
print("mtx = ", mtx)
print("dist = ", dist)
print("rvecs = ", rvecs)
print("tvecs = ", tvecs)


#ret = 0.45446323665325006
#mtx = [[2.00757763e+03, 0.00000000e+00, 1.89394125e+03], [0.00000000e+00, 2.00667602e+03, 1.08819197e+03], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
#dist = [[ 0.01836286, -0.03718181,  0.00151514, -0.00331427,  0.03112932]]
