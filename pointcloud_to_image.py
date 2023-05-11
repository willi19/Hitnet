import cv2 as cv
import numpy as np
import stereo

# to left image
# parameters
# rvecs = stereo.get_stereo_params... but don't know h, w...

K_l = np.array([[2.013465366250551e+03, 0, 1.917291299561125e+03],
                    [0, 2.011637357030795e+03, 1.080569684355115e+03],
                    [0, 0, 1]])
K_r = np.array([[2.014237350618531e+03, 0, 1.918576988422282e+03],
                    [0, 2.012498352479776e+03, 1.081098668804436e+03],
                    [0, 0, 1]])
proj_mat_l = np.array( [[2.01206785e+03, 0.00000000e+00, 1.99553317e+03],
 [0.00000000e+00, 2.01206785e+03, 1.08194806e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
proj_mat_r = np.array([[ 2.01206785e+03,  0.00000000e+00,  2.06875333e+03], # , -3.49401625e+05
 [ 0.00000000e+00,  2.01206785e+03,  1.08194806e+03],
 [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
dist_l = np.array([[0.008524840829842, -0.005865050351308, 0, 0, 0]])
dist_r = np.array([[0.002502340635667, 0.001623048589750, 0, 0, 0]])
R = np.array([[0.999750826020927, -0.011484952653109, 0.019141100622229],
            [0.010354116792458, 0.998253249358668, 0.058165646306991],
            [-0.019775695586315, -0.057952963750076, 0.998123427165528]])
T = np.array([-1.735071331350499e+02, 2.144239700218058, 6.785451871316824])


rvecs = np.linalg.inv(R)
tvecs = np.zeros(3)
mtx = K_l
dist = dist_l

file = open("point_cloud.txt", "r")
projected_img = np.zeros((2376, 5000, 3), np.uint8) # 3840, 2160

for point in file:
    image = np.asarray(point.split(" ")[0:3])
    image = np.float32(image)

    rgb = np.asarray(point.split(" ")[3:6])
    rgb = np.float32(rgb) 

    imgpts, jac = cv.projectPoints(image, rvecs, tvecs, mtx, dist)

    x = round(imgpts[0][0][0])
    y = round(imgpts[0][0][1])

    for i in range(3):
        projected_img[y][x][i] = rgb[i]
    
    # print("done")
projected_img = cv.cvtColor(projected_img, cv.COLOR_BGR2RGB)
file.close()
#cv.imshow('projected_img', projected_img)
cv.imwrite('projected_img.png', projected_img)


cv.waitKey(0)
cv.destroyAllWindows()