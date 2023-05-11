# Note: world coordinate = left rectified camera coordinate

import cv2
import numpy as np

def get_object_points(file_path):
    print("Loading points...")
    file = open(file_path, "r")

    xyzs = []
    rgbs = []

    for point in file:
        xyz = list(point.split(" ")[0:3])
        rgb = list(point.split(" ")[3:6])
        xyzs.append(xyz)
        rgbs.append(rgb)
    
    objectPoints = np.array(xyzs, dtype=np.float32)
    objectRGBs = np.array(rgbs, dtype=np.float32)

    file.close()

    return objectPoints, objectRGBs


def reproject_to_rect(objectPoints, objectRGBs):
    print("Reprojecting to rectified image...")
    rvec = np.eye(3)
    tvec = np.zeros(3)
    distCoeffs = np.zeros(5)
    # cameraMatrix = np.array([[2.01206785e+03, 0.00000000e+00, 1.99553317e+03],
    #                         [0.00000000e+00, 2.01206785e+03, 1.08194806e+03],
    #                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # I don't know why: streoRectify(..., flags=cv2.CALIB_ZERO_DISPARITY) gives correct projection results
    cameraMatrix = np.array([[2.01206785e+03, 0.00000000e+00, 2.03214325e+03],
                            [0.00000000e+00, 2.01206785e+03, 1.08194806e+03],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    
    imagePoints, jacobian = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)

    w,h = 3840, 2160

    projected_img = np.zeros((h, w, 3), np.uint8) # 3840, 2160
    for i in range(objectPoints.shape[0]):
        rgb = objectRGBs[i]
        x,y = imagePoints[i][0]
        
        x = round(x)
        y = round(y)

        if 0 <= x < w and 0 <= y < h:
            for k in range(3):
                projected_img[y][x][k] = rgb[k]
        
    
    projected_img = cv2.cvtColor(projected_img, cv2.COLOR_BGR2RGB)

    return projected_img


def reproject_to_orig(objectPoints, objectRGBs):
    print("Reprojecting to original image...")
    rot = np.array([[ 0.99955591, -0.02153708, -0.02059465],
                    [ 0.02212938,  0.99933505,  0.02897834],
                    [ 0.01995684, -0.02942122,  0.99936786]])
    rvec = np.linalg.inv(rot)
    tvec = np.zeros(3)
    distCoeffs = np.array([[0.008524840829842, -0.005865050351308, 0, 0, 0]])
    cameraMatrix = np.array([[2.013465366250551e+03, 0, 1.917291299561125e+03],
                            [0, 2.011637357030795e+03, 1.080569684355115e+03],
                            [0, 0, 1]])

    imagePoints, jacobian = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)

    w,h = 3840, 2160

    projected_img = np.zeros((h, w, 3), np.uint8) # 3840, 2160
    for i in range(objectPoints.shape[0]):
        rgb = objectRGBs[i]
        x,y = imagePoints[i][0]
        
        x = round(x)
        y = round(y)

        if 0 <= x < w and 0 <= y < h:
            for k in range(3):
                projected_img[y][x][k] = rgb[k]
        
    
    projected_img = cv2.cvtColor(projected_img, cv2.COLOR_BGR2RGB)
    
    return projected_img


def my_project_points(objectPoints, R, T, cameraMatrix, distCoeffs):
    P = cameraMatrix @ (np.concatenate((R,T), axis=1))
    
    imagePoints = []
    for point3d in objectPoints:
        point3d_homo = np.concatenate((point3d,[1.]),axis=0)
        point2d_homo = P @ point3d_homo
        point2d = point2d_homo[:2] / point2d_homo[2]
        imagePoints.append(point2d)
    imagePoints = np.array(imagePoints)

    return imagePoints

def test():
    file_path = "./results/0504/3dpoints4300.txt"
    objectPoints, objectRGBs = get_object_points(file_path)

    R = np.eye(3)
    T = np.array([[0], [0], [0]])
    distCoeffs = np.zeros(5)
    cameraMatrix = np.array([[2.01206785e+03, 0.00000000e+00, 2.03214325e+03],
                            [0.00000000e+00, 2.01206785e+03, 1.08194806e+03],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    imagePoints = my_project_points(objectPoints, R, T, cameraMatrix, distCoeffs)

    w,h = 3840, 2160
    print(objectPoints.shape)
    projected_img = np.zeros((h, w, 3), np.uint8) # 3840, 2160
    for i in range(objectPoints.shape[0]):
        rgb = objectRGBs[i]
        print(imagePoints[i])
        x,y = imagePoints[i][0]

        x = round(x)
        y = round(y)

        if 0 <= x < w and 0 <= y < h:
            for k in range(3):
                projected_img[y][x][k] = rgb[k]
        
    
    projected_img = cv2.cvtColor(projected_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./results/0508/rectified_reprojected.jpg', projected_img)
    return


def main():
    test()
    return

    file_path = "./results/0504/3dpoints4300.txt"
    objectPoints, objectRGBs = get_object_points(file_path)

    projected_img_rect = reproject_to_rect(objectPoints, objectRGBs)
    cv2.imwrite('./results/0508/rectified_reprojected.jpg', projected_img_rect)
    
    projected_img_orig = reproject_to_orig(objectPoints, objectRGBs)
    cv2.imwrite('./results/0508/original_reprojected.jpg', projected_img_orig)

    return

if __name__ == "__main__":
    main()