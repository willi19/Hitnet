import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from hitnet import HitNet, ModelType, draw_disparity, draw_depth, CameraConfig, load_img
import sys

def get_stereo_params(w, h):
    params = dict()
    K_l = np.array([[2.013465366250551e+03, 0, 1.917291299561125e+03],
                    [0, 2.011637357030795e+03, 1.080569684355115e+03],
                    [0, 0, 1]])
    K_r = np.array([[2.014237350618531e+03, 0, 1.918576988422282e+03],
                    [0, 2.012498352479776e+03, 1.081098668804436e+03],
                    [0, 0, 1]])
    dist_l = np.array([[0.008524840829842, -0.005865050351308, 0, 0, 0]])
    dist_r = np.array([[0.002502340635667, 0.001623048589750, 0, 0, 0]])
    R = np.array([[0.999750826020927, -0.011484952653109, 0.019141100622229],
                [0.010354116792458, 0.998253249358668, 0.058165646306991],
                [-0.019775695586315, -0.057952963750076, 0.998123427165528]])
    T = np.array([-1.735071331350499e+02, 2.144239700218058, 6.785451871316824])
    
    rectify_scale = 1
    rot_l, rot_r, proj_mat_l, proj_mat_r, Q, _, _ = cv2.stereoRectify(K_l, dist_l, K_r, dist_r, (w,h), R, T, rectify_scale, (0,0), flags=0)

    params['K_l'] = K_l
    params['K_r'] = K_r
    params['dist_l'] = dist_l
    params['dist_r'] = dist_r
    params['R'] = R
    params['T'] = T
    params['rot_l'] = rot_l
    params['rot_r'] = rot_r
    params['proj_mat_l'] = proj_mat_l
    params['proj_mat_r'] = proj_mat_r
    params['Q'] = Q

    return params


def stereo_rectify(img_l, img_r):
    print("Rectifying images...")

    # Load calibration results
    K_l = np.array([[2.013465366250551e+03, 0, 1.917291299561125e+03],
                    [0, 2.011637357030795e+03, 1.080569684355115e+03],
                    [0, 0, 1]])
    K_r = np.array([[2.014237350618531e+03, 0, 1.918576988422282e+03],
                    [0, 2.012498352479776e+03, 1.081098668804436e+03],
                    [0, 0, 1]])
    dist_l = np.array([[0.008524840829842, -0.005865050351308, 0, 0, 0]])
    dist_r = np.array([[0.002502340635667, 0.001623048589750, 0, 0, 0]])

    R = np.array([[0.999750826020927, -0.011484952653109, 0.019141100622229],
                [0.010354116792458, 0.998253249358668, 0.058165646306991],
                [-0.019775695586315, -0.057952963750076, 0.998123427165528]])
    T = np.array([-1.735071331350499e+02, 2.144239700218058, 6.785451871316824])

    h, w = img_l.shape[:2]

    # Compute rectification transforms
    rectify_scale = 1
    rot_l, rot_r, proj_mat_l, proj_mat_r, Q, _, _ = cv2.stereoRectify(K_l, dist_l, K_r, dist_r, (w,h), R, T, rectify_scale, (0,0), flags=0)

    # Fix negative baseline
    # if proj_mat_r[0,3] < 0:
    #     proj_mat_r[0,3] *= -1
    #     Q[3,2] *= -1
    #     Q[3,3] *= -1

    # Compute undistortion / rectification map
    stereo_map_l = cv2.initUndistortRectifyMap(K_l, dist_l, rot_l, proj_mat_l, (w,h), cv2.CV_16SC2)
    stereo_map_r = cv2.initUndistortRectifyMap(K_r, dist_r, rot_r, proj_mat_r, (w,h), cv2.CV_16SC2)

    stereo_map_l_x = stereo_map_l[0]
    stereo_map_l_y = stereo_map_l[1]
    stereo_map_r_x = stereo_map_r[0]
    stereo_map_r_y = stereo_map_r[1]

    # Rectify images
    rect_l = cv2.remap(img_l, stereo_map_l_x, stereo_map_l_y, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, stereo_map_r_x, stereo_map_r_y, cv2.INTER_LINEAR)

    return rect_l, rect_r, Q


def get_disparity_map(rect_l, rect_r):
    print("Computing disparity map...")

    model_type = ModelType.middlebury

    if model_type == ModelType.middlebury:
        model_path = "models/middlebury_d400.pb"
    elif model_type == ModelType.flyingthings:
        model_path = "models/flyingthings_finalpass_xl.pb"
    elif model_type == ModelType.eth3d:
        model_path = "models/eth3d.pb"
    
    hitnet_depth = HitNet(model_path, model_type)

    height, width = rect_l.shape[:2]

    rect_l_sm = cv2.resize(rect_l, (int(width/5), int(height/5)))   # TODO: handle general width, height
    rect_r_sm = cv2.resize(rect_r, (int(width/5), int(height/5)))
    # half-size (width: 620..698, height: 555)

    disparity_map_sm = hitnet_depth(rect_l_sm, rect_r_sm)
    disparity_map = cv2.resize(disparity_map_sm, (width,height))

    return disparity_map


def disparity_to_3d(rect_l, disparity_map, Q, save_path):
    print("Converting image to 3d points...")
    points = cv2.reprojectImageTo3D(disparity_map, Q)
    colors = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)

    mask = disparity_map > disparity_map.min()
    out_points = points[mask]
    out_colors = colors[mask]

    output_file = open(save_path,"w")
    for i in range(len(out_points)):
        for v in out_points[i]:
            output_file.write(str(v)+" ")

        for v in out_colors[i]:
            output_file.write(str(v)+" ")
        output_file.write("\n")
    
    return

def main():
    img_num = 7
     # Stereo rectify images
    img_path = f'videos/samples4/obj/left/left{img_num:03d}.jpg'
    img_l = cv2.imread(img_path)
    img_r = cv2.imread(img_path.replace("left", "right"))
    rect_l, rect_r, Q = stereo_rectify(img_l, img_r)

    disparity_map = get_disparity_map(rect_l, rect_r)
    cv2.imwrite(f"results/0511/disparity{img_num:03d}.jpg", disparity_map)

    save_path = f"results/0511/3dpoints{img_num:03d}.txt"
    disparity_to_3d(rect_l, disparity_map, Q, save_path)

    # folder_path = "./results/0511"
    # for img_path in glob(folder_path + "/left*"):
    #     # Get image number
    #     img_num = img_path.split("/")[-1][4:-4]

    #     # Stereo rectify images
    #     img_l = cv2.imread(img_path)
    #     img_r = cv2.imread(img_path.replace("left", "right"))
    #     rect_l, rect_r, Q = stereo_rectify(img_l, img_r)

    #     # Get disparity map
    #     disparity_map = get_disparity_map(rect_l, rect_r)

    #     # Get 3d points from disparity map
    #     save_path = folder_path + "/3dpoints" + img_num + ".txt"
    #     disparity_to_3d(rect_l, disparity_map, Q, save_path)

    #     # # Visualize results
    #     # h, w = img_l.shape[:2]
    #     # h_sm, w_sm = h//4, w//4

    #     # rect_l_sm = cv2.resize(rect_l, (w_sm, h_sm))
    #     # rect_r_sm = cv2.resize(rect_r, (w_sm, h_sm))
    #     # rect_image = np.concatenate((rect_l_sm, rect_r_sm), axis=1)

    #     # # Draw horizontal lines
    #     # l = 20
    #     # x1, x2 = 0, w-1
    #     # for j in range(l):
    #     #     y1 = y2 = int(j*h_sm/(l+1))
    #     #     color = (0, 0, 255)
    #     #     line_thickness = 1
    #     #     cv2.line(rect_image, (x1, y1), (x2, y2), color, thickness=line_thickness)

    #     # cv2.imwrite(folder_path + "/rectified" + img_num + ".jpg", rect_image)
    #     # cv2.imwrite(folder_path + "/disparity" + img_num + ".jpg", disparity_map)



if __name__ == "__main__":
    main()
