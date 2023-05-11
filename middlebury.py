import cv2
import os
from hitnet import HitNet, ModelType, draw_disparity, draw_depth, CameraConfig, load_img
import numpy as np
import matplotlib.pyplot as plt

cam0=[[4396.869, 0, 1353.072], [0, 4396.869, 989.702],[0, 0, 1]]
cam1=[[4396.869, 0, 1538.86],[0, 4396.869, 989.702],[0, 0, 1]]

doffs=185.788 #
baseline=144.049
width=2880
height=1980
ndisp=640
isint=0
vmin=17
vmax=619
dyavg=0
dymax=0

left_img = cv2.imread("left_flower.png")
left_img_sm = cv2.resize(left_img,(width//4,height//4))
right_img = cv2.imread("right_flower.png")
right_img_sm = cv2.resize(right_img,(width//4,height//4))


model_type = ModelType.middlebury

if model_type == ModelType.middlebury:
	model_path = "models/middlebury_d400.pb"
elif model_type == ModelType.flyingthings:
	model_path = "models/flyingthings_finalpass_xl.pb"
elif model_type == ModelType.eth3d:
	model_path = "models/eth3d.pb"


# Initialize model
hitnet_depth = HitNet(model_path, model_type)
disparity_map_sm = hitnet_depth(left_img_sm, right_img_sm)

plt.imshow(disparity_map_sm,'gray')
plt.show()
		
disparity_map = cv2.resize(disparity_map_sm,(width,height))

Q = np.float32([[ 1 , 0  ,0, -cam0[0][2]],
 [ 0 , 1,  0 ,-cam0[1][2]],
 [ 0 , 0 , 0,  cam0[0][0]],
 [ 0, 0  , -1/baseline ,-doffs/baseline]])

points = cv2.reprojectImageTo3D(disparity_map, Q)
colors = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)


mask = disparity_map > disparity_map.min()
out_points = points[mask]
out_colors = colors[mask]

output_file = open("point_cloud.txt","w")
for i in range(len(out_points)):
    for v in out_points[i]:
        output_file.write(str(v)+" ")
	
    for v in out_colors[i]:
        output_file.write(str(v)+" ")
    output_file.write("\n")





