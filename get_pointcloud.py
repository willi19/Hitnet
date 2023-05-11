import cv2
import sys
from hitnet import HitNet, ModelType, draw_disparity, draw_depth, CameraConfig, load_img
import numpy as np
import matplotlib.pyplot as plt


model_type = ModelType.middlebury

if model_type == ModelType.middlebury:
	model_path = "models/middlebury_d400.pb"
elif model_type == ModelType.flyingthings:
	model_path = "models/flyingthings_finalpass_xl.pb"
elif model_type == ModelType.eth3d:
	model_path = "models/eth3d.pb"

hitnet_depth = HitNet(model_path, model_type)

# focal length, baseline, image path
f = 2011.1457382023004 #pixel
b = 173.80509425711213 #mm

cam_left = [[2.013465366250551e+03,0,1.917291299561125e+03],
            [0,2.011637357030795e+03,1.080569684355115e+03],
            [0,0,1]]
cam_right = [[2.014237350618531e+03,0,1.918576988422282e+03],
             [0,2.012498352479776e+03,1.081098668804436e+03],
             [0,0,1]]

height = 2160
width = 3840
doffs = 185.788 

left_img = cv2.imread("videos/samples3/obj/rectified/left4300.jpg")
left_img_sm = cv2.resize(left_img,(int(width/5),int(height/5)))
right_img = cv2.imread("videos/samples3/obj/rectified/right4300.jpg")
right_img_sm = cv2.resize(right_img,(int(width/5),int(height/5)))
#half-size (width: 620..698, height: 555)

cv2.imshow('rect_l_sm', left_img_sm)
cv2.waitKey()
cv2.destroyAllWindows()

# sys.exit(0)

disparity_map_sm = hitnet_depth(left_img_sm, right_img_sm)

plt.imshow(disparity_map_sm,'gray')
plt.show()

disparity_map = cv2.resize(disparity_map_sm,(width,height))

# Q = np.float32([[ 1.00000000e+00 , 0.00000000e+00  ,0.00000000e+00, -2.03214325e+03],
#  [ 0.00000000e+00 , 1.00000000e+00,  0.00000000e+00 ,-1.08194806e+03],
#  [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00,  2.01206785e+03],
#  [ 0.00000000e+00 , 0.00000000e+00  ,-5.75861047e-03 ,-doffs/b]])

Q = np.float32([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00 ,-2.02307072e+03],
 [ 0.00000000e+00 , 1.00000000e+00 , 0.00000000e+00, -1.07804686e+03],
 [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 2.01206785e+03],
 [ 0.00000000e+00 , 0.00000000e+00  ,-5.75861047e-03  ,3.38754197e-03]])

points = cv2.reprojectImageTo3D(disparity_map, Q)
colors = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)

print(left_img.shape)

mask = disparity_map > disparity_map.min()
out_points = points[mask]
out_colors = colors[mask]

# output_file = open("new_point_cloud.txt","w")
# for i in range(len(out_points)):
#     for v in out_points[i]:
#         output_file.write(str(v)+" ")
	
#     for v in out_colors[i]:
#         output_file.write(str(v)+" ")
#     output_file.write("\n")





