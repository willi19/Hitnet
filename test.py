import cv2

w,h = 960, 540
img1 = cv2.resize(cv2.imread("results/0508/rectified.jpg"), (w,h))
img2 = cv2.resize(cv2.imread("results/0508/reprojected_to_rect.jpg"), (w,h))
size = (h,w)

fps = 5
toggle_num = 50
N = 1
frame_array = []
for i in range(toggle_num):
    for _ in range(N):
        frame_array.append(img1)
    for _ in range(N):
        frame_array.append(img2)

pathOut = './results/0508/toggle_rect.MP4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
out = cv2.VideoWriter(pathOut,fourcc, fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
cv2.destroyAllWindows()
out.release()




