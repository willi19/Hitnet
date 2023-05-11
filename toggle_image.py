import cv2
# 2160, 3840
img1 = cv2.resize(cv2.imread('projected_img_left.png'), (1250, 594)) # 270, 480
img1 = img1[27:567, 0:960]

to_orig = True
w,h = 960, 540
if to_orig:
    img2 = cv2.resize(cv2.imread("results/0508/original.jpg"), (w,h))
    img3 = cv2.resize(cv2.imread("results/0508/original_reprojected.jpg"), (w,h))
else:
    img2 = cv2.resize(cv2.imread("results/0508/rectified.jpg"), (w,h))
    img3 = cv2.resize(cv2.imread("results/0508/rectified_reprojected.jpg"), (w,h))


cv2.imshow('Image', img1)
i=0
while True:
    key = cv2.waitKey(0)

    if(key == ord('q')):
        break

    if(key == ord('t')):
        if i == 0:
            cv2.imshow("Image", img3)
            i=1
        else:
            cv2.imshow("Image", img2)
            i=0


cv2.destroyAllWindows()