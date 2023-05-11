import cv2
import os

def find_init():
    cap_left = cv2.VideoCapture("/home/vclab/Desktop/Hitnet/videos/samples3/left.MP4")
    cap_right = cv2.VideoCapture("/home/vclab/Desktop/Hitnet/videos/samples3/right.MP4")
    
    #900~2000
    for i in range(500000000):
        retval, frame_left = cap_left.read()
        if not retval:
            break
        frame_left = cv2.resize(frame_left,(960,540))
        if i > 4:
            retval, frame_right = cap_right.read()
            frame_right = cv2.resize(frame_right,(960,540))
        
        if i > 3000:
            print(i)
            cv2.imshow('left',frame_left)
            cv2.imshow('right',frame_right)

        key = cv2.waitKey(0)
        if key == 80:
            break

#3100
#3300
def get_frames(video_path_l, video_path_r, init_l, init_r, save_path, sample_num=50):
    cap_left = cv2.VideoCapture(video_path_l)
    cap_right = cv2.VideoCapture(video_path_r)

    if (not cap_left.isOpened() or not cap_right.isOpened):
        print("Failure opening video")
        return
    
    frame_count_l = 2005#int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))         # 2083
    frame_count_r = 2000#int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))        # 2082

    for _ in range(init_l): cap_left.read()
    for _ in range(init_r): cap_right.read()

    max_count = min(frame_count_l - init_l, frame_count_r - init_r)

    sample_count = 1
    period = 10

    save_path_l = save_path + "/left"
    save_path_r = save_path + "/right"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path_l, exist_ok=True)
    os.makedirs(save_path_r, exist_ok=True)

    for i in range(max_count):
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()
        if not ret_l or not ret_r:
            continue

        if i%period == 0:
            cv2.imwrite(save_path_l + f"/left{sample_count:03d}.jpg", frame_l)
            cv2.imwrite(save_path_r + f"/right{sample_count:03d}.jpg", frame_r)
            sample_count += 1
        
        if sample_count > sample_num:
            break

    return

def get_ith_frames(video_path_l, video_path_r, init_l, init_r, save_path, frame_list):
    cap_left = cv2.VideoCapture(video_path_l)
    cap_right = cv2.VideoCapture(video_path_r)

    for _ in range(init_l): cap_left.read()
    for _ in range(init_r): cap_right.read()

    if (not cap_left.isOpened() or not cap_right.isOpened):
        print("Failure opening video")
        return
    
    save_path_l = save_path + "/left"
    save_path_r = save_path + "/right"
    
    cur_frame = init_l
    frame_list.sort()
    while True:
        if cur_frame > frame_list[-1]:
            break
            
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()

        if not ret_l or not ret_r:
            break
        cur_frame += 1
        print(cur_frame)
        if cur_frame in frame_list:
            cv2.imwrite(save_path_l + f"/left{cur_frame:03d}.jpg", frame_l)
            cv2.imwrite(save_path_r + f"/right{cur_frame:03d}.jpg", frame_r)

    return 


def see_result(left_folder, right_folder):
    left_img_path = os.listdir(left_folder)
    right_img_path = os.listdir(right_folder)

    left_img_path.sort()
    right_img_path.sort()

    for ind, (left_img_name, right_img_name) in enumerate(zip(left_img_path,right_img_path)):
        left_cal_img = cv2.imread(os.path.join(left_folder,left_img_name))
        right_cal_img = cv2.imread(os.path.join(right_folder,right_img_name))

        left_cal_img = cv2.resize(left_cal_img,(960,540))
        right_cal_img = cv2.resize(right_cal_img,(960,540))

        cv2.imshow('left',left_cal_img)
        cv2.imshow('right',right_cal_img)
        print(ind)
        key = cv2.waitKey(0)
        if key == 27:
            break

#remove 5~20, 42~47

def main():
    init_l = 905
    init_r = 900
    file_dir = "/home/vclab/Desktop/Hitnet/videos/samples3"
    video_path_l = file_dir+"/left.MP4"
    video_path_r = file_dir+"/right.MP4"

    save_path = "/home/vclab/Desktop/Hitnet/videos/samples3"
    save_path_obj = save_path+"/obj"
    #see_result(file_dir+"/left",file_dir+"/right")
    #get_frames(video_path_l, video_path_r, init_l, init_r, save_path, sample_num=50)
    #find_init()
    get_ith_frames(video_path_l,video_path_r,init_l, init_r, save_path_obj,[4000,4300])
    return

if __name__ == "__main__":
    main()