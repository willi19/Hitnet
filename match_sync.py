import cv2
import os

def find_init(video_path_l, video_path_r, save_path):
    sync_path = save_path + "/sync"
    os.makedirs(sync_path, exist_ok=True)
    cap_left = cv2.VideoCapture(video_path_l)
    cap_right = cv2.VideoCapture(video_path_r)

    for _ in range(200):
        cap_left.read()
        cap_right.read()

    for i in range(100):
        retval, frame_left = cap_left.read()
        if not retval:
            break
        #print(frame_left.shape)
        frame_left = cv2.resize(frame_left,(960,540))
        cv2.imwrite(sync_path + f"/left{i:03d}.png", frame_left)

        retval, frame_right = cap_right.read()
        frame_right = cv2.resize(frame_right,(960,540))
        cv2.imwrite(sync_path + f"/right{i:03d}.png", frame_right)


        key = cv2.waitKey(0)
        if key == 27:
            break


def get_frames(video_path_l, video_path_r, init_l, init_r, save_path, sample_num=20):
    cap_left = cv2.VideoCapture(video_path_l)
    cap_right = cv2.VideoCapture(video_path_r)

    if (not cap_left.isOpened() or not cap_right.isOpened):
        print("Failure opening video")
        return
    
    frame_count_l = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))         # 4408
    frame_count_r = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))        # 4409

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

def main():
    init_l = 2000
    init_r = 2001
    video_path_l = "/home/vclab/Desktop/Hitnet/videos/left4.MP4"
    video_path_r = "/home/vclab/Desktop/Hitnet/videos/right4.MP4"
    save_path = "/home/vclab/Desktop/Hitnet/videos/samples4/obj"
    #find_init(video_path_l, video_path_r, save_path)
    get_frames(video_path_l, video_path_r, init_l, init_r, save_path, sample_num=200)

    return

if __name__ == "__main__":
    main()