import os, os.path
import shutil
import glob
import tarfile
from multiprocessing import Process, Queue
import tifffile
import cv2
import math
import numpy as np
import argparse


def move(source, target_root):
    keyframe_folder = os.path.split(source)[-1]
    dataset_folder = os.path.split(os.path.split(source)[0])[-1]
    target = os.path.join(target_root, dataset_folder, keyframe_folder)
    shutil.copytree(source, target)


def unzip(keyframe_folder):
    data_folder = os.path.join(keyframe_folder, 'data')
    if(not os.path.exists(data_folder)):
        return
    for file in os.listdir(data_folder):
        file = os.path.join(data_folder, file)
        if os.path.splitext(file)[-1] == '.gz':
            tar = tarfile.open(file, 'r')
            tmp = os.path.splitext(file)[0]
            tar.extractall(os.path.splitext(tmp)[0])
            tar.close()


def cut(video_file):
    cap = cv2.VideoCapture(video_file)
    isOpened = cap.isOpened
    single_pic_store_dir = os.path.splitext(video_file)[0]
    # 如果已经处理过则不再处理
    if os.path.exists(single_pic_store_dir):
        return
    else:
        os.mkdir(single_pic_store_dir)
    i = 0
    while isOpened:
        (flag, frame) = cap.read()  # 读取一张图像
        if (flag == True):
            fileName = ('Image{:0>6d}.png').format(i)
            save_path = os.path.join(single_pic_store_dir, fileName)
            cv2.imwrite(save_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            i += 1
        else:
            break
    return single_pic_store_dir


def extraction_per_process(keyframe_folder):
    data_folder = os.path.join(keyframe_folder, 'data')
    if os.path.exists(data_folder):
        video_dir = os.path.join(data_folder, 'rgb.mp4')
        if os.path.exists(video_dir):
            cut(video_dir)


def get_parameters(Calibration_parameter_path):
    if os.path.exists(Calibration_parameter_path):
        f = cv2.FileStorage(Calibration_parameter_path, cv2.FileStorage_READ)
        R = np.array(f.getNode('R').mat()).astype(float)
        T = np.array(f.getNode('T').mat()).astype(float)
        M1 = np.array(f.getNode('M1').mat()).astype(float)
        D1 = np.array(f.getNode('D1').mat()).astype(float)
        M2 = np.array(f.getNode('M2').mat()).astype(float)
        D2 = np.array(f.getNode('D2').mat()).astype(float)
    return R, T, M1, D1, M2, D2


def stereo_rectify(R, T, M1, D1, M2, D2, Image_size):
    mtype = cv2.CV_32FC1
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(M1, np.squeeze(D1), M2, np.squeeze(D2), Image_size, R, np.squeeze(T), flags=cv2.CALIB_ZERO_DISPARITY, alpha= 0)
    mapLx, mapLy = cv2.initUndistortRectifyMap(M1, D1, R1, P1, Image_size, mtype)
    mapRx, mapRy = cv2.initUndistortRectifyMap(M2, D2, R2, P2, Image_size, mtype)

    return R1, R2, P1, P2, Q, mapLx, mapLy, mapRx, mapRy


def remap(src, mapLx, mapLy, R):
    size = src.shape
    assert (len(size) == 3)
    src_2 = src[:, :, 2]
    mask = (src_2 > 5) & (src_2 < 100)  # 取z轴坐标大于5的为有效位置，无效位置都是填充的0.
    mask = np.expand_dims(mask, axis=2)

    mapLx_floor = np.floor(mapLx).astype(np.int16)   # mapLx和mapLy分别是坐标映射的x和y坐标，用I和Irec表示原图和极线校正的结果，那么对于Irec中(y, x)位置，它在原图中的位置为(mapLy(y,x),mapLx(y,x))
    mapLx_ceil = np.ceil(mapLx).astype(np.int16)   # 往往map计算的映射坐标都是小数，所以有四种取整组合，对于这四点，进行双线性插值
    mapLy_floor = np.floor(mapLy).astype(np.int16)
    mapLy_ceil = np.ceil(mapLy).astype(np.int16)

    loc1 = (mapLy_floor, mapLx_floor)
    loc2 = (mapLy_floor, mapLx_ceil)
    loc3 = (mapLy_ceil, mapLx_floor)
    loc4 = (mapLy_ceil, mapLx_ceil)

    mask1 = mask[loc1].astype(np.float32)
    mask2 = mask[loc2].astype(np.float32)
    mask3 = mask[loc3].astype(np.float32)
    mask4 = mask[loc4].astype(np.float32)

    weight1 = np.expand_dims((1 - (mapLy - loc1[0])) * (1 - (mapLx - loc1[1])), axis=2) * mask1
    weight2 = np.expand_dims((1 - (mapLy - loc2[0])) * (1 - (loc2[1] - mapLx)), axis=2) * mask2
    weight3 = np.expand_dims((1 - (loc3[0] - mapLy)) * (1 - (mapLx - loc3[1])), axis=2) * mask3
    weight4 = np.expand_dims((1 - (loc4[0] - mapLy)) * (1 - (loc4[1] - mapLx)), axis=2) * mask4
    valid = ((mask1 + mask2 + mask3 + mask4) > 1).astype(np.float32)   # 当原图中对应位置周围有两个有效点时就认为该位置是可以有效填充的

    weight_sum = weight1 + weight2 + weight3 + weight4
    weight_sum = np.where(valid, weight_sum, 0.000001)    # 防止分母为0
    res = (src[loc1] * weight1 + src[loc2] * weight2 + src[loc3] * weight3 + src[loc4] * weight4) / weight_sum * valid

    h, w, c = res.shape
    res = res.reshape(-1, 3).T
    warped = np.matmul(R, res)           # 变换到理想坐标系下
    warped = warped.T.reshape(h, w, 3).astype(np.float32)

    return warped


def rectify_image(keyframe_folder):
    data_folder = os.path.join(keyframe_folder, 'data')
    if not os.path.exists(data_folder):
        return
    rgb_source = os.path.join(data_folder, 'rgb')
    rgb_new = os.path.join(data_folder, 'rec_rgb')
    if not os.path.exists(rgb_new):
        os.mkdir(rgb_new)
    gt_source = os.path.join(data_folder, 'scene_points')
    if not os.path.exists(gt_source):
        return
    gt_new = os.path.join(data_folder, 'rec_gt_with_mask')
    if not os.path.exists(gt_new):
        os.mkdir(gt_new)
    rgb_list = os.listdir(rgb_source)
    gt_list = os.listdir(gt_source)
    temp = os.path.join(gt_source, gt_list[0])
    frame = tifffile.imread(temp)
    h, w, c = frame.shape
    Image_size = (w, h // 2)
    Calibration_parameter_path = os.path.join(keyframe_folder, 'endoscope_calibration.yaml')
    R, T, M1, D1, M2, D2 = get_parameters(Calibration_parameter_path)
    R1, R2, P1, P2, Q, mapLx, mapLy, mapRx, mapRy = stereo_rectify(R, T, M1, D1, M2, D2, Image_size)
    for name in rgb_list:
        name = os.path.join(rgb_source, name)
        frame = cv2.imread(name)
        left = frame[0:h // 2, :, :]
        right = frame[h // 2:, :, :]
        rec_left = cv2.remap(left, mapLx, mapLy, cv2.INTER_LINEAR)
        rec_right = cv2.remap(right, mapRx, mapRy, cv2.INTER_LINEAR)
        id = os.path.splitext(name)[0][-6:]
        rec_left_path = os.path.join(rgb_new, 'Left_Image{}.png'.format(id))
        rec_right_path = os.path.join(rgb_new, 'Right_Image{}.png'.format(id))
        cv2.imwrite(rec_left_path, rec_left, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(rec_right_path, rec_right, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    for name in gt_list:
        name = os.path.join(gt_source, name)
        scene_points = tifffile.imread(name)
        left_gt = scene_points[0:h // 2, :, :]
        right_gt = scene_points[h // 2:, :, :]
        rec_left_gt = remap(left_gt, mapLx, mapLy, R1)
        rec_right_gt = remap(right_gt, mapRx, mapRy, R2)

        id = os.path.splitext(name)[0][-6:]
        rec_left_gt_path = os.path.join(gt_new, 'Left_gt{}.tiff'.format(id))
        rec_right_gt_path = os.path.join(gt_new, 'Right_gt{}.tiff'.format(id))

        tifffile.imsave(rec_left_gt_path, rec_left_gt)
        tifffile.imsave(rec_right_gt_path, rec_right_gt)

    P1_path = os.path.join(keyframe_folder, 'P1.npy')
    P2_path = os.path.join(keyframe_folder, 'P2.npy')
    Q_path = os.path.join(keyframe_folder, 'Q.npy')
    R1_path = os.path.join(keyframe_folder, 'R1.npy')
    R2_path = os.path.join(keyframe_folder, 'R2.npy')
    np.save(P1_path, P1)
    np.save(P2_path, P2)
    np.save(Q_path, Q)
    np.save(R1_path, R1)
    np.save(R2_path, R2)
    f = P1[0][0]  # 焦距和基线
    b = np.linalg.norm(T)
    baseline_and_focallength = np.array([b, f])
    baseline_and_focallength_path = os.path.join(keyframe_folder, 'baseline_and_focallength.npy')
    np.save(baseline_and_focallength_path, baseline_and_focallength)



def unzip_dataset(keyframe_list):
    workers = len(keyframe_list)
    process_pool = []
    for i in range(workers):
        process_pool.append(Process(target=unzip, args=(keyframe_list[i],)))
    print("...........................start to unzip files.....................")
    for t in process_pool:
        t.start()

    count = 0
    for t in process_pool:
        print("Waiting for {:d}th process to complete".format(count))
        count += 1
        while t.is_alive():
            t.join(timeout=1)
    print(".............................unziping completed...............................")


def video_to_image(keyframe_list):
    process_pool = []
    workers = len(keyframe_list)
    for i in range(workers):
        process_pool.append(
            Process(target=extraction_per_process, args=(keyframe_list[i],)))
    print(".................................start to extract frame from videos....................................")
    for t in process_pool:
        t.start()
    count = 0
    for t in process_pool:
        print("Waiting for {:d}th process to complete".format(count))
        count += 1
        while t.is_alive():
            t.join(timeout=1)
    print("................................all videos have transformed to images.................................")


def rectify_dataset(keyframe_list):
    process_pool = []
    workers = len(keyframe_list)
    for i in range(workers):
        process_pool.append(
            Process(target=rectify_image, args=(keyframe_list[i],)))
    print(".................................start to rectify images....................................")
    for t in process_pool:
        t.start()
    count = 0
    for t in process_pool:
        print("Waiting for {:d}th process to complete".format(count))
        count += 1
        while t.is_alive():
            t.join(timeout=1)
    print(".................................all images have been rectified.................................")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset_preprocessing')
    parser.add_argument('--datapath', default='/home/mz/llreda/Stereo_Dataset',
                    help='datapath')
    args = parser.parse_args()
    target_root = args.datapath
    dataset_list = []
    for dataset in list(glob.glob('{}/dataset_*'.format(target_root))):
        dataset_list.append(dataset)

    keyframe_list = []
    for dataset in dataset_list:
        for keyframe_folder in list(glob.glob('{}/keyframe_*'.format(dataset))):
            if os.path.splitext(keyframe_folder)[-1] != '.zip':
                keyframe_list.append(keyframe_folder)

    for keyframe in keyframe_list:
        print(keyframe)    # keyframe folder as the processing unit

    unzip_dataset(keyframe_list)
    video_to_image(keyframe_list)
    rectify_dataset(keyframe_list)

