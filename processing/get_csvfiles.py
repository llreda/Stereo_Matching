import os, os.path
import numpy as np
import glob
import pandas as pd
import argparse


def get_dataset_csv(root, datasets, phase, savedir='.'):
    """
    get a .csv file of the images and depth_gts's path
    :param root: the root dir of stereo_dataset
    :param datasets: the list of selected dataset_{id}.  (id in [1,2,3,4,5,6,7,8,9])
    :param phase: one of ['train', 'validation', 'test')
    :param savedir: the location stored the generated .csv
    :return:
    """
    rec_left = []
    rec_right = []
    rec_gt_left = []
    rec_gt_right = []
    repro_matrixes = []

    for dataset_folder in datasets:
        abs_dataset_folder = os.path.join(root, dataset_folder)
        for keyframe_folder in list(glob.glob('{}/keyframe_*'.format(abs_dataset_folder))):
            if not os.path.exists(os.path.join(keyframe_folder, 'data')):
                continue
            keyframe = os.path.split(keyframe_folder)[1]
            data_folder = os.path.join(keyframe_folder, 'data')
            rec_rgb_folder = os.path.join(data_folder, 'rec_rgb')
            rec_gt_folder = os.path.join(data_folder, 'rec_gt_with_mask')
            repro_matrix = os.path.join(dataset_folder, keyframe, 'Q.npy')
            rgb_left_list = list(glob.glob('{}/Left_Image*'.format(rec_rgb_folder)))
            for left in rgb_left_list:
                id = os.path.splitext(left)[0][-6:]
                right = os.path.join(rec_rgb_folder, 'Right_Image{}.png'.format(id))
                left_gt = os.path.join(rec_gt_folder, 'Left_gt{}.tiff'.format(id))
                right_gt = os.path.join(rec_gt_folder, 'Right_gt{}.tiff'.format(id))

                if os.path.exists(left) and os.path.exists(right) and os.path.exists(left_gt) and os.path.exists(
                        right_gt):
                    left = os.path.join(dataset_folder, keyframe, 'data', 'rec_rgb', f'Left_Image{id}.png')
                    right = os.path.join(dataset_folder, keyframe, 'data', 'rec_rgb', f'Right_Image{id}.png')
                    left_gt = os.path.join(dataset_folder, keyframe, 'data', 'rec_gt_with_mask', f'Left_gt{id}.tiff')
                    right_gt = os.path.join(dataset_folder, keyframe, 'data', 'rec_gt_with_mask', f'Right_gt{id}.tiff')

                    rec_left.append(left)
                    rec_right.append(right)
                    rec_gt_left.append(left_gt)
                    rec_gt_right.append(right_gt)
                    repro_matrixes.append(repro_matrix)
    df = pd.DataFrame({'Rectified_left_image_path': rec_left,
                       'Rectified_right_image_path': rec_right,
                       'Rectified_left_gt_path': rec_gt_left,
                       'Rectified_right_gt_path': rec_gt_right,
                       'Reprojection_matrix_path': repro_matrix
                       })

    df.to_csv('{}/{}_SCARED.csv'.format(savedir, phase))
    # np.savez('./{}_repro_matrixes.npz'.format(phase), repro_matrixes)
    print('........................{}  getting complete..........................'.format(phase))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get_Dateset_Indexing')
    parser.add_argument('--train_sets', default=[7, ],
                        help='select train_sets')
    parser.add_argument('--validation_sets', default=[8, ],
                        help='select validation_sets')
    parser.add_argument('--test_sets', default=[],
                        help='select test_sets')
    parser.add_argument('--savedir', default='./csvfiles', help='the location stored the generated .csv')
    parser.add_argument('--datapath', default='/home/mz/llreda/Stereo_Dataset',
                        help='datapath')
    args = parser.parse_args()
    if not os.path.exists(args.datapath):
        print('Invalid datapath!!!')
        exit()

    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)
    train_list = []
    validation_list = []
    test_list = []
    for id in args.train_sets:
        train_list.append('dataset_{}'.format(id))

    for id in args.validation_sets:
        validation_list.append('dataset_{}'.format(id))

    for id in args.test_sets:
        test_list.append('dataset_{}'.format(id))

    if (len(train_list) > 0):
        get_dataset_csv(args.datapath, train_list, 'train', args.savedir)

    if (len(validation_list)):
        get_dataset_csv(args.datapath, validation_list, 'validation', args.savedir)

    if (len(test_list)):
        get_dataset_csv(args.datapath, test_list, 'test', args.savedir)
