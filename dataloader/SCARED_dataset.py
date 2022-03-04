import os, os.path
from torch.utils.data import Dataset
import pandas as pd
import cv2
import tifffile
import numpy as np
from utils import transforms
import logging


class SCARED_Dadaset(Dataset):
    def __init__(self, config,  phase='train'):
        self.datapath = config.datapath
        self.downsampling = config.downsampling
        self.dataset_csv_root = config.dataset_csv_root
        self.phase = phase
        self.valid_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        if phase == 'train':
            self.transform = config.train_transform

        if self.dataset_csv_root is None:
            self.dataset_csv_root = './csvfiles'

        self.dataset_csv_root = os.path.join(self.dataset_csv_root, f'{self.phase}_SCARED.csv')

        if not os.path.exists(self.dataset_csv_root):
            logger = logging.getLogger('filelog')
            logger.error("No Image information .csv!!!")
            exit()

        self.rgbdm_frame = pd.read_csv(self.dataset_csv_root)

    def __len__(self):
        return len(self.rgbdm_frame)

    def __getitem__(self, idx):
        sample = {}
        rec_left_path = os.path.join(self.datapath, self.rgbdm_frame['Rectified_left_image_path'].values[idx])
        rec_right_path = os.path.join(self.datapath, self.rgbdm_frame['Rectified_right_image_path'].values[idx])
        rec_left_gt_path = os.path.join(self.datapath, self.rgbdm_frame['Rectified_left_gt_path'].values[idx])
        # rec_right_gt_path = self.rgbdm_frame['Rectified_right_gt_path'].values[idx]
        reprojection_matrix_left = os.path.join(self.datapath, self.rgbdm_frame['Reprojection_matrix_path'].values[idx])

        sample['left'] = cv2.imread(rec_left_path)
        sample['right'] = cv2.imread(rec_right_path)
        sample['Q'] = np.load(reprojection_matrix_left).astype(np.float32)
        sample['left'] = cv2.resize(sample['left'], (0, 0), fx=1. / self.downsampling, fy=1. / self.downsampling)
        sample['right'] = cv2.resize(sample['right'], (0, 0), fx=1. / self.downsampling, fy=1. / self.downsampling)

        rec_left_gt = tifffile.imread(rec_left_gt_path)
        rec_left_gt = rec_left_gt.astype(np.float32)
        left_z = rec_left_gt[:, :, 2]
        med_left = np.median(left_z[left_z > 0])
        mask_left = (left_z > 5) & (left_z < 5 * med_left)
        sample['depth'] = rec_left_gt
        mask_left = mask_left[..., np.newaxis]
        mask_left = np.repeat(mask_left, 3, axis=2)
        sample['mask'] = mask_left

        if self.phase == 'train':
            if self.transform is not None:
                sample = self.transform(sample)
            else:
                sample = self.valid_transform(sample)
        else:
            sample = self.valid_transform(sample)

        return sample

    # TODO: dataset 和 dataloader 类的本质以及个性化方式