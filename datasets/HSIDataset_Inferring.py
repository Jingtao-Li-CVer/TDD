import logging
import os
import sys
sys.path.append("./../")
import torch
from skimage import io
from torch.utils.data import Dataset
from utils.img_io import read_img, write_img
from skimage.transform import resize
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A



class HSIDataset_Inferring(Dataset):
    def __init__(self, img_dir, mask_dir, test_input_sizes, test_patch_sizes, test_pad_sizes, training_channels, normalize, transforms):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.test_patch_sizes = test_patch_sizes
        self.training_channels = training_channels
        self.test_pad_sizes = test_pad_sizes
        self.normalize = normalize
        self.test_input_size = test_input_sizes
        self.transform = transforms

        self.dataset_length = len((self.img_dir))

        logging.info(f'Creating dataset with {self.dataset_length} examples')

    def __len__(self):
        return self.dataset_length
 
    def __getitem__(self, i):
        img_path = self.img_dir[i]
        label_path = self.mask_dir[i]
        test_patch_size = self.test_patch_sizes[i]
        test_pad_size = self.test_pad_sizes[i]
        test_input_size = self.test_input_size[i]

        (_, img_file_name) = os.path.split(img_path)
        img_file_name = img_file_name.split('.')[-2]
        if 'River' in img_path:
            img = read_img(img_path=img_path)
        else:
            img = read_img(img_path=img_path).astype(np.float32)

        if img.shape[2] < self.training_channels:
            img = resize(img, (img.shape[0], img.shape[1], self.training_channels), order=1)

        if self.normalize[i]:
            img = img/img.max((0,1))

        mask = read_img(label_path).astype(np.uint8)
        mask[mask >= 1] = 1
        mask  = 1 - mask

        if 'HYDICE' in img_path:
            sample = self.transform(image=img, mask = mask)
        else:
            t2 = A.Compose(self.transform[1:])
            sample = t2(image=img, mask = mask)

        return sample['image'].float(), sample['mask'].float(), img_file_name, test_patch_size, test_pad_size, test_input_size
