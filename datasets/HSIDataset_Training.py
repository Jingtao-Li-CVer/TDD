import logging
import os
import sys
sys.path.append("./../")
import torch
from skimage import io
from torch.utils.data import Dataset
from utils.img_io import read_img, write_img
import numpy as np
from albumentations.pytorch import ToTensorV2
from random import choice
from skimage.transform import resize
import albumentations as A
import random


class HSIDataset_Training(Dataset): 
    def __init__(self, img_dir, max_height, min_height, transforms):
        self.img_dir = img_dir
        self.transform = transforms

        self.img_files = os.listdir(self.img_dir)

        self.dataset_length = len(os.listdir(self.img_dir)) 
        self.location_selection = A.Compose([     
                A.CoarseDropout(max_holes=1 , min_holes=1, max_height=max_height, max_width=max_height, min_height=min_height, min_width=min_height, fill_value=1, mask_fill_value=1, p=1.0)
            ])
  
        self.shift_transform =A.Compose([A.ShiftScaleRotate(shift_limit=0.0, p=1), A.IAAPiecewiseAffine(scale=(0.1, 0.3), p=1.0, order=0)])
        self.shuffle_transform = A.ChannelShuffle(p=1.0)

        logging.info(f'Creating dataset with {self.dataset_length} examples') 

    def __len__(self): 
        return self.dataset_length 

    def __getitem__(self, i):
 
        file_name = self.img_files[i] 
             
        img_path = os.path.join(self.img_dir, file_name)

        img = read_img(img_path=img_path).astype(np.float32)
        mask = np.zeros((img.shape[0], img.shape[1])) 

        sample = self.location_selection(image=img, mask = mask)

        # Strategy 1: Channel shuffling
        img2 = self.shuffle_transform(image=img)['image']
        
        # # Strategy 2: Adding Gaussian Noise
        # img_noise = np.random.normal(0, 1, img.shape)
        # img2 = img + img_noise
        # max_values = img2.max((0,1))
        # max_values[max_values == 0] = 1
        # img2 = img2/max_values         

        # # Strategy 3: Adding Uniform Noise
        # img_noise = np.random.rand(img.shape[0], img.shape[1], img.shape[2])
        # img2 = img + img_noise
        # max_values = img2.max((0,1))
        # max_values[max_values == 0] = 1
        # img2 = img2/max_values     
        
        locs = np.where(sample['mask'] == 1)
        sample['image'][locs[0], locs[1], :] = img2[locs[0], locs[1], :] 
        sample = self.shift_transform(image=sample['image'], mask = sample['mask']) 
        img = sample['image']
        mask = sample['mask'] 

        sample = self.transform(image=img, mask = mask) 

        return sample['image'].float(), sample['mask'].float()
