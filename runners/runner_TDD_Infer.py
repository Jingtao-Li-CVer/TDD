import hydra
from runners.base_runner import BaseRunner
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple
from numpy import ndarray as NDArray
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
from albumentations.pytorch import ToTensorV2
from utils.savefig import save_heatmap
from utils.average_meter import AverageMeter
import time
from utils.overlap_infer import overlap_infer 
from metrics import compute_auroc
import albumentations as A
from utils.img_io import read_img, write_img
import os
from tqdm import tqdm 
import time



class runner_TDD_Infer(BaseRunner):
    def _train(self, epoch: int) -> None:
        pass

    def _test(self, epoch: int) -> None:
        self.model.eval()
        save_dir = os.path.join(self.working_dir, "epochs-" + str(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not hasattr(self, 'test_images') or not hasattr(self, 'test_img_gts'):
            self.test_images = []
            self.test_file_names = []
            self.test_patch_sizes = []
            self.test_pad_sizes = []
            self.test_input_sizes = []

            for test_image, test_file_name, test_patch_size, test_pad_size, test_input_size in (self.dataloaders["test"]):
                self.test_images.append(test_image)
                self.test_file_names.append(test_file_name[0])
                self.test_patch_sizes.append(test_patch_size[0])
                self.test_pad_sizes.append(test_pad_size[0])
                self.test_input_sizes.append(test_input_size[0])
            
        for i in tqdm(range(len(self.test_images))):

            mb_img = self.test_images[i]
            file_name = self.test_file_names[i]
            file_name = str(i) + '_' + file_name

            logging.info("test model on the image: " + file_name)

            with torch.no_grad():
                if mb_img.shape[1] > self.cfg.params.training_channels:
                    slice_number = int(mb_img.shape[1]/self.cfg.params.training_channels + 1)
                else:
                    slice_number = 1
                predicted_test_overlap = 0
                for j in range(slice_number):
                    logging.info("test on the " + str(j) + " slice")
                    if j == slice_number - 1:
                        img_t = mb_img[:,-self.cfg.params.training_channels:,:,:]
                    else:
                        img_t = mb_img[:,j:j+self.cfg.params.training_channels,:,:]
                    cfg_test = dict(title_size=[self.test_patch_sizes[i], self.test_patch_sizes[i]],
                    pad_size=[self.test_pad_sizes[i], self.test_pad_sizes[i]], batch_size=1,  
                    padding_mode='mirror', num_classes=1, device=self.cfg.params.device, test_size=self.test_input_sizes[i])  
                    data_overlap_input = img_t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
                    predicted_test_overlap = (predicted_test_overlap + overlap_infer(cfg_test, model=self.model, img=data_overlap_input)['score_map'])/2.0
            
            save_path = os.path.join(self.working_dir, "epochs-" + str(epoch))
            if not os.path.exists(save_path):
                os.makedirs(save_path) 

            anomaly_map = predicted_test_overlap.transpose(2,0,1)[0, :, :] 
            save_ratio = 20

            write_img((anomaly_map), os.path.join(save_path, file_name + '_anomaly_map.tif'))
            save_heatmap(anomaly_map, save_path, save_height=anomaly_map.shape[0]/save_ratio, save_width=anomaly_map.shape[1]/save_ratio ,dpi=150, file_name=file_name) 


    def load(self, save_dir):
        filename = os.path.join(save_dir, 'checkpoint-model.pth')
        self.model.load_state_dict(torch.load(filename, map_location=torch.device(self.cfg.params.device)))