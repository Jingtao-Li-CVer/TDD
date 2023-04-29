from typing import List
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy import ndarray as NDArray
import seaborn as sns
from tqdm import tqdm
import logging 
import cv2
 

def save_heatmap(data, working_dir, save_width, save_height, dpi=300, file_name='heatmap.png'):
    save_path = os.path.join(working_dir, file_name + "_heatmap.png")
    plt.figure(figsize=(save_width, save_height))
    sns.heatmap(data, cmap="jet", cbar=False) #热图
    plt.show()
    plt.axis('off')
    plt.savefig(save_path, dpi=dpi, bbox_inches = 'tight') 
    plt.close()