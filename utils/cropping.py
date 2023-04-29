import os
from tqdm import tqdm
import numpy as np
from img_io import read_img, write_img
from skimage.transform import resize

def get_img_name(img_path: str, extension: list):
    img_list = [fn for fn in os.listdir(img_path)
                if any(fn.endswith(ext) for ext in extension)]
    return img_list


def get_img_id(filename: str, split: str = '.'):
    split_index = filename.rfind(split)
    return filename[:split_index]


def create_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def patch_cropping(img_path: str, label_path: str, file_id: str, save_path: str, patch_size: int, stride: int,
                   img_extension: str, label_extension: str, gdal_mode: bool = False, only_crop_image=False):
    img = read_img(img_path=img_path, )
    label = read_img(img_path=label_path,)
    label[label>=1] = 1

    if not only_crop_image:
        assert img.shape[:2] == label.shape
    height, width, bands = img.shape

    f = open(os.path.join(save_path, 'image_id_patch_{}_stride_{}.txt'.format(patch_size, stride)), 'a')
    img_save_path = os.path.join(save_path, 'img_patch_{}_stride_{}'.format(patch_size, stride))
    label_save_path = os.path.join(save_path, 'label_patch_{}_stride_{}'.format(patch_size, stride))
    create_path(path=img_save_path)
    create_path(path=label_save_path)

    patch_id_list = []
    for x in tqdm(range(0, width, stride)): 
        if x + patch_size > width:
            x = width - patch_size
        for y in range(0, height, stride):
            if y + patch_size > height:
                y = height - patch_size

            patch_id = file_id + '_Sx_{}_Sy_{}_Ex_{}_Ey_{}'.format(x, y, x + patch_size, y + patch_size)
            if patch_id in patch_id_list:
                continue
            else:
                if only_crop_image:
                    img_patch = img[y:y + patch_size, x:x + patch_size].copy()
                    write_img(img=img_patch, save_path=os.path.join(img_save_path, patch_id + '.' + img_extension),)
                else:
                    img_patch = img[y:y + patch_size, x:x + patch_size].copy()
                    # img_patch = resize(img_patch, (256, 256))
                    write_img(img=img_patch, save_path=os.path.join(img_save_path, patch_id + '.' + img_extension),)
                    label_patch = label[y:y + patch_size, x:x + patch_size].copy()
                    # label_patch = resize(label_patch, (256, 256)) 
                    write_img(img=label_patch, save_path=os.path.join(label_save_path, patch_id + '.' + label_extension),)
                
                f.write(patch_id)
                f.write("\n")

            patch_id_list.append(patch_id)
    f.close()


def main(config: dict):
    img_folder_path = config['img_folder_path']
    label_folder_path = config['label_folder_path']
    save_path = config['save_path']
    patch_size = config['patch_size']
    stride = config['stride']
    filename_difference = config['filename_difference']
    img_extension = config['img_extension']
    label_extension = config['label_extension']
    gdal_mode = config['gdal_mode']
    only_crop_image = config['only_crop_image']

    if single_image:
        img_path = img_folder_path
        label_path = label_folder_path

        create_path(path=save_path)

        img_id = "image_"

        patch_cropping(img_path=img_path, label_path=label_path, file_id=img_id, save_path=save_path,
                    patch_size=patch_size, stride=stride, img_extension=img_extension,
                    label_extension=label_extension, gdal_mode=gdal_mode, only_crop_image=only_crop_image)
    else:
        img_name_list = get_img_name(img_path=img_folder_path, extension=[img_extension])
        label_name_list = get_img_name(img_path=label_folder_path, extension=[label_extension])

        create_path(path=save_path)

        for i in tqdm(range(len(img_name_list))):
            img_id = get_img_id(img_name_list[i])

            print(img_id)
    
            label_name = img_id.replace(filename_difference['img'], filename_difference['label']) + '.' + label_extension
            # label_name = label_name.replace("E080","E080_UA2012")
            print(label_name)
            assert label_name in label_name_list

            img_path = os.path.join(img_folder_path, img_name_list[i])
            label_path = os.path.join(label_folder_path, label_name)

            patch_cropping(img_path=img_path, label_path=label_path, file_id=img_id, save_path=save_path,
                        patch_size=patch_size, stride=stride, img_extension=img_extension,
                        label_extension=label_extension, gdal_mode=gdal_mode, only_crop_image=only_crop_image)


if __name__ == '__main__':
    single_image = True # crop single image or images in a folder
    only_crop_image = False # crop single image or both image and label
    img_folder_path = r"./data/AVIRIS-1/image.img" # single image path or folder path 
    label_folder_path = './data/AVIRIS-1/label.img' # single label path or folder path 
    save_path = r"./data/AVIRIS-1/training_patches"  # saved path of cropped patches
    patch_size = 50
    stride = 25
    filename_difference = {'img': "", 'label': ""} 
    img_extension = 'tif'  
    label_extension = 'tif'
    gdal_mode = True 

    cfg = dict(
        single_image=single_image,
        only_crop_image = only_crop_image,
        img_folder_path=img_folder_path,
        label_folder_path=label_folder_path,
        save_path=save_path,
        patch_size=patch_size,
        stride=stride,
        filename_difference=filename_difference,
        img_extension=img_extension,
        label_extension=label_extension,
        gdal_mode=gdal_mode
    )

    main(config=cfg)
