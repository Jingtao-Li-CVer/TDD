import logging
import numpy as np
from pytorch_toolbelt.inference import tta
import torch
import albumentations as A
from tqdm import tqdm 



def pad_borders(img, tile_size, padding_mode):
    assert padding_mode in ['constant', 'mirror']
    image_h, image_w = img.shape[:2]
    new_h, new_w = (image_h // tile_size[0] + 1) * tile_size[0], (image_w // tile_size[1] + 1) * tile_size[1]
    if padding_mode == 'constant':
        pad_img = np.pad(img, ((0, new_h - image_h), (0, new_w - image_w), (0, 0)), 'constant')
    else:
        pad_img = np.pad(img, ((0, new_h - image_h), (0, new_w - image_w), (0, 0)), 'reflect')
    return pad_img


def pad_test_img(img, pad_size, padding_mode):
    assert padding_mode in ['constant', 'mirror']
    if padding_mode == 'constant':
        pad_img = np.pad(img, ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (0, 0)), 'constant')
    else:
        pad_img = np.pad(img, ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (0, 0)), 'reflect')
    return pad_img


min_ratio = 10
def batch_predict(img_list, model, device, test_size):
    global min_ratio 
    with torch.no_grad():
        resize_transform = A.Resize(height=int(test_size), width=int(test_size), p=1.0) 
        resized_imgs = [resize_transform(image = img)['image'] for img in img_list]
        img_list_after_transforms = [torch.unsqueeze(torch.from_numpy(img).permute(2, 0, 1), dim=0) for img in resized_imgs]
        minibatch = torch.cat(img_list_after_transforms, dim=0)  # [batchsize,c,h,w]
        output, _ = model(minibatch.to(device))  #tta.d4_image2mask(model, minibatch.to(device))
        resize_transform2 = A.Resize(height=img_list[0].shape[0], width=img_list[0].shape[0], p=1.0) 
        resized_img = resize_transform2(image=output[5].permute((2,3,0,1)).detach().cpu().numpy()[:,:,0,0])['image']
        resized_img = torch.from_numpy(resized_img).to(device=device).unsqueeze(0).unsqueeze(1)
        output_prob = resized_img 

    result = dict()
    result['label_map'] = torch.argmax(output_prob, dim=1).cpu().numpy()
    result['score_map'] = output_prob.permute(0, 2, 3, 1).cpu().numpy()
    return result


def infer(image, model, cfg):
    tile_size = cfg['title_size']
    batch_size = cfg['batch_size']
    pad_size = cfg['pad_size']
    padding_mode = cfg['padding_mode']
    num_classes = cfg['num_classes']
    device = cfg['device']
    test_size = cfg['test_size'] 
    height, width, channel = image.shape
    pos_list = list()
    image_tile_list = list()

    # crop the padding image into tile pieces
    padded_image = pad_test_img(image, pad_size, padding_mode)
    padded_height, padded_width, _ = padded_image.shape
    #print(padded_image.shape)
    for h_id in range(0, height // tile_size[1]):
        for w_id in range(0, width // tile_size[0]):
            left = w_id * tile_size[0]
            upper = h_id * tile_size[1]
            right = min(left + tile_size[0] + pad_size[0] * 2, padded_width)
            lower = min(upper + tile_size[1] + pad_size[1] * 2, padded_height)
            image_tile = padded_image[upper:lower, left:right, :]
            image_tile_list.append(image_tile)
            pos_list.append([left, upper, right, lower])

    # predict
    label_map = np.zeros((height, width), dtype=np.uint8)
    score_map = np.zeros((height, width, num_classes), dtype=np.float32)
    num_tiles = len(image_tile_list)
    for i in tqdm(range(0, num_tiles, batch_size)):  
        begin = i
        end = min(i + batch_size, num_tiles)
        res = batch_predict(img_list=image_tile_list[begin:end], model=model, device=device, test_size=test_size)
        for j in range(begin, end):
            left, upper, right, lower = pos_list[j]
            # tile_label_map = res[j - begin]
            tile_label_map = res["label_map"][j - begin]  # [batchsize,h,w]->[h,w]
            tile_score_map = res["score_map"][j - begin]  # [batchsize,h,w,classnum]->[h,w,c]
            #print(tile_label_map.shape)
            #print(tile_score_map.shape)
            tile_upper = pad_size[1]
            tile_lower = tile_label_map.shape[0] - pad_size[1]
            tile_left = pad_size[0]
            tile_right = tile_label_map.shape[1] - pad_size[0]
            label_map[upper:lower - 2 * pad_size[0], left:right - 2 * pad_size[1]] = \
                tile_label_map[tile_upper:tile_lower, tile_left:tile_right]
            score_map[upper:lower - 2 * pad_size[0], left:right - 2 * pad_size[1], :] = \
                tile_score_map[tile_upper:tile_lower, tile_left:tile_right, :]

    result = {"label_map": label_map, "score_map": score_map}
    return result


def overlap_infer(config_test, model, img):
    title_size = config_test['title_size'] 
    padding_mode = config_test['padding_mode']
    height, width = img.shape[:2]
    padded_img = pad_borders(img, title_size, padding_mode)
    #print(padded_img.shape)
    result_img = infer(image=padded_img, model=model, cfg=config_test)
    result = {"label_map": result_img["label_map"][:height, :width],
              "score_map": result_img["score_map"][:height, :width, :]}
    return result