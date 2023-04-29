# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time


class TDD_seg_model(nn.Module):
    def __init__(self, cfg={'Attention_order': "LGLGL",     
       'Channel': [1024, 512, 512, 256, 128, 64], 
       'loss_ratio': [0.5, 0.5, 0.5, 0.8, 0.8, 1]}, n_channels=3):
        super(TDD_seg_model, self).__init__() 
        self.encoder = Encoder(n_channels) 

        self.decoder = nn.ModuleList() 
        self.cfg = cfg
        for i in range(5):
            self.decoder.append(
                DecoderCell(in_channel=cfg['Channel'][i],
                            out_channel=cfg['Channel'][i + 1], 
                            mode=cfg['Attention_order'][i]))
        self.decoder.append(DecoderCell(in_channel=cfg['Channel'][5],
                                        out_channel=1,
                                        mode='C'))
        

    def forward(self, *input):
        if len(input) == 2:
            x = input[0]
            tar = input[1]
            test_mode = False
        if len(input) == 3:
            x = input[0]
            tar = input[1]
            test_mode = input[2]
        if len(input) == 1:
            x = input[0]
            tar = None
            test_mode = True
        en_out = self.encoder(x)
        dec = None
        pred = []
        for i in range(6): 
            dec, _pred = self.decoder[i](en_out[5 - i], dec)
            pred.append(_pred)
        loss = 0  

        if not test_mode:
            for i in range(6): 
                temp_weight = torch.clone(tar).squeeze(1)
                temp_weight[temp_weight == 0] = 0.1
                loss += F.binary_cross_entropy(pred[5 - i].squeeze(1), tar.squeeze(1), weight=temp_weight) * self.cfg['loss_ratio'][5 - i]
                if tar.size()[2] > 28:   
                    tar = F.interpolate(tar, scale_factor=(0.5, 0.5), mode="nearest") 
 
        return pred, loss


def make_layers(cfg, in_channels):
    layers = []
    dilation_flag = False
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'm':
            layers += [nn.MaxPool2d(kernel_size=1, stride=1)]
            dilation_flag = True
        else:
            if not dilation_flag:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

        

class Encoder(nn.Module):
    def __init__(self, n_channels):
        super(Encoder, self,).__init__()
        configure = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'm', 512, 512, 512, 'm']
        self.seq = make_layers(configure, n_channels)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=12, dilation=12)
        self.conv7 = nn.Conv2d(1024, 1024, 3, 1, 1) 

    def forward(self, *input):
        x = input[0]
        conv1 = self.seq[:4](x) 
        conv2 = self.seq[4:9](conv1)
        conv3 = self.seq[9:16](conv2) 
        conv4 = self.seq[16:23](conv3)
        conv5 = self.seq[23:](conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        return conv1, conv2, conv3, conv4, conv5, conv7


class DecoderCell(nn.Module):
    def __init__(self, in_channel, out_channel, mode):
        super(DecoderCell, self).__init__()
        self.bn_en = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=1, padding=0)
        self.mode = mode
        if mode == 'G':
            self.am = GAM(in_channel)
        elif mode == 'L':
            self.am = LAM(in_channel)
        elif mode == 'C':
            self.am = None
        else:
            assert 0
        if not mode == 'C':
            self.conv2 = nn.Conv2d(2 * in_channel, out_channel, kernel_size=1, padding=0)
            self.bn_feature = nn.BatchNorm2d(out_channel)
            self.conv3 = nn.Conv2d(out_channel, 1, kernel_size=1, padding=0)
        else:
            self.conv2 = nn.Conv2d(in_channel, 1, kernel_size=1, padding=0)

    def forward(self, *input):
        assert len(input) <= 2
        if input[1] is None:
            en = input[0] 
            dec = input[0]
        else:
            en = input[0]
            dec = input[1]

        if dec.size()[2] * 2 == en.size()[2]:
            dec = F.interpolate(dec, scale_factor=2, mode='bilinear', align_corners=True)
        elif dec.size()[2] != en.size()[2]:
            assert 0
        en = self.bn_en(en)
        en = F.relu(en)
        fmap = torch.cat((en, dec), dim=1)  # F
        fmap = self.conv1(fmap)
        fmap = F.relu(fmap)
        if not self.mode == 'C':
            fmap_att = self.am(fmap)  # F_att
            x = torch.cat((fmap, fmap_att), 1)
            x = self.conv2(x)
            x = self.bn_feature(x)
            dec_out = F.relu(x)
            _y = self.conv3(dec_out)
            _y = torch.sigmoid(_y)
        else:
            dec_out = self.conv2(fmap)
            _y = torch.sigmoid(dec_out)

        return dec_out, _y


class GAM(nn.Module):
    def __init__(self, in_channel):
        super(GAM, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(in_channel, 1, batch_first=True) 
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(in_channel, in_channel*2, kernel_size=1)

    def forward(self, *input):
        x = input[0]
        shape = x.shape
        qk = self.conv1(x)
        q, k = qk[:,0:self.in_channel, :, :], qk[:,self.in_channel:2*self.in_channel, :, :]

        q = q.flatten(2).permute((0,2,1))
        k = k.flatten(2).permute((0,2,1)) 
        x2 = x.flatten(2).permute((0,2,1))

        attn_output, _ = self.multihead_attn(q, k, x2)
        attn_output = attn_output.permute((0,2,1)) 
        attn_output = attn_output.reshape((attn_output.shape[0], attn_output.shape[1], shape[2], shape[3])) 
        return attn_output 


class LAM(nn.Module):
    def __init__(self, in_channel):
        super(LAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 128, kernel_size=7, dilation=2, padding=6)
        self.conv2 = nn.Conv2d(128, 49, kernel_size=1)

    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = self.conv1(x)
        kernel = self.conv2(kernel)
        kernel = F.softmax(kernel, 1)
        kernel = kernel.reshape(size[0], 1, size[2] * size[3], 7 * 7)
        x = F.unfold(x, kernel_size=[7, 7], dilation=[2, 2], padding=6)
        x = x.reshape(size[0], size[1], size[2] * size[3], -1)
        x = torch.mul(x, kernel)
        x = torch.sum(x, dim=3)
        x = x.reshape(size[0], size[1], size[2], size[3])
        return x