# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 20:50:01 2021

@author: User
"""
#%%特徵可視化
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch
import time
from efficientnet_pytorch import EfficientNet

import torch.nn as nn
import torch.nn.functional as F 
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet
from efficientnet_pytorch.utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)
from collections import OrderedDict
from efficientnet_pytorch import EfficientNet



def show_feature_map(img_src, conv_features,i,location):
    '''
    img_src: img file
    conv_feature: convolution output
    location: save file
    '''
    img = Image.open(img_file).convert('RGB')
    height, width = img.size
    heat = conv_features.squeeze(0)
    heat_mean = torch.mean(heat,dim=0)
    heatmap = heat_mean.cpu().numpy()
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap,(img.size[0],img.size[1]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    plt.imshow(heatmap)
    plt.show()
    superimg = heatmap*0.4+np.array(img)[:,:,::-1] 
    cv2.imwrite('./superimg.jpg',superimg)
    i = str(i)
    img_ = np.array(Image.open('./superimg.jpg').convert('RGB'))
    plt.imshow(img_)
    plt.show()
    plt.savefig('D' +  '/' + i +".jpg", dpi=500)

class SaveConvFeatures(): 
    
    def __init__(self, m): # module to hook
        self.hook = m.register_forward_hook(self.hook_fn) 
    def hook_fn(self, module, input, output): 
        self.features = output.data 
    def remove(self):
        self.hook.remove()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
t = transforms.Compose([transforms.Resize((256, 256)),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        ]
                       )

img_file = r".jpg"            # img file
img = Image.open(img_file)
img = t(img).unsqueeze(0).to(device)
model = torch.load('.pkl',map_location='cpu')      # Ures model file     
custom_model_RAFDB = model
custom_model = model
custom_model_affectnet = custom_model.cuda()
print(custom_model_RAFDB)

for i in range(0,2):
    hook_ref = SaveConvFeatures(custom_model_affectnet._conv_head)                  #desired layer
    print(hook_ref)
    custom_model_affectnet(img)    
    conv_features = hook_ref.features 
    print('Feature map output dimension：', conv_features.shape) 
    hook_ref.remove()
    show_feature_map(img_file, conv_features,i,"Save_file")                          # save feature map 