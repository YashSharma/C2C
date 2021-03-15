import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import skimage.io as io

import os
import cv2
import copy
import umap
import random
import pandas as pd
from tqdm import tqdm
import openslide as opsl
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

import albumentations
from albumentations.pytorch import ToTensorV2, ToTensor

NUM_CLUSTER = 8
NUM_IMG_PER_CLUSTER = 8
CLUSTER_BATCH_SIZE = 128 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sample_path(li, bpath):
    
    img_name = list(set([x.split('__')[0].split('_')[0] for x in li]))

#     select_img = pd.read_csv('checkpoint/celiac_valid_checkpoint_at_img.csv')['img_name'].tolist()
#     valid_name = [x for x in img_name if x in select_img]

    valid_name = random.sample(img_name, int(0.2*len(img_name)))
    train_name = list(set(img_name) - set(valid_name))
        
    train_path = [os.path.join(bpath, x) for x in li if x.split('__')[0].split('_')[0] in train_name]
    valid_path = [os.path.join(bpath, x) for x in li if x.split('__')[0].split('_')[0] in valid_name]
    
    return train_path, valid_path

def convert_list_to_nested_list(image_list):
    sample_df = pd.DataFrame({'patch_name': image_list})
    sample_df['img_name'] = sample_df['patch_name'].apply(lambda x: x.split('/')[-1].split('__')[0])
    return list(sample_df.groupby('img_name')['patch_name'].apply(list))

if __name__=="__main__":
    
    
    base_path = '/project/GutIntelligenceLab/ys5hd/MSDS/images_512x512_non_resized/threshold_0.5/'
    celiac_path = os.path.join(base_path, 'train/Celiac')
    normal_path = os.path.join(base_path, 'train/Normal')    
    
    # Extract Celiac
    celiac_train, celiac_valid = sample_path(os.listdir(celiac_path), celiac_path)
    # Extract Normal
    normal_train, normal_valid = sample_path(os.listdir(normal_path), normal_path)

    # Train Patches
    train_patches = celiac_train + normal_train
    # Valid Patches
    valid_patches = celiac_valid + normal_valid    
    
    # Train Image List
    train_images = convert_list_to_nested_list(train_patches)
    # Valid Image List
    valid_images = convert_list_to_nested_list(valid_patches)
  
    # Initialize Model
    model_ft = MSDSAttentionSequence(2)
    
    # Update Norm layer
    for m in model_ft.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.9
            m.weight = nn.Parameter(torch.ones(m.weight.shape))
            m.running_mean = torch.zeros(m.weight.shape)
            m.bias = nn.Parameter(torch.zeros(m.weight.shape))
            m.running_var = torch.ones(m.weight.shape)
            m.track_running_stats = False    
            
    # Cross Entropy Loss 
    criterion = nn.CrossEntropyLoss()
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-4)
    
    model_ft, model_ft_final = train_model(model_ft, criterion, optimizer_ft, train_images, valid_images, num_epochs=5)