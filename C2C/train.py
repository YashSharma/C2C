import copy
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import albumentations
from albumentations.pytorch import ToTensorV2, ToTensor

from C2C.dataloader import *
from C2C.eval_model import *
from C2C.utils import *
from C2C.cluster import run_clustering

def train_model(model, criterion_dic, optimizer, df, data_transforms, num_cluster=8, num_img_per_cluster=8,
                use_kld=True, num_epochs=25, fpath='checkpoint.pt'):
    """ Function for training
    """
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_test_acc = 0.0
        
    # Loss 
    criterion_ce = criterion_dic['CE']
    criterion_kld = criterion_dic['KLD']
    
    # Separating train, valid images and their labels in a dictionary
    train_images = dict(df.loc[df['is_valid']==0].groupby('wsi')['path'].apply(list))
    valid_images = dict(df.loc[df['is_valid']==1].groupby('wsi')['path'].apply(list))
    train_images_label = dict(df.loc[df['is_valid']==0].groupby('wsi')['label'].apply(max))
    valid_images_label = dict(df.loc[df['is_valid']==1].groupby('wsi')['label'].apply(max))    
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Run Clustering
        train_images, train_images_cluster, valid_images, valid_images_cluster = \
                            run_clustering(train_images, valid_images, model, data_transforms=data_transforms,
                                          num_cluster=num_cluster)    

        # Using mutual information to track cluster assignment change
        if epoch>0:
            print('NMI: {}'.format(cal_nmi(list(train_images_cluster.values()), train_images_cluster_last)))
        
        train_images_cluster_last = list(train_images_cluster.values()).copy()
        
        # Reinitialize dataloader with update clusters
        dataloaders, dataset_sizes = reinitialize_dataloader(train_images, train_images_cluster, train_images_label,\
                                                             valid_images, valid_images_cluster, valid_images_label,\
                                                             data_transforms=data_transforms, num_cluster=num_cluster,\
                                                             num_img_per_cluster=num_img_per_cluster)       
                
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                epoch_acc = eval_model(valid_images, valid_images_label, model, data_transforms=data_transforms)
                
                if epoch_acc >= best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    checkpoint = {
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    save_ckp(checkpoint, fpath)                               
                    
                continue
                
            running_loss_wsi = 0.0
            running_loss_patch = 0.0
            running_corrects = 0
            running_loss_kld = 0.0

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # Iterate over data.
            for i, (inputs, labels, inputs_cluster) in enumerate(dataloaders[phase]):
                
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, outputs_patch, outputs_attn = model(inputs)
                    if labels == 1:
                        patch_labels = torch.ones(len(outputs_patch), dtype=torch.long).to(labels.device)
                    else:
                        patch_labels = torch.zeros(len(outputs_patch), dtype=torch.long).to(labels.device)
                    
                    _, preds = torch.max(outputs, 1)
                    loss_patch = criterion_ce(outputs_patch, patch_labels)
                    loss_wsi = criterion_ce(outputs, labels)
                    loss_kld = criterion_kld(outputs_attn, np.array(inputs_cluster))
                    
                    # Loss
                    if use_kld:
                        loss = loss_wsi + 0.01*loss_patch + 0.01*loss_kld
                    else:
                        loss = loss_wsi + 0.01*loss_patch

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # statistics
                running_loss_wsi += loss_wsi.item() * len(inputs)
                running_loss_patch += loss_patch.item() * len(inputs)
                running_corrects += torch.sum(preds == labels.data)
                # When number of patch less than 4 in a cluster, we skip KLD loss
                try:
                    running_loss_kld += loss_kld.item() * len(inputs)
                except:
                    print('No KLD for cluster with 1 patch')
                            
            epoch_loss_wsi = running_loss_wsi / dataset_sizes[phase]
            epoch_loss_patch = running_loss_patch / dataset_sizes[phase]
            epoch_loss_kld = running_loss_kld / dataset_sizes[phase]            
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss Patch: {:.4f} Loss WSI: {:.4f} Loss KLD: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss_patch, epoch_loss_wsi, epoch_loss_kld, epoch_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Final epoch model
    model_final = copy.deepcopy(model)
    
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model