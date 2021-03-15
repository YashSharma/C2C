import cv2
import torch
import random
import numpy as np
import albumentations
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2, ToTensor

class WSIDataloader(Dataset):
    """
    Dataloader for iterating through all patches in a WSI
    """    
    def __init__(self, image_path, transform=None):
        self.input_images = image_path
        self.transform = transform  

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        im_path = self.input_images[idx]
        im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
        if self.transform:
            im = self.transform(image=im)['image']            
        return im, self.input_images[idx]

# Return num_images x 3 x 512 x 512
class WSIClusterDataloader(Dataset):
    """
    Dataloader for sampling instance from each cluster in a WSI 
    """
    def __init__(self, image_path, cluster_assignment, label_dic, num_cluster=8, num_img_per_cluster=8,
                 transform=None):
        self.input_images = image_path
        self.cluster = cluster_assignment
        self.label = label_dic
        self.id_map = dict(zip(range(len(image_path)), image_path.keys()))
        self.transform = transform  
        self.num_cluster = num_cluster
        self.num_img_per_cluster = num_img_per_cluster

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image_list = []
        cluster_list = []                
        for cls in range(self.num_cluster):
            cls_id = np.where(np.array(self.cluster[self.id_map[idx]]) == cls)[0]
            if len(cls_id) == 0:
                continue
            random.shuffle(cls_id)
            for im_count, im_id in enumerate(cls_id):
                if im_count >= self.num_img_per_cluster:
                    break
                im = cv2.cvtColor(cv2.imread(self.input_images[self.id_map[idx]][im_id]), cv2.COLOR_BGR2RGB)
                image_list.append(im)
                cluster_list.append(cls)
                
        if self.transform:
            for im_id, im in enumerate(image_list):
                # Albumentation added
                image_list[im_id] = self.transform(image=im)['image']
        image = torch.stack(image_list)
        label = int(self.label[self.id_map[idx]])

        return image, label, cluster_list  

def reinitialize_dataloader(train_images, train_images_cluster, train_images_label, \
                            valid_images, valid_images_cluster, valid_images_label, \
                            data_transforms, num_cluster=8, num_img_per_cluster=8):
    """ Reinitialize WSI cluster dataloader with updated cluster assignment
    """
    
    train_data = WSIClusterDataloader(train_images, train_images_cluster, train_images_label, num_cluster, num_img_per_cluster,
                                      transform=data_transforms)
    val_data = WSIClusterDataloader(valid_images, valid_images_cluster, valid_images_label, num_cluster, num_img_per_cluster,
                                    transform=data_transforms)

    batch_size = 1
    num_workers = 0

    dataloaders = {'train': torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers),
                  'val': torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers)}

    dataset_sizes = {'train': len(train_data), 'val': len(val_data)}
    
    return dataloaders, dataset_sizes