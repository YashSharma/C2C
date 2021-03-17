import torch
import faiss
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader

from C2C.models.resnet import PatchClassifier, Enc
from C2C.dataloader import *



def get_representation(dl, enc):
    """
    Computes representation of patches of a WSI 
    
    Parameters:
        - dataloader: dataloader for patches coming from a WSI
        - enc: encoder for generating the representation
    
    Return:
        - img_rep: contain list of representations for all the patches
        - path_list: contain path corresponding to representation of all the images for selective filtering in dataloader
    """
    img_rep = np.array([])
    path_list = []
    for i, (input_image, input_image_path) in enumerate(dl):
        path_list += list(input_image_path)
        if len(img_rep):
            img_rep = np.concatenate((img_rep, enc(input_image.cuda()).detach().cpu().numpy()))
        else:
            img_rep = enc(input_image.cuda()).detach().cpu().numpy()
    return img_rep, path_list

def cluster_representation(im, num_cluster=8):
    """
    Cluster patches

    Parameters:
        - im: contain a list of patch embedding and patch path
    
    Return:
        - labels: kmean cluster for each patch
        - cluster distance: distance of patch from centroid
        - path list: list of patch path
    """
    img_embedding, path_list = im[0], im[1]
    img_embedding = normalize(img_embedding).astype('float32')
    kmeans = faiss.Kmeans(img_embedding.shape[1], min(num_cluster, len(img_embedding)))
    kmeans.train(img_embedding)
    label_metric, label = kmeans.assign(img_embedding)    
    return label, label_metric, path_list    

def select_topk(dl, enc):    
    """
    Function for sampling top-k highest probabilities patch
    """
    img_rep = np.array([])
    path_list = []
    for i, (input_image, input_image_path) in enumerate(dl):
        path_list += list(input_image_path)
        if len(img_rep):
            img_rep = np.concatenate((img_rep, enc(input_image.cuda()).detach().cpu().numpy()))
        else:
            img_rep = enc(input_image.cuda()).detach().cpu().numpy()
    
    img_rep = img_rep[:, 1]
    return img_rep, path_list
    
def run_clustering(train_img_dic, valid_img_dic, model_base, data_transforms, num_cluster=8,
                   for_validation=False, topk=False):
    """
    Function for running clustering 
    """    

    if topk:
        enc = PatchClassifier(model_base)
    else:
        enc = Enc(model_base)
    enc.eval()
    enc = enc.cuda()
    
    valid_img = {}
    valid_img_cls = {}
    for im, im_list in tqdm(valid_img_dic.items()):    
        valid_img[im] = im_list
        valid_img_cls[im] = [0]*len(im_list)
                            
    if for_validation:
        return valid_img, valid_img_cls

    train_img = {}
    train_img_cls = {}
    with torch.no_grad():
        for im, im_list in tqdm(train_img_dic.items()):
            td = WSIDataloader(im_list, transform=data_transforms)
            tdl = torch.utils.data.DataLoader(td, batch_size=128, shuffle=False)
            
            if topk:
                # Use patch classifier to identify most probable diseased patches
                img_rep, path_list = select_topk(tdl, enc)
                cluster = np.ones(len(path_list))
                # MAX NUM PATCHES = 64 HARDCODE
                cluster[np.argsort(img_rep.flatten())[::-1][:64]] = 0
                pl = path_list
            else:
                cluster, cluster_distance, pl = cluster_representation(get_representation(tdl, enc),
                                                                      num_cluster=num_cluster)            
            train_img_cls[im] = list(np.array(cluster))
            train_img[im] = list(np.array(pl))
    
    del enc
    return train_img, train_img_cls, valid_img, valid_img_cls