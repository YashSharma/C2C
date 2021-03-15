import torch
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score

def save_ckp(state, fpath=None):
    ''' Save model
    '''
    if fpath == None:
        fpath =  'checkpoint.pt'
    torch.save(state, fpath)
    
def load_ckp(checkpoint_fpath, model, optimizer):
    ''' load model
    '''
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def cal_nmi(list1, list2):
    ''' Compute Normalized Mutual Information 
    '''
    nmi_list = []
    for li1, li2 in zip(list1, list2):
        nmi_list.append(normalized_mutual_info_score(li1, li2))
    return np.mean(nmi_list)

