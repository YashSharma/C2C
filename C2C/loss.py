import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class KLDLoss(nn.Module):
    """KL-divergence loss between attention weight and uniform distribution"""
    
    def __init__(self):
        super(KLDLoss, self).__init__()
        
    def forward(self, attn_val, cluster):
        """
        Example:
          Input - attention value = torch.tensor([0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.05, 
                                0.1, 0.05, 0.1, 0.05, 0.1, 0.05])
                  cluster = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
          Output - 0.0043
        """
        kld_loss = 0
        is_cuda = attn_val.device
        cluster = np.array(cluster)
        for cls in np.unique(cluster):
            index = np.where(cluster==cls)[0]
            # HARD CODE
            if len(index)<=4:
                continue            
            kld_loss += F.kl_div(F.log_softmax(attn_val[index], dim=0)[None],\
                            torch.ones(len(index), 1)[None].to(is_cuda)/len(index),\
                            reduction = 'batchmean')
            
        return kld_loss