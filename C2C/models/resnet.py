import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class WSIClassifier(nn.Module):
    def __init__(self, n_class=2, bn_track_running_stats=False):
        super(WSIClassifier, self).__init__()
        self.L = 64
        self.D = 32
        self.K = 1
        
        resnet = models.resnet18(pretrained=True)
        
        # Since patches in each batch belong to a WSI, switching off batch statistics tracking
        # Or reinitializing batch parameters and changing momentum for quick domain adoption
        if bn_track_running_stats:
            for modules in resnet.modules():
                if isinstance(modules, nn.BatchNorm2d):                
                    modules.track_running_stats = False
        else:
            for modules in resnet.modules():
                if isinstance(modules, nn.BatchNorm2d):                                
                    modules.momentum = 0.9
                    modules.weight = nn.Parameter(torch.ones(modules.weight.shape))
                    modules.running_mean = torch.zeros(modules.weight.shape)
                    modules.bias = nn.Parameter(torch.zeros(modules.weight.shape))
                    modules.running_var = torch.ones(modules.weight.shape)                  
            
        modules = list(resnet.children())[:-1]          
        self.resnet_head = nn.Sequential(*modules)
        self.resnet_tail = nn.Sequential(nn.Linear(512, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, self.L),
                                        nn.ReLU())

        self.attention = nn.Sequential(nn.Linear(self.L, self.D),
                                        nn.Tanh(),
                                        nn.Linear(self.D, self.K))
        
        self.classifier = nn.Sequential(nn.Linear(self.L*self.K, n_class))
        self.patch_classifier = nn.Sequential(nn.Linear(self.L*self.K, n_class))

    def forward(self, x):
        x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        x = self.resnet_head(x)
        x = x.view(x.size(0), -1)
        x = self.resnet_tail(x)        
        xp = self.patch_classifier(x)
        
        A_unnorm = self.attention(x)
        A = torch.transpose(A_unnorm, 1, 0)
        A = F.softmax(A, dim=1)
        
        M = torch.mm(A, x)
        Y_prob = self.classifier(M)        
        return Y_prob, xp, A_unnorm
    
class Enc(nn.Module):
    def __init__(self, model_base):
        super(Enc, self).__init__()
        self.resnet_head = model_base.resnet_head
        self.resnet_tail = model_base.resnet_tail
        
    def forward(self, x):
        x = self.resnet_head(x)
        x = x.view(x.size(0), -1)
        x = self.resnet_tail(x)
        
        return x    
    
class PatchClassifier(nn.Module):
    def __init__(self, model_base):
        super(PatchClassifier, self).__init__()
        self.resnet_head = model_base.resnet_head
        self.resnet_tail = model_base.resnet_tail
        self.patch_classifier = model_base.patch_classifier
        
    def forward(self, x):
        x = self.resnet_head(x)
        x = x.view(x.size(0), -1)
        x = self.resnet_tail(x)
        x = self.patch_classifier(x)
        
        return x            
    
# Get Embedding Representation, Attn Value
class EncAttn(nn.Module):
    def __init__(self, model_base):
        super(EncAttn, self).__init__()
        self.resnet_head = model_base.resnet_head
        self.resnet_tail = model_base.resnet_tail
        self.attention = model_base.attention
        
    def forward(self, x):
        x = self.resnet_head(x)
        x = x.view(x.size(0), -1)
        x = self.resnet_tail(x)
        attn = self.attention(x)
        
        return attn, x    