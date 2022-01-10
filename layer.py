import torch
import torch.nn as nn


class ck2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4,stride=1,padding=1,norm='bnorm',relu=0.0):
        super().__init__()
        layer = []
        layer += [nn.Conv2d(in_ch, out_ch,kernel_size=kernel_size,stride=stride,padding=padding)]
        
        if norm != None:
            layer += [nn.BatchNorm2d(out_ch)]
        
        if relu != None:
            layer += [nn.LeakyReLU(relu)]
        
        self.net = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.net(x)
    
class cdk2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4,stride=1,padding=1,norm='bnorm',relu=None,dropout=None):
        super().__init__()
        
        layer = []
        layer += [nn.ConvTranspose2d(in_ch, out_ch,kernel_size=kernel_size,stride=stride,padding=padding)]
        
        if norm != None:
            layer += [nn.BatchNorm2d(out_ch)]
        
        if relu != None:
            layer += [nn.LeakyReLU(relu)]
            
        if dropout != None:
            layer += [nn.Dropout2d(dropout)]
        
        self.net = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.net(x)