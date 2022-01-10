from layer import *

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g=64, norm="bnorm"):
        super(Generator,self).__init__()
        
        self.encoder1 = ck2d(channels_noise, features_g    , kernel_size=4,stride=2,padding=1,norm=None,relu=0.2)
        self.encoder2 = ck2d(features_g    , features_g * 2, kernel_size=4,stride=2,padding=1,norm=norm,relu=0.2) 
        self.encoder3 = ck2d(features_g * 2, features_g * 4, kernel_size=4,stride=2,padding=1,norm=norm,relu=0.2)
        self.encoder4 = ck2d(features_g * 4, features_g * 8, kernel_size=4,stride=2,padding=1,norm=norm,relu=0.2) 
        self.encoder5 = ck2d(features_g * 8, features_g * 8, kernel_size=4,stride=2,padding=1,norm=norm,relu=0.2) 
        self.encoder6 = ck2d(features_g * 8, features_g * 8, kernel_size=4,stride=2,padding=1,norm=norm,relu=0.2) 
        self.encoder7 = ck2d(features_g * 8, features_g * 8, kernel_size=4,stride=2,padding=1,norm=norm,relu=0.2) 
        self.encoder8 = ck2d(features_g * 8, features_g * 8, kernel_size=4,stride=2,padding=1,norm=norm,relu=0.0) 
        
        self.decoder1 = cdk2d(    features_g * 8, features_g * 8, kernel_size=4,stride=2,padding=1,norm=norm,relu=0.0,dropout=0.5) 
        self.decoder2 = cdk2d(2 * features_g * 8, features_g * 8, kernel_size=4,stride=2,padding=1,norm=norm,relu=0.0,dropout=0.5) 
        self.decoder3 = cdk2d(2 * features_g * 8, features_g * 8, kernel_size=4,stride=2,padding=1,norm=norm,relu=0.0,dropout=0.5) 
        self.decoder4 = cdk2d(2 * features_g * 8, features_g * 8, kernel_size=4,stride=2,padding=1,norm=norm,relu=0.0) 
        self.decoder5 = cdk2d(2 * features_g * 8, features_g * 4, kernel_size=4,stride=2,padding=1,norm=norm,relu=0.0) 
        self.decoder6 = cdk2d(2 * features_g * 4, features_g * 2, kernel_size=4,stride=2,padding=1,norm=norm,relu=0.0) 
        self.decoder7 = cdk2d(2 * features_g * 2, features_g    , kernel_size=4,stride=2,padding=1,norm=norm,relu=0.0) 
        self.decoder8 = cdk2d(2 * features_g    , channels_img  , kernel_size=4,stride=2,padding=1,norm=None,relu=None)
        
    def forward(self, x):           # 4 * 3 * 256 * 256
        enc1 = self.encoder1(x)     # 4 * 64 * 128 * 128
        enc2 = self.encoder2(enc1)  # 4 * 128 * 64 * 64
        enc3 = self.encoder3(enc2)  # 4 * 256 * 32 * 32
        enc4 = self.encoder4(enc3)  # 4 * 512 * 16 * 16
        enc5 = self.encoder5(enc4)  # 4 * 512 * 8  * 8
        enc6 = self.encoder6(enc5)  # 4 * 512 * 4  * 4
        enc7 = self.encoder7(enc6)  # 4 * 512 * 2  * 2
        enc8 = self.encoder8(enc7)  # 4 * 512 * 1  * 1
        
        dec1 = self.decoder1(enc8)  # 4 * 512 * 2  * 2      
        dec2 = self.decoder2(torch.cat([dec1, enc7], dim=1))# 4 * 1024 * 2  * 2
        dec3 = self.decoder3(torch.cat([dec2, enc6], dim=1))# 4 * 1024 * 4  * 4
        dec4 = self.decoder4(torch.cat([dec3, enc5], dim=1))# 4 * 1024 * 8  * 8
        dec5 = self.decoder5(torch.cat([dec4, enc4], dim=1))# 4 * 512 * 16 * 16
        dec6 = self.decoder6(torch.cat([dec5, enc3], dim=1))# 4 * 256 * 32 * 32
        dec7 = self.decoder7(torch.cat([dec6, enc2], dim=1))# 4 * 128  * 64 * 64
        dec8 = self.decoder8(torch.cat([dec7, enc1], dim=1))# 4 * 64  * 128 * 128
        
        x = torch.tanh(dec8)# 4 * 3  * 256 * 256
        return x
    

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d=64, norm="bnorm"):
        super(Discriminator, self).__init__()
    
        self.disc1 = ck2d(channels_img  , features_d    ,kernel_size=4,stride=2,padding=1,norm=None,relu=0.2)
        self.disc2 = ck2d(features_d    , features_d * 2,kernel_size=4,stride=2,padding=1,norm=norm,relu=0.2)
        self.disc3 = ck2d(features_d * 2, features_d * 4,kernel_size=4,stride=2,padding=1,norm=norm,relu=0.2)
        self.disc4 = ck2d(features_d * 4, features_d * 8,kernel_size=4,stride=1,padding=1,norm=norm,relu=0.2)
        self.disc5 = ck2d(features_d * 8, 1             ,kernel_size=4,stride=1,padding=1,norm=None,relu=None)
        

        
    def forward(self, x):       # 4 * 6   * 256  * 256
        disc = self.disc1(x)    # 4 * 64  * 128  * 128
        disc = self.disc2(disc) # 4 * 128 * 64  * 64
        disc = self.disc3(disc) # 4 * 256 * 32  * 32
        disc = self.disc4(disc) # 4 * 512 * 31  * 31
        disc = self.disc5(disc) # 4 * 1   * 30  * 30
        
        x = torch.sigmoid(disc)
        return x
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)