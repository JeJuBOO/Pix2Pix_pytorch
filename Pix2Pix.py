from pickle import FALSE
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import *

import matplotlib.pyplot as plt
import numpy as np
import os

## HyperParameters
lr = 0.0002
batch_size = 4
image_size = 256
channels_img = 3
num_epoch = 500
wgt_l1 = 1e3
wgt_gan = 1e0

features_d = 64
features_g = 64

## 데이터 불러오기
    #Jittering
transform_train = transforms.Compose([Rescale((286, 286)),
                                      RandomCrop((image_size,image_size)),
                                      ToTensor(),
                                      Normalize(),
                                      ])
transform_val = transforms.Compose([Normalize(), ToTensor()])
transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

dataset = Dataset("C:/Users/BOO/OneDrive/바탕 화면/Facades/facades/train/",transform=transform_train, direction='A2B')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

dataset_val = Dataset("C:/Users/BOO/OneDrive/바탕 화면/Facades/facades/val/",transform=transform_val, direction='A2B')
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create discriminator and generator
netD = Discriminator(2*channels_img, features_d).to(device)
netG = Generator(channels_img, channels_img, features_g).to(device)
    
initialize_weights(netD)
initialize_weights(netG)

# Setup Optimizer for G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

cri_l1 = nn.L1Loss().to(device)
cri_gan = nn.BCELoss().to(device)

writer_input = SummaryWriter(f'./runs/facades_a2b/test_input')
writer_label = SummaryWriter(f'./runs/facades_a2b/test_label')
writer_fake = SummaryWriter(f'./runs/facades_a2b/test_fake')
writer_loss = SummaryWriter(f'./runs/facades_a2b/loss')
print("훈련 시작 합니다.")
step = 0


for epoch in range(num_epoch):
    netD.train()
    netG.train()
    
    lossG_gan_train = []
    lossG_l1_train = []
    lossD_real_train = []
    lossD_fake_train = []
    lossG_train = []
    lossD_train = []
    D_x_train = []
    for batch_idx, train in enumerate(dataloader, 1):
        
        input = train['dataA'].to(device)
        label = train['dataB'].to(device)
    
        fake = netG(input)
        
        real_image = torch.cat([input, label], dim=1)
        fake_image = torch.cat([input, fake], dim=1)
        
        # Train Discriminator: max log(D(y,x)) + log(1 - D(G(x),x))
        netD.zero_grad()
        
        D_real = netD(real_image).reshape(-1)
        lossD_real = cri_gan(D_real, torch.ones_like(D_real))
        
        D_x = D_real.mean()
    
        D_fake = netD(fake_image.detach()).reshape(-1)
        lossD_fake = cri_gan(D_fake, torch.zeros_like(D_fake))
        
        lossD = (lossD_real + lossD_fake)
        
        lossD.backward()
        optimizerD.step()
               
        # Train Generator : min log(1 - D(G(x))) || max log(D(G(x)))
        netG.zero_grad()
        D_fake = netD(fake_image).reshape(-1)
        lossG_gan = cri_gan(D_fake, torch.ones_like(D_fake))
        lossG_l1 = cri_l1(fake, label)
        
        lossG = (wgt_l1 * lossG_l1) + (wgt_gan * lossG_gan)
        
        if epoch > 100:
            lossG.backward()
            optimizerG.step()
            
        D_x_train += [D_x.item()]
        lossG_gan_train += [lossG_gan.item()]
        lossG_l1_train += [lossG_l1.item()]
        lossD_real_train += [lossD_real.item()]
        lossD_fake_train += [lossD_fake.item()]
        lossG_train += [lossG.item()]
        lossD_train += [lossD.item()]
    
    writer_loss.add_scalar('Gan loss train', np.mean(lossG_gan_train), global_step=step)
    writer_loss.add_scalar('L1 loss train', np.mean(lossG_l1_train), global_step=step)
    writer_loss.add_scalar('Disc real loss train', np.mean(lossD_real_train), global_step=step)
    writer_loss.add_scalar('Disc fake loss train', np.mean(lossD_fake_train), global_step=step)
    writer_loss.add_scalar('loss Disc train', np.mean(lossD_train), global_step=step)
    writer_loss.add_scalar('loss Gan train', np.mean(lossG_train), global_step=step)
    print(f'Epoch [{epoch}/{num_epoch}] \
        Loss D : {np.mean(lossD_train):.4f}, Loss G : {np.mean(lossG_train):.4f} D(x): {np.mean(D_x_train):.4f}')
    
    
    with torch.no_grad():
                    
        netG.eval()
        netD.eval()
        lossG_val = []
        lossD_val = []
        lossG_gan_val = []
        lossG_l1_val = []
        lossD_real_val = []
        lossD_fake_val = []
        for i, val in enumerate(dataloader_val, 1):
            
            input_v = val['dataA'].to(device)
            label_v = val['dataB'].to(device)
        
            fake_v = netG(input_v)
            
            real_image_v = torch.cat([input_v, label_v], dim=1)
            fake_image_v = torch.cat([input_v, fake_v], dim=1)
            
            # forward netD
            D_real_v = netD(real_image_v)
            D_fake_v = netD(fake_image_v)
            
            lossD_real_v = cri_gan(D_real_v, torch.ones_like(D_real_v))
            lossD_fake_v = cri_gan(D_fake_v, torch.zeros_like(D_fake_v))
            lossD_v = 0.5 * (lossD_real_v + lossD_fake_v)

            lossG_gan_v = cri_gan(D_fake_v, torch.ones_like(D_fake_v))
            lossG_l1_v = cri_l1(fake_v, label_v)
            lossG_v = (wgt_l1 * lossG_l1_v) + (wgt_gan * lossG_gan_v)
            
            lossG_gan_val += [lossG_gan_v.item()]
            lossG_l1_val += [lossG_l1_v.item()]
            lossD_real_val += [lossD_real_v.item()]
            lossD_fake_val  += [lossD_fake_v.item()]
            lossG_val += [lossG_v.item()]
            lossD_val += [lossD_v.item()]
            
            
        writer_loss.add_scalar('Gan loss val', np.mean(lossG_gan_val), global_step=step)
        writer_loss.add_scalar('L1 loss val', np.mean(lossG_l1_val), global_step=step)
        writer_loss.add_scalar('Disc real loss val', np.mean(lossD_real_val), global_step=step)
        writer_loss.add_scalar('Disc fake loss val', np.mean(lossD_fake_val), global_step=step)
        writer_loss.add_scalar('loss Disc val', np.mean(lossD_val), global_step=step)
        writer_loss.add_scalar('loss Gan val', np.mean(lossG_val), global_step=step)
        
        img_grid_input = torchvision.utils.make_grid(input_v, normalize=True)
        img_grid_label = torchvision.utils.make_grid(label_v, normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake_v, normalize=True)
        writer_input.add_image('Facades input Images', img_grid_input, global_step=step)
        writer_label.add_image('Facades label Images', img_grid_label, global_step=step)
        writer_fake.add_image('Facades fake Images', img_grid_fake, global_step=step)
        step += 1


torch.save(netD, 'Save_model/Disc_a2b.pt')
torch.save(netG, 'Save_model/Gan_a2b.pt')







