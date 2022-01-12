import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import *

# test
# HyperParameters
lr = 0.0002
batch_size = 4
image_size = 256
channels_img = 3
num_epoch = 300
wgt_l1 = 1e2
wgt_gan = 1e0

features_d = 64
features_g = 64

transform_test = transforms.Compose([Normalize(), ToTensor()])

dataset_test = Dataset("C:/Users/BOO/OneDrive/바탕 화면/Facades/facades/test/",transform=transform_test, direction='A2B')
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netD = Discriminator(2*channels_img, features_d).to(device)
netG = Generator(channels_img, channels_img, features_g).to(device)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

cri_l1 = nn.L1Loss().to(device)
cri_gan = nn.BCELoss().to(device)

writer_input = SummaryWriter(f'./runs/facades_a2b_test/test_input')
writer_label = SummaryWriter(f'./runs/facades_a2b_test/test_label')
writer_fake = SummaryWriter(f'./runs/facades_a2b_test/test_fake')
writer_loss = SummaryWriter(f'./runs/facades_a2b_test/loss')


netD = torch.load('./Save_model/Disc_a2b.pt')
netG = torch.load('./Save_model/Gan_a2b.pt')

netD.eval()
netG.eval()

lossG_test = []
lossD_test = []
lossG_gan_test = []
lossG_l1_test = []
lossD_real_test = []
lossD_fake_test = []
for i, test in enumerate(dataloader_test, 1):
    
    input = test['dataA'].to(device)
    label = test['dataB'].to(device)

    fake = netG(input)
    
    real_image = torch.cat([input, label], dim=1)
    fake_image = torch.cat([input, fake], dim=1)
    
    # forward netD
    D_real = netD(real_image)
    D_fake = netD(fake_image)
    
    lossD_real = cri_gan(D_real, torch.ones_like(D_real))
    lossD_fake = cri_gan(D_fake, torch.zeros_like(D_fake))
    lossD = 0.5 * (lossD_real + lossD_fake)

    lossG_gan = cri_gan(D_fake, torch.ones_like(D_fake))
    lossG_l1 = cri_l1(fake, label)
    lossG = (wgt_l1 * lossG_l1) + (wgt_gan * lossG_gan)
    

    lossG_gan_test += [lossG_gan.item()]
    lossG_l1_test += [lossG_l1.item()]
    lossD_real_test += [lossD_real.item()]
    lossD_fake_test += [lossD_fake.item()]
    lossG_test += [lossG.item()]
    lossD_test += [lossD.item()]
    img_grid_input = torchvision.utils.make_grid(input, normalize=True)
    img_grid_label = torchvision.utils.make_grid(label, normalize=True)               
    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
    writer_input.add_image('Facades input Images', img_grid_input, global_step=i)
    writer_label.add_image('Facades label Images', img_grid_label, global_step=i)
    writer_fake.add_image('Facades fake Images', img_grid_fake, global_step=i)
    
print(f'Loss D : {np.mean(lossD_test):.4f}, Loss G : {np.mean(lossG_test):.4f}')
print(f'gan loss : {np.mean(lossG_gan_test):.4f}, L1 loss : {np.mean(lossG_l1_test):.4f}')
print(f'real loss : {np.mean(lossD_real_test):.4f}, fake loss : {np.mean(lossD_fake_test):.4f}')