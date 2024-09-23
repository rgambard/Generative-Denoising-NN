import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.optim.lr_scheduler import StepLR

class VAEBlock(nn.Module):
    def __init__(self, repr_size, input_size, latent_size):
        super(VAEBlock, self).__init__()
        self.latent_size = latent_size
        self.unetin = UNet(repr_size+input_size, latent_size*2+input_size)
        self.unetmodifr = UNet(latent_size+repr_size, repr_size)
        self.unetmodifx = UNet(latent_size+repr_size+input_size,input_size)

    def forward(self, r, x): # r is the representation, x is the input
        input_unetin = torch.cat((r,x), dim = 1)
        zdistr = self.unetin(input_unetin)

        mean = zdistr[:,:self.latent_size,:,:]
        area = zdistr[:,self.latent_size:2*self.latent_size,:,:]
        infox = zdistr[:,2*self.latent_size:,:,:]

        epsilon = 0.01 # minimum area
        maxarea = 0.5 # maximum area
        area = torch.sigmoid(area)*(maxarea-epsilon)
        area = epsilon+area
        mean = torch.maximum(mean, -1/2+area/2)
        mean = torch.minimum(mean, 1/2-area/2)

        #generating z from its moments and computing loss
        shape = list(x.shape)
        shape[1] = self.latent_size
        noise = torch.rand(shape, device = x.device)-1/2
        z = mean+area*noise# sample Z from zdistr
        loss_z = torch.sum(-torch.log(area), dim=(1,2,3))# compute the conditional entropy 
        loss_z += torch.sqrt(torch.sum(mean**2, dim = (1,2,3)))

        inputunetx = torch.cat((z,r,infox),dim=1)
        diffx = self.unetmodifx(inputunetx)
        x = x+diffx

        diffr = self.unetmodifr(torch.cat((z, r),dim=1))
        r = r+diffr
        return (r, x, loss_z)

    def gen(self, r, temp = 1):
        shape = list(r.shape)
        shape[1] = self.latent_size
        z = temp*(torch.rand(shape, device = r.device)-1/2)
        diffr = self.unetmodifr(torch.cat((z, r),dim=1))
        r = r+diffr
        return r


class VAE(nn.Module):
    def __init__(self, input_size, repr_size, inter_size, latent_size, depth=3):
        super(VAE, self).__init__()
        self.repr_size = repr_size

        self.unetin = UNet(input_size, inter_size)
        self.unetout = UNet(repr_size, input_size)
        self.vaeblocks = nn.ModuleList([VAEBlock(repr_size, inter_size, latent_size) for d in range(depth)])

    def forward(self,inp):
        x = self.unetin(inp)
        repr_shape = list(inp.shape)
        repr_shape[1] = self.repr_size

        r = torch.zeros(repr_shape, device = inp.device)
        loss_z_total = 0
        for vaeblock in self.vaeblocks:
            r, x, loss_z = vaeblock(r, x)
            loss_z_total += loss_z

        out = self.unetout(r)
        return out, loss_z_total

    def gen(self, device, n=10, temp = 1):
        repr_shape = (n, self.repr_size, 28,28)
        r = torch.zeros(repr_shape, device = device)
        for vaeblock in self.vaeblocks:
            r = vaeblock.gen(r, temp = temp)
        out = self.unetout(r)
        return out


class UNet_Res(nn.Module):
    def __init__(self,input_channels, output_channels, depth = 3, int_channels = 16):
        super(UNet_Res, self).__init__()
        self.unetin = UNet(input_channels, int_channels)
        self.unetout = UNet(int_channels, output_channels)
        self.unets = nn.ModuleList([UNet(int_channels, int_channels) for i in range(depth)])

    def forward(self,x):
        x = self.unetin(x)
        i = 1
        for unet in self.unets:
            res_input = (x-x.mean())/x.std()
            res_output = unet(res_input)
            res_output = (res_output-res_output.mean())/res_output.std()
            i+=1
        x = self.unetout(x)
        return x
        


class UNet(nn.Module):
    def __init__(self,input_channels, output_channels, int_channels =16):
        super(UNet, self).__init__()
        self.conv11 = nn.Conv2d(input_channels,int_channels, 3, 1, 1)
        self.conv12 = nn.Conv2d(int_channels, int_channels, 3, 1, 1)
        self.conv21 = nn.Conv2d(int_channels, int_channels, 3, 1, 1)
        self.conv22 = nn.Conv2d(int_channels,int_channels, 3, 1, 1)
        self.conv31 = nn.Conv2d(int_channels,int_channels, 3, 1, 1)
        self.conv32 = nn.Conv2d(int_channels,int_channels, 3, 1, 1)
        self.conv33 = nn.Conv2d(int_channels,int_channels, 3, 1, 1)
        self.convu21 = nn.Conv2d(int_channels,int_channels, 3, 1, 1)
        self.convu22 = nn.Conv2d(int_channels,int_channels, 3, 1, 1)
        self.convu11 = nn.Conv2d(int_channels,int_channels, 3, 1, 1)
        self.convu12 = nn.Conv2d(int_channels, int_channels, 3, 1, 1)
        self.convout = nn.Conv2d(int_channels,output_channels,1,1)
        torch.nn.init.kaiming_normal_(self.convout.weight,nonlinearity = 'linear')


    def forward(self, x):
        x11 = F.relu(self.conv11(x))
        x12 = self.conv12(x11)
        x12 = (x12-x12.mean())/x12.std()
        x20 = F.interpolate(x12,scale_factor = 0.5)
        x21 = F.relu(self.conv21(x20))
        x22 = self.conv22(x21)
        x30 = F.interpolate(x22,scale_factor = 0.5)
        x31 = F.relu(self.conv31(x30))
        x32 = F.relu(self.conv32(x31))
        x33 = self.conv33(x31)
        xu20 = F.interpolate(x33,scale_factor = 2)
        xu21 = F.relu((xu20+x22)/math.sqrt(2))
        xu22 = F.relu(self.convu21(xu21))
        xu23 = self.convu22(xu22)
        xu10 = F.interpolate(xu23,scale_factor = 2)
        xu11 = F.relu((xu10+x12)/math.sqrt(2))
        xu12 = F.relu(self.convu11(xu11))
        xu13 = F.relu(self.convu12(xu12))

        out = self.convout(xu13)
        return out

class Denoiser(nn.Module):
    def __init__(self,noisy_input_channels, output_channels, depth = 3):
        super(Denoiser, self).__init__()
        self.unet_res = UNet_Res(noisy_input_channels,output_channels, depth = depth)
    def forward(self,x):
        x = self.unet_res(x)
        return x

class GenImage(nn.Module):
    def __init__(self,input_channels, output_channels):
        super(GenImage, self).__init__()
        self.unet_res = UNet_Res(input_channels, output_channels, depth = 3)

    def forward(self,x):
        gen_im = self.unet_res(x)
        gen_im = F.tanh(gen_im)
        return gen_im


