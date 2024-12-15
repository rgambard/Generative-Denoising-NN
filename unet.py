import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.optim.lr_scheduler import StepLR

            
class UNet_Res(nn.Module):
    def __init__(self,input_channels, output_channels, depth = 3, int_channels = 64):
        super(UNet_Res, self).__init__()
        self.unetin = UNet(input_channels, int_channels, int_channels = int_channels)
        self.unetout = UNet(int_channels, output_channels, int_channels = int_channels)
        self.unets = nn.ModuleList([UNet(int_channels, int_channels, int_channels=int_channels) for i in range(depth)])

    def forward(self,x):
        x = self.unetin(x)
        i = 1
        for unet in self.unets:
            res_input = (x-x.mean(dim=(1,2,3))[:,None,None,None])/x.std(dim=(1,2,3))[:,None,None,None]
            res_output = unet(res_input)
            x = x+res_output
            i+=1
        x = (x-x.mean(dim=(1,2,3))[:,None,None,None])/x.std(dim=(1,2,3))[:,None,None,None]
        x = self.unetout(x)
        return x
        


class UNet(nn.Module):
    def __init__(self,input_channels, output_channels, int_channels =16):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.conv11 = nn.Conv2d(input_channels,int_channels, 3, 1, 1) # kernel, stride, padding
        self.conv12 = nn.Conv2d(int_channels, int_channels, 3, 1, 1)
        self.conv13 = nn.Conv2d(int_channels, int_channels, 3, 1, 1)
        self.conv21 = nn.Conv2d(int_channels, int_channels, 3, 1, 1)
        self.conv22 = nn.Conv2d(int_channels,int_channels, 3, 1, 1)
        self.conv31 = nn.Conv2d(int_channels,int_channels, 3, 1, 1)
        self.conv32 = nn.Conv2d(int_channels,int_channels, 3, 1, 1)
        self.conv33 = nn.Conv2d(int_channels,int_channels, 3, 1, 1)
        self.convu21 = nn.Conv2d(int_channels,int_channels, 3, 1, 1)
        self.convu22 = nn.Conv2d(int_channels,int_channels, 3, 1, 1)
        self.convu11 = nn.Conv2d(int_channels,int_channels, 3, 1, 1)
        self.convu12 = nn.Conv2d(int_channels, int_channels, 3, 1, 1)
        self.convu13 = nn.Conv2d(int_channels, int_channels, 3, 1, 1)
        self.convout = nn.Conv2d(int_channels,output_channels,1,1)
        self.init_weight()

    def init_weight(self):
        x = torch.randn(128,self.input_channels,32,32)
        with torch.no_grad():
            self.conv11.weight /= self.conv11(x).std()/math.sqrt(2)
            x11 = F.relu(self.conv11(x))
            self.conv12.weight /= self.conv12(x11).std()/math.sqrt(2)
            x12 = F.relu(self.conv12(x11))
            self.conv13.weight /= self.conv13(x12).std()/math.sqrt(2)
            x13 = self.conv13(x12)/math.sqrt(2)
            x20 = F.interpolate(x13,scale_factor = 0.5)
            self.conv21.weight /= self.conv21(x20).std()/math.sqrt(2)
            x21 = F.relu(self.conv21(x20))
            self.conv22.weight /= self.conv22(x21).std()/math.sqrt(2)
            x22 = self.conv22(x21)/math.sqrt(2)
            x30 = F.interpolate(x22,scale_factor = 0.5)
            self.conv31.weight /= self.conv31(x30).std()/math.sqrt(2)
            x31 = F.relu(self.conv31(x30))
            self.conv32.weight /= self.conv32(x31).std()/math.sqrt(2)
            x32 = F.relu(self.conv32(x31))
            self.conv33.weight /= self.conv33(x32).std()/math.sqrt(2)
            x33 = self.conv33(x32)/math.sqrt(2)
            xu20 = F.interpolate(x33,scale_factor = 2)
            xu21 = F.relu((xu20+x22))
            self.convu21.weight /= self.convu21(xu21).std()/math.sqrt(2)
            xu22 = F.relu(self.convu21(xu21))
            self.convu22.weight /= self.convu22(xu22).std()/math.sqrt(2)
            xu23 = self.convu22(xu22)/math.sqrt(2)
            xu10 = F.interpolate(xu23,scale_factor = 2)
            xu11 = F.relu((xu10+x13))
            self.convu11.weight /= self.convu11(xu11).std()/math.sqrt(2)
            xu12 = F.relu(self.convu11(xu11))
            self.convu12.weight /= self.convu12(xu12).std()/math.sqrt(2)
            xu13 = F.relu(self.convu12(xu12))
            self.convu13.weight /= self.convu13(xu13).std()/math.sqrt(2)
            xu14 = F.relu(self.convu13(xu13))

            self.convout.weight /= self.convout(xu14).std()
            out = self.convout(xu14)

            x = torch.randn(128,self.input_channels,32,32)
            print(self.forward(x).std())

        



    def forward(self, x):

        x11 = F.relu(self.conv11(x))
        x12 = F.relu(self.conv12(x11))
        x13 = self.conv13(x12)/math.sqrt(2)
        x20 = F.interpolate(x13,scale_factor = 0.5)
        x21 = F.relu(self.conv21(x20))
        x22 = self.conv22(x21)/math.sqrt(2)
        x30 = F.interpolate(x22,scale_factor = 0.5)
        x31 = F.relu(self.conv31(x30))
        x32 = F.relu(self.conv32(x31))
        x33 = self.conv33(x32)/math.sqrt(2)
        xu20 = F.interpolate(x33,scale_factor = 2)
        xu21 = F.relu((xu20+x22))
        xu22 = F.relu(self.convu21(xu21))
        xu23 = self.convu22(xu22)/math.sqrt(2)
        xu10 = F.interpolate(xu23,scale_factor = 2)
        xu11 = F.relu((xu10+x13))
        xu12 = F.relu(self.convu11(xu11))
        xu13 = F.relu(self.convu12(xu12))
        xu14 = F.relu(self.convu13(xu13))

        out = self.convout(xu14)
        return out

class MixNet(nn.Module):
    def __init__(self,int_channels =16):
        super(ConvNet, self).__init__()
        self.convl11 = nn.Conv2d(int_channels,int_channels, 3, 1, 1) # kernel, stride, padding
        self.convl12 = nn.Conv2d(int_channels,int_channels, 3, 1, 1) # kernel, stride, padding
        self.convl21 = nn.Conv2d(int_channels,int_channels, 3, 1, 1) # kernel, stride, padding
        self.convl22 = nn.Conv2d(int_channels,int_channels, 3, 1, 1) # kernel, stride, padding
        self.convl31 = nn.Conv2d(int_channels,int_channels, 3, 1, 1) # kernel, stride, padding
        self.convl32 = nn.Conv2d(int_channels,int_channels, 3, 1, 1) # kernel, stride, padding
        self.convl41 = nn.Conv2d(int_channels,int_channels, 3, 1, 1) # kernel, stride, padding
        self.convl42 = nn.Conv2d(int_channels,int_channels, 3, 1, 1) # kernel, stride, padding

    def mix(self,x,x1,x2,x3):
        x1 += F.interpolate(x,x1.shape)
        x2 += F.interpolate(x1,x2.shape)
        x3 += F.interpolate(x2,x3.shape)
        x4 += F.interpolate(x3,x4.shape)
        pass
    def forward(self, x, x1, x2, x3):
        x += self.convl12(F.relu(self.conv11(x)))
        x1 += self.convl22(F.relu(self.convl21(x1)))
        x2 += self.convl31(F.relu(self.convl32(x2)))
        x3 += self.convl41(F.relu(self.convl42(x3)))
        return x, x1, x2, x3

class Denoiser(nn.Module):
    def __init__(self,noisy_input_channels, output_channels, depth = 3):
        super(Denoiser, self).__init__()
        self.unet_res = UNet_Res(noisy_input_channels,output_channels, depth = depth)
    def forward(self,x):
        x = self.unet_res(x)
        return x

class EnergyModel(nn.Module):
    def __init__(self,noisy_input_channels, output_channels, depth = 10):
        super(EnergyModel, self).__init__()
        self.unet_res = UNet_Res(noisy_input_channels, 1, depth = depth)
        self.output_channels = output_channels

    def forward(self,samples):
        samples.requires_grad_(True)
        logp = self.unet_res(samples)
        dlogp = torch.autograd.grad(logp.sum(), samples, create_graph=True)[0]
        return dlogp[:,:self.output_channels,:,:]
