import argparse
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
        sigma = zdistr[:,self.latent_size:2*self.latent_size,:,:]
        infox = zdistr[:,2*self.latent_size:,:,:]

        sigma = torch.sigmoid(sigma)
        sigma = 0.01+sigma

        #generating z from its moments and computing loss
        shape = list(x.shape)
        shape[1] = self.latent_size
        noise = torch.randn(shape, device = x.device)
        z = mean+sigma*noise# sample Z from zdistr
        loss_z = 1/2*torch.sum(-2*torch.log(sigma)+mean**2+sigma**2)# compute the conditional entropy 

        inputunetx = torch.cat((z,r,infox),dim=1)
        diffx = self.unetmodifx(inputunetx)
        x = x+diffx

        diffr = self.unetmodifr(torch.cat((z, r),dim=1))
        r = r+diffr
        return (r, x, loss_z)

    def gen(self, r, temp = 1):
        shape = list(r.shape)
        shape[1] = self.latent_size
        z = temp*torch.randn(shape, device = r.device)
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
            x = x+unet(x)
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
        x = (x-x.mean())/x.std()
        x11 = F.relu(self.conv11(x))
        x12 = self.conv12(x11)
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
        xu13 = (xu13-xu13.mean())/xu13.std()

        out = self.convout(xu13)
        return out


def forward(model, data):
    im = data
    noise = torch.randn_like(im)
    eps = 0.05

    im_in = im+noise*eps
    im_out, loss_z = model(im_in)
    loss_im = torch.sum((im_in-im_out)**2)
    loss = loss_im+loss_z
    return im_in, im_out, loss, loss_im, loss_z

def train(args, model,  device, train_loader, optimizer, epoch):
    model.train()
    mean_loss = 0
    mean_loss_im = 0
    mean_loss_z = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
     
        optimizer.zero_grad()

        im_in, im_out, loss, loss_im, loss_z = forward(model, data)
        
        mean_loss += loss.item()
        mean_loss_im += loss_im.item()
        mean_loss_z += loss_z.item()
        loss.backward()

        optimizer.step()

        if (batch_idx+1) % args.log_interval == 0:
            mean_loss = mean_loss/(args.log_interval*data.shape[0])
            mean_loss_im = mean_loss_im/(args.log_interval*data.shape[0])
            mean_loss_z = mean_loss_z/(args.log_interval*data.shape[0])
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Loss_z : {:.6f} Loss_im {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), mean_loss, mean_loss_z, mean_loss_im))
            mean_loss = 0
            mean_loss_im = 0
            mean_loss_z = 0
            if args.dry_run:
                break


from PIL import Image
import time
def save_image(im,name):
    normalized = im-torch.min(im)
    normalized = normalized/torch.max(normalized)
    utils.save_image(normalized,name)

nid= "v1"
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    test_loss_im = 0
    test_loss_z = 0
    ctime = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            im_in, im_out, loss, loss_im, loss_z  = forward(model, data)

            test_loss += loss
            test_loss_im += loss_im
            test_loss_z += loss_z

    test_loss /= len(test_loader.dataset)
    test_loss_im /= len(test_loader.dataset)
    test_loss_z /= len(test_loader.dataset)

    print('\nTest set {:.4f} Average loss: {:.4f} lossz : {:.4f} lossim : {:.4f} \n'.format(time.time()-ctime,test_loss, test_loss_z, test_loss_im))

    
    for temp in [0.1,0.5,1]:
        im_gen  = model.gen(device, temp = temp)
        save_image(im_gen, "im/vae_generated"+str(temp)+".jpg")
    for i in range(1):
        save_image(im_in[i], "im/vae_input"+str(i)+".jpg")
        save_image(im_out[i], "im/vae_output"+str(i)+".jpg")





class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

def main():
    # Training settings
    args_dict = {'batch_size' : 64, 'test_batch_size' : 1000, 'epochs' : 14, 'lr' : 0.0001, 'gamma' : 0.7, 'no_cuda' :False, 'dry_run':False, 'seed': 1, 'log_interval' : 100, 'save_model' : False}
    args = dotdict(args_dict)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = VAE(1,16,16,2, depth = 10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model,  device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
