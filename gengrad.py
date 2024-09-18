import argparse
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from unet import UNet_Res, UNet

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

"""
class VAEBlock(nn.Module):
    def __init__(self, repr_size, input_size, latent_size):
        super(VAEBlock, self)
        self.unet1 = UNet(repr_size+input_size, latent_size*2)
        self.unet2 = UNet(latent_size+repr_size, repr_size)

    def forward(self, r, x): # r is the representation, x is the input
        input_enc = torch.cat(r,x, dim = 1)
        zdistr = self.unet1(input_enc)
        z = 0# sample Z from zdistr
        loss_z = 0# compute the conditional entropy and the KL loss
        enc_trans = self.unet2(torch.cat(z, r))
        return (r+enc_trans,loss_z)
"""
class Discriminator(nn.Module):
    def __init__(self,input_channels, output_channels, depth = 3):
        super(Discriminator, self).__init__()
        self.unet_in = UNet(input_channels, 16)
        self.unet_res = UNet_Res(16, 16, depth = depth)
    def forward(self,x):
        x = self.unet_in(x)
        x = (x-x.mean())/x.std()
        x = self.unet_res(x)
        x = (x-x.mean())/x.std()
        values = torch.squeeze(F.max_pool2d(x,28),(2,3))
        return torch.softmax(values, 1)

class Generator(nn.Module):
    def __init__(self,input_channels, output_channels, depth = 3):
        super(Generator, self).__init__()
        self.unet_res = UNet_Res(input_channels, output_channels, depth = depth)
    def forward(self,x):
        x = self.unet_res(x)
        x = F.tanh(x)
        return x

def train(args, model_gen, model_disc, device, train_loader, optimizer_gen, optimizer_disc, epoch):
    model_gen.train()
    model_disc.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
     
        optimizer_disc.zero_grad()
        optimizer_gen.zero_grad()
        im = data

        eps = torch.rand(1).item()
        noise_in = torch.randn_like(im)
        im_input = eps*noise_in+(1-eps)*im
        im_diff = im-im_input


        shape = list(im.shape)
        shape[1]=3

        epsp = torch.rand(1).item()/2
        gen_noise = model_gen(im_input)
        corr_im = im_input+im_diff
        noise_corr = torch.randn_like(im)
        noisy_corrected_image = epsp*noise_corr+(1-epsp)*corr_im

        noise_in1 = torch.randn_like(im)
        im_input1 = epsp*noise_in1+(1-epsp)*im


        if not discriminator:
            square_norm = torch.sum((gen_noise-im_diff)**2,(2,3))
            loss = -torch.sum(torch.sqrt(square_norm))
        else:
            randposreal = torch.randint(0,2,(shape[0],))
            disc_input = torch.empty(shape, device = im.device)
            disc_input[:,[0,],:,:]= im_input
            disc_input[torch.arange(shape[0])[:,None],1+randposreal[:,None],:,:]= im_input1
            disc_input[torch.arange(shape[0])[:,None],1+(1-randposreal)[:,None],:,:]= noisy_corrected_image
            disc_logits = model_disc(disc_input)
            loss = -torch.sum(disc_logits[torch.arange(shape[0]),randposreal])

        loss.backward()


        
        for name,param in model_gen.named_parameters():
            if param.grad!=None:
                param.grad = -param.grad
        
        optimizer_gen.step()
        optimizer_disc.step()

        if (batch_idx+1) % args.log_interval  == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


from PIL import Image
import time
def save_image(im,name):
    normalized = im-torch.min(im)
    normalized = normalized/torch.max(normalized)
    image_pil = transforms.ToPILImage()(normalized)
    image_pil.save(name)

nid= "vnoise"
def test(model_gen, model_disc, device, test_loader):
    model_gen.eval()
    model_disc.eval()
    test_loss = 0
    eps = 0.5
    ctime = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            im = data
            noise_in = torch.randn_like(im).to(im.device)
            im_input = eps*noise_in+(1-eps)*im
            im_diff = im-im_input

            shape = list(im.shape)
            shape[1] = 3
            
            gen_noise = model_gen(im_input)
             
            if not discriminator:
                square_norm = torch.sum((gen_noise-im_diff)**2,(2,3))
                loss = torch.sum(torch.sqrt(square_norm))
            else:
                randposreal = torch.randint(0,2,(shape[0],))
                disc_input = torch.empty(shape, device = im.device)
                disc_input[:,[0,],:,:]= im_input
                disc_input[torch.arange(shape[0])[:,None],1+randposreal[:,None],:,:]= im_diff
                disc_input[torch.arange(shape[0])[:,None],1+(1-randposreal)[:,None],:,:]=gen_noise
                disc_logits = model_disc(disc_input)
                loss = torch.sum(disc_logits[torch.arange(shape[0]),randposreal])


            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    orig = im
    noisy = im_input
    corr = im_input+gen_noise
    for i in range(5):
        save_image(noisy[i],"im/noisy"+str(i)+".jpg")
        save_image(corr[i],"im/corrected"+str(i)+".jpg")

    print('\nTest set {:.4f} Average loss: {:.4f} \n'.format(time.time()-ctime,test_loss))





class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

discriminator = False
def main():
    # Training settings
    args_dict = {'batch_size' : 64, 'test_batch_size' : 1000, 'epochs' : 14, 'lr' : 0.0005, 'gamma' : 0.7, 'no_cuda' :False, 'dry_run':False, 'seed': 1, 'log_interval' : 200, 'save_model' : False}
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

    model_gen = Generator(1,1, depth = 3).to(device)
    model_disc = Discriminator(3,2, depth = 3).to(device)
    optimizer_gen = optim.Adam(model_gen.parameters(), lr=args.lr)
    optimizer_disc = optim.Adam(model_disc.parameters(), lr=args.lr)

    scheduler_gen = StepLR(optimizer_gen, step_size=1, gamma=args.gamma)
    scheduler_disc = StepLR(optimizer_disc, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model_gen, model_disc, device, train_loader, optimizer_gen, optimizer_disc, epoch)
        test(model_gen, model_disc, device, test_loader)
        scheduler_gen.step()
        scheduler_disc.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
