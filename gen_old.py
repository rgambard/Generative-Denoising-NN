import argparse
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
class UNet_Res(nn.Module):
    def __init__(self,input_channels, output_channels, depth = 3):
        super(UNet_Res, self).__init__()
        int_channels = 16
        self.unetin = UNet(input_channels, int_channels)
        self.unetout = UNet(int_channels,output_channels)
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
    def __init__(self,input_channels, output_channels):
        super(UNet, self).__init__()
        self.conv11 = nn.Conv2d(input_channels,16, 3, 1, 1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv21 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv22 = nn.Conv2d(16,16, 3, 1, 1)
        self.conv31 = nn.Conv2d(16,16, 3, 1, 1)
        self.conv32 = nn.Conv2d(16,16, 3, 1, 1)
        self.conv33 = nn.Conv2d(16,16, 3, 1, 1)
        self.convu21 = nn.Conv2d(16,16, 3, 1, 1)
        self.convu22 = nn.Conv2d(16,16, 3, 1, 1)
        self.convu11 = nn.Conv2d(16,16, 3, 1, 1)
        self.convu12 = nn.Conv2d(16, 16, 3, 1, 1)
        self.convout = nn.Conv2d(16,output_channels,1,1)
        torch.nn.init.kaiming_normal_(self.convout.weight,nonlinearity = 'linear')


    def forward(self, x):
        x= (x-x.mean())/x.std()
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


def train(args, model_gen, model_disc, device, train_loader, optimizer_gen, optimizer_disc, epoch):
    model_gen.train()
    model_disc.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
     
        optimizer_disc.zero_grad()
        optimizer_gen.zero_grad()
        im = data
        noise = torch.randn_like(im)
        eps = torch.rand(1).item()


        shape = list(im.shape)
        shape[1] = 2

        gen_im = model_gen(noise)
        gen_im = (gen_im-gen_im.mean())/gen_im.std()
        #gen_im = noise+gen_im
        gen_im_base = gen_im.detach()
        indices_rand = torch.randint(0,2,im.shape, device = device)
        one_hot_indices = torch.zeros(shape, device = device)
        one_hot_indices[torch.arange(shape[0])[:,None,None,None],indices_rand,torch.arange(shape[2])[None,None,:,None], torch.arange(shape[3])[None,None,None,:]] = 1
        new_im = torch.zeros(shape, device = device)
        new_im[torch.arange(shape[0])[:,None,None,None],indices_rand,torch.arange(shape[2])[None,None,:,None], torch.arange(shape[3])[None,None,None,:]] = im
        new_im[torch.arange(shape[0])[:,None,None,None],1-indices_rand,torch.arange(shape[2])[None,None,:,None], torch.arange(shape[3])[None,None,None,:]] =gen_im
        #new_im[torch.arange(shape[0])[:,None,None,None],1-indices_rand,torch.arange(shape[2])[None,None,:,None], torch.arange(shape[3])[None,None,None,:]] = noise

        estimated_one_hot_indices = model_disc(new_im)
        estimated_one_hot_indices = F.softmax(estimated_one_hot_indices,1)

        disc_loss = -torch.sum(estimated_one_hot_indices[torch.arange(shape[0])[:,None,None,None],indices_rand,torch.arange(shape[2])[None,None,:,None], torch.arange(shape[3])[None,None,None,:]])
        disc_loss.backward()
        for name,param in model_gen.named_parameters():
            if param.grad!=None:
                param.grad = -param.grad
        optimizer_gen.step()
        optimizer_disc.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), disc_loss.item()))
            save_image(gen_im_base[0],'im/gen_0'+nid+'.jpg')
            save_image(gen_im_base[1],'im/gen_1'+nid+'.jpg')
            save_image(gen_im_base[2],'im/gen_2'+nid+'.jpg')
            if args.dry_run:
                break


from PIL import Image
import time
def save_image(im,name):
    normalized = im-torch.min(im)
    normalized = normalized/torch.max(normalized)
    image_pil = transforms.ToPILImage()(normalized)
    image_pil.save(name)

nid= "v1"
def test(model_gen, model_disc, device, test_loader):
    model_gen.eval()
    model_disc.eval()
    test_loss = 0
    ctime = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            im = data
            noise = torch.randn_like(im).to(im.device)
            shape = list(im.shape)
            shape[1] = 2

            gen_im = model_gen(noise)
            gen_im = (gen_im-gen_im.mean())/torch.std(gen_im)
            indices_rand = torch.randint(0,2,im.shape).to(im.device)
            one_hot_indices = torch.zeros(shape).to(im.device)
            one_hot_indices[torch.arange(shape[0])[:,None,None,None],indices_rand,torch.arange(shape[2])[None,None,:,None], torch.arange(shape[3])[None,None,None,:]] = 1
            new_im = torch.zeros(shape).to(im.device)
            new_im[torch.arange(shape[0])[:,None,None,None],indices_rand,torch.arange(shape[2])[None,None,:,None], torch.arange(shape[3])[None,None,None,:]] = im
            new_im[torch.arange(shape[0])[:,None,None,None],1-indices_rand,torch.arange(shape[2])[None,None,:,None], torch.arange(shape[3])[None,None,None,:]] = gen_im

            estimated_one_hot_indices = model_disc(new_im)
            estimated_one_hot_indices = F.softmax(estimated_one_hot_indices,1)

            disc_loss = -torch.sum(estimated_one_hot_indices[torch.arange(shape[0])[:,None,None,None],indices_rand,torch.arange(shape[2])[None,None,:,None], torch.arange(shape[3])[None,None,None,:]])
            test_loss += disc_loss

    test_loss /= len(test_loader.dataset)
    save_image(gen_im[1],'im/gen_1.jpg')
    save_image(gen_im[2],'im/gen_2.jpg')
    save_image(gen_im[3],'im/gen_3.jpg')
    save_image(gen_im[4],'im/gen_4.jpg')

    print('\nTest set {:.4f} Average loss: {:.4f} \n'.format(time.time()-ctime,test_loss))





class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

def main():
    # Training settings
    args_dict = {'batch_size' : 64, 'test_batch_size' : 1000, 'epochs' : 14, 'lr' : 0.005, 'gamma' : 0.7, 'no_cuda' :False, 'dry_run':False, 'seed': 1, 'log_interval' : 200, 'save_model' : False}
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

    model_gen = UNet_Res(1,1, depth = 1).to(device)
    model_disc = UNet_Res(2,2, depth = 1).to(device)
    optimizer_gen = optim.Adam(model_gen.parameters(), lr=args.lr/10)
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
