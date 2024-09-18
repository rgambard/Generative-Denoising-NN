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
        self.convin = nn.Conv2d(input_channels, int_channels,1)
        self.convout = nn.Conv2d(int_channels, output_channels,1)
        torch.nn.init.kaiming_normal_(self.convin.weight,nonlinearity = 'linear')
        torch.nn.init.kaiming_normal_(self.convout.weight,nonlinearity = 'linear')
        self.unets = nn.ModuleList([UNet(int_channels, int_channels) for i in range(depth)])
    def forward(self,x):
        x = self.convin(x)
        i = 1
        for unet in self.unets:
            x = x+unet(x/math.sqrt(i))
            i+=1
        out = self.convout(x/math.sqrt(i))
        return out
        


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
        """
        torch.nn.init.kaiming_normal_(self.conv11.weight,nonlinearity = 'linear')
        torch.nn.init.kaiming_normal_(self.conv12.weight,nonlinearity = 'linear')
        torch.nn.init.kaiming_normal_(self.conv21.weight,nonlinearity = 'linear')
        torch.nn.init.kaiming_normal_(self.conv22.weight,nonlinearity = 'linear')
        torch.nn.init.kaiming_normal_(self.conv31.weight,nonlinearity = 'linear')
        torch.nn.init.kaiming_normal_(self.conv32.weight,nonlinearity = 'linear')
        torch.nn.init.kaiming_normal_(self.conv33.weight,nonlinearity = 'linear')
        torch.nn.init.kaiming_normal_(self.convu21.weight,nonlinearity = 'linear')
        torch.nn.init.kaiming_normal_(self.convu22.weight,nonlinearity = 'linear')
        torch.nn.init.kaiming_normal_(self.convu11.weight,nonlinearity = 'linear')
        torch.nn.init.kaiming_normal_(self.convu12.weight,nonlinearity = 'linear')
        """
        #torch.nn.init.kaiming_normal_(self.convout.weight,nonlinearity = 'linear')


    def forward(self, x):
        x11 = F.relu(self.conv11(x))
        x12 = F.relu(self.conv12(x11))
        x20 = F.interpolate(x12,scale_factor = 0.5)
        x21 = F.relu(self.conv21(x20))
        x22 = F.relu(self.conv22(x21))
        x30 = F.interpolate(x22,scale_factor = 0.5)
        x31 = F.relu(self.conv31(x30))
        x32 = F.relu(self.conv32(x31))
        x33 = F.relu(self.conv33(x31))
        xu20 = F.interpolate(x33,scale_factor = 2)
        xu21 = (xu20+x22)/math.sqrt(2)
        xu22 = F.relu(self.convu21(xu21))
        xu23 = F.relu(self.convu22(xu22))
        xu10 = F.interpolate(xu23,scale_factor = 2)
        xu11 = (xu10+x12)/math.sqrt(2)
        xu12 = F.relu(self.convu11(xu11))
        xu13 = F.relu(self.convu12(xu12))

        out = self.convout(xu13)
        return out


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        noise = torch.randn_like(data)
        epsilon = torch.rand(1).item()
        eps = torch.ones_like(data)*epsilon
        data = torch.cat((eps,(1-epsilon)*data+epsilon*noise),1)
        output = model(data)
        loss = torch.sum((noise-output)**2)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


from PIL import Image
def save_image(im,name):
    normalized = im-torch.min(im)
    normalized = normalized/torch.max(normalized)
    image_pil = transforms.ToPILImage()(normalized)
    image_pil.save(name)

def test(model, device, test_loader):
    model.eval()
    for epsilon in (0.4, 0.8):
        test_loss = 0
        base_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                noise = torch.randn_like(data)
                eps = torch.ones_like(data)*epsilon
                noisy_input = torch.cat((eps,(1-epsilon)*data+epsilon*noise),1)
                output = model(noisy_input)
                test_loss += torch.sum((noise-output)**2)
                base_loss += torch.sum(noise**2)

        test_loss /= len(test_loader.dataset)
        base_loss /= len(test_loader.dataset)
        orig = data[0]
        noisy = noisy_input[0][1]
        pred_noise = output[0]
        corr = 1/(1-epsilon)*(noisy-epsilon*pred_noise)
        save_image(orig,'im/eps_{:.2f}_orig.jpg'.format(epsilon))
        save_image(noisy,'im/eps_{:.2f}_noisy.jpg'.format(epsilon))
        save_image(corr,'im/eps_{:.2f}_corr.jpg'.format(epsilon))

        print('\nTest set epsilon {:.2f}: Average loss: {:.4f} base {:.4f} \n'.format(
            epsilon,test_loss, base_loss))
    # generating image !!!!
    for steps in []:
        image = torch.randn_like(data[:5,:,:,:])
        for epsilon in torch.linspace(0.9,0.0,steps):
            with torch.no_grad():
                eps = torch.ones_like(image)*epsilon
                noise = torch.randn_like(image)
                noisy_input = torch.cat((eps,(1-epsilon)*image+epsilon*noise),1)
                output = model(noisy_input)
                image = 1/(1-epsilon)*(image-epsilon*output)
                image = (image-image.mean())/torch.std(image)
        save_image(image[0],"im/generated_{:.1f}.jpg".format(steps))





class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

def main():
    # Training settings
    args_dict = {'batch_size' : 64, 'test_batch_size' : 1000, 'epochs' : 14, 'lr' : 1.0, 'gamma' : 0.7, 'no_cuda' : False, 'dry_run':False, 'seed': 1, 'log_interval' : 200, 'save_model' : False}
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

    model = UNet_Res(2,1, depth = 5).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
