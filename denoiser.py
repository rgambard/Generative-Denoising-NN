import argparse
import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from unet import UNet_Res, UNet, Denoiser
from utils import save_image, dotdict

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

def forward(data, model):
    im = data

    eps = torch.rand(data.shape[0], device = data.device)*0.95
    noise_in = torch.randn_like(im)
    im_input = eps[:,None,None,None]*noise_in+(1-eps[:,None,None,None])*im
    mod_input = torch.cat((im_input, eps[:,None,None,None].expand(im.shape)), dim=1)

    gen_noise = model(mod_input)
    im_corrected = 1/(1-eps[:,None,None,None])*(im_input-eps[:,None,None,None]*gen_noise)

    square_norm = torch.sum((im_corrected-im)**2,(2,3))
    loss = torch.sum(torch.sqrt(square_norm))
    return loss, im_input, im_corrected


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
     
        optimizer.zero_grad()

        loss, im_input, im_corrected= forward(data, model)

        loss.backward()
        running_loss += loss.item()
        
        optimizer.step()

        if (batch_idx+1) % args.log_interval  == 0:
            running_loss = running_loss/(args.log_interval*args.batch_size)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), running_loss))
            running_loss = 0
            if args.dry_run:
                break

nid= "vnoise"
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    ctime = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            im = data
            loss, im_input, corr= forward(data, model)

            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set {:.4f} Average loss: {:.4f} \n'.format(time.time()-ctime,test_loss))
    orig = im
    noisy = im_input
    save_image(noisy[:10],"im/noisy.jpg")
    save_image(corr[:10],"im/corrected.jpg")
    save_image(orig[:10],"im/originals.jpg")





def main():
    # Training settings
    args_dict = {'batch_size' : 64, 'test_batch_size' : 1000, 'epochs' : 5, 'lr' : 0.001, 'gamma' : 0.7, 'no_cuda' :False, 'dry_run':False, 'seed': 1, 'log_interval' : 200, 'save_model' :True}
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

    model = Denoiser(2,1,depth = 5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model , device, train_loader, optimizer, epoch)
        test(model,  device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "denoiser.pt")


if __name__ == '__main__':
    main()
