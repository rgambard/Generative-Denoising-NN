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


sigmas = torch.Tensor([0.01,0.2,0.5,0.8])
lambdassigmas = sigmas**2

def forward(data, model):
    im = data

    # parameters of the langevin dynamics steps
    ind_randoms= torch.randint(0, sigmas.shape[0], data.shape[0], device = data.device)
    sigmas_batch = sigmas[ind_randoms]
    lambdas_batch = lambdassigmas[ind_randoms]

    noise_in = torch.randn_like(im)
    im_input = sigmas_batch[:,None,None,None]*noise_in+im # noisy images
    # we append the sigmas to the model input as a new dimension of the image
    mod_input = torch.cat((im_input, sigmas_batch[:,None,None,None].expand(im.shape)), dim=1)

    pred_score = model(mod_input)/sigmas_batch[:,None,None,None] # we divide by sigma such that the variance 
    # of the last layer is constant and equal to 1
    # corrected image using the score expression
    im_corrected = im_input+sigmas_batch[:,None,None,None]**2*pred_score

    score = noise_in/sigmas_batch[:,None,None,None]**2
    square_norm = torch.sum((pred_score -score)**2,(1,2,3)) # square norm of loss per image
    loss = torch.sum(lambdas_batch*square_norm)
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
    gen_shape = im.shape
    gen_shape[0] = 10
    gen_im = sampleLangevin(model, device, im.shape)
    save_image(gen_im)


def sampleLangevin(model,device, im_shape, epsilon = 0.2, T=20):
    with torch.no_grad():
        xt = torch.randn(im_shape)
        for i in range(sigmas.shape[0]):
            sigmai = sigmas[i]
            alpha = epsilon*sigmai**2/sigmas[-1]**2
            for t in range(T):
                zt = torch.randn(im_shape)
                score = model(xt)
                xt = xt + alphai/2*score+torch.sqrt(alphai)*zt
    return xt




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

    # loading dataset
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

    model = Denoiser(2,1,depth = 1).to(device)
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
