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


sigmas = 5*torch.pow(torch.ones(10)*0.6,torch.arange(10))
#sigmas = torch.linspace(5,0.01,15)
print(sigmas)


def forward(data, model):
    im = data

    # parameters of the langevin dynamics steps
    ind_randoms= torch.randint(0, sigmas.shape[0], (data.shape[0],), device = data.device)
    sigmas_batch = sigmas[ind_randoms]

    noise_in = torch.randn_like(im)
    im_input = (sigmas_batch[:,None,None,None]*noise_in+im)
    #im_input_norm = torch.sqrt(torch.sum(im_input**2,dim=(1,2,3)))
    #im_input_renormalized = (im_input-torch.mean(im_input,dim=(1,2,3))[:,None,None,None])/im_input_norm[:,None,None,None]
    # we append the sigmas to the model input as a new dimension of the image
    mod_input = torch.cat((im_input, sigmas_batch[:,None,None,None].expand(im.shape)), dim=1)

    pred_score = model(mod_input)/sigmas_batch[:,None,None,None] # we divide by sigma such that the variance 
    # of the last layer is constant and equal to 1
    # corrected image using the score expression
    im_corrected = im_input+sigmas_batch[:,None,None,None]**2*pred_score

    score = -noise_in/sigmas_batch[:,None,None,None]
    square_norm = torch.sum((pred_score -score)**2,(1,2,3)) # square norm of loss per image
    loss = torch.sum(sigmas_batch**2*square_norm)
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
    gen_shape = list(im.shape)
    gen_shape[0] = 10
    gen_im = sampleLangevin(model, device, gen_shape)
    save_image(gen_im, "im/generated.jpg")


def sampleLangevin(model,device, im_shape, epsilon = 25e-5, T=100):
    print("generating images...")
    with torch.no_grad():
        xt = torch.randn(im_shape, device = device)*(1+sigmas[0])
        for i in range(sigmas.shape[0]):
            sigmai = sigmas[i]
            alphai = epsilon*sigmai**2/sigmas[-1]**2
            for t in range(T):
                zt = torch.randn_like(xt)
                xt_norm = torch.sqrt(torch.sum(xt**2,dim=(1,2,3)))
                xt_renormalized= (xt-torch.mean(xt, dim=(1,2,3))[:,None,None,None])/xt_norm[:,None,None,None]
                mod_input = torch.cat((xt,sigmai[None,None,None,None].expand(xt.shape)), dim=1)
                score = model(mod_input)/sigmai
                xt = xt + alphai/2*score+torch.sqrt(alphai)*zt
    print("images generated ! ")
    return xt




def main():
    global sigmas
    # Training settings
    args_dict = {'batch_size' : 64, 'test_batch_size' : 128, 'epochs' :40, 'lr' : 0.001, 'gamma' : 0.9, 'no_cuda' :False, 'dry_run':False, 'seed': 1, 'log_interval' : 200, 'save_model' :True, 'only_test':False, 'model_path':"denoiser.pt", 'load_model_from_disk':False}
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
    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset1 = datasets.CelebA(root='data/celeba', split='train', download=True, transform=transform)
    dataset2 = datasets.CelebA(root='data/celeba', split='test', download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Denoiser(2,1,depth = 3).to(device)
    if args.load_model_from_disk:
        model.load_state_dict(torch.load(args.model_path, weights_only= True))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    sigmas = sigmas.to(device)
    for epoch in range(1, args.epochs + 1):
        if not args.only_test:
            train(args, model , device, train_loader, optimizer, epoch)
            scheduler.step()
        test(model,  device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), args.model_path)


main()
