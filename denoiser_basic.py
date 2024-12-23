import argparse
import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from unet import  Denoiser
from utils import save_image, dotdict

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


# the sigmas in the training loop
sigmas = torch.linspace(1,0.01,1000)

# forward pass to compute score
def forward(data, model, sigmas_batch = None):
    im = data

    ind_randoms= torch.randint(0, sigmas.shape[0], (data.shape[0],), device = data.device)
    
    noise_in = torch.randn_like(im)
    if sigmas_batch is None:
        sigmas_batch = sigmas[ind_randoms]

        im_input = torch.sqrt(sigmas_batch[:,None,None,None])*noise_in+(torch.sqrt(1-sigmas_batch[:,None,None,None]))*im
    else :
        im_input = im
    

    mod_input = im_input
    pred_im = model(mod_input)
    tpred_im = 1/math.sqrt(2)*pred_im-im_input
    pred_score = -(im_input-torch.sqrt(1-sigmas_batch[:,None,None,None])*tpred_im)/sigmas_batch[:,None,None,None]
    im_corrected = tpred_im

    score = -torch.sqrt(sigmas_batch[:,None,None,None])*noise_in/sigmas_batch[:,None,None,None]
    square_norm = torch.sum((tpred_im-im)**2,(1,2,3)) # square norm of loss per image
    loss = torch.sum(square_norm)
    return loss, im_input, im_corrected, pred_score


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
     
        optimizer.zero_grad()

        loss, im_input, im_corrected, pred_score= forward(data, model)

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
    #if True:
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            im = data
            loss, im_input, corr, pred_score= forward(data, model)

            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set {:.4f} Average loss: {:.4f} \n'.format(time.time()-ctime,test_loss))
    orig = im
    noisy = im_input
    save_image(noisy[:10],"im/noisy.jpg")
    save_image(corr[:10],"im/corrected.jpg")
    save_image(orig[:10],"im/originals.jpg")
    gen_shape = list(im.shape)
    gen_shape[0] = 64
    gen_im = sampleLangevin(model, device, gen_shape)
    save_image(gen_im, "im/generatedc.jpg")

def sampleLangevin(model,device, im_shape, epsilon = 0.01, T=1):
    with torch.no_grad():
        xt = torch.randn(im_shape, device = device)
        for i in range(0,sigmas.shape[0]):
            sigmai = sigmas[i]
            alphai = epsilon#*sigmai
            for t in range(T):
                zt = torch.randn_like(xt)
                loss, im_input, corr, pred_score= forward(xt,model, sigmas_batch = sigmai*torch.ones(xt.shape[0], device = xt.device))

                xt = xt + alphai/2*pred_score.detach()+math.sqrt(alphai)*zt
                
    print("images generated ! ")
    return xt




def main():
    global sigmas
    # Training settings
    args_dict = {'batch_size' : 64, 'test_batch_size' :64, 'epochs' :1000, 'lr' : 0.0002, 'gamma' : 0.995, 'no_cuda' :False, 'dry_run':False, 'seed': 1, 'log_interval' : 200, 'save_model' :True, 'only_test':False, 'model_path':"denoisercelebab.pt", 'load_model_from_disk':False, 'dataset':"CELEBA", 'test':False}
    args = dotdict(args_dict)
    parser = argparse.ArgumentParser(description="A simple argument parser example.")

    # Add arguments
    parser.add_argument('--dataset', type=str, required=True, help='Dataset can be one of MNIST, CIFAR, CELEBA')
    parser.add_argument('--test', type= str, required = False, help='wether to only test a model, requires path to the testing weights')

    # Parse the arguments
    margs = parser.parse_args()
    args.dataset = margs.dataset
    if margs.test is not None:
        print("TEST")
        args.test = True
        print(args.test)
        args.model_path = margs.test

    if args.test:
        args.load_model_from_disk = True
        args.only_test = True
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

    dataset = args.dataset
    if dataset == "CIFAR":
        transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),#mean, std
        transforms.RandomHorizontalFlip(p=0.5)
        ])

        dataset1 = datasets.CIFAR10(root='data/', train=True, download=True, transform=transform)
        dataset2 = datasets.CIFAR10(root='data/', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
        model = Denoiser(3,3).to(device)
    elif dataset == "CELEBA":
        transform = transforms.Compose([transforms.Resize((64,64)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])])
        dataset1 = datasets.CelebA("./data/celeba", split = 'train',download=False, transform=transform)
        dataset2 = datasets.CelebA("./data/celeba", split = 'test', download=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
        model = Denoiser(3,3).to(device)

    elif dataset=="MNIST":
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

        model = Denoiser(1,1).to(device)

    if args.load_model_from_disk:
        model.load_state_dict(torch.load(args.model_path, weights_only= True))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    sigmas = sigmas.to(device)
    for epoch in range(1, args.epochs + 1):
        if not args.only_test:
            train(args, model , device, train_loader, optimizer, epoch)
            scheduler.step()
            if args.save_model:
                torch.save(model.state_dict(), args.model_path)
        if epoch%3 == 0:
            test(model,  device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), args.model_path)


main()
