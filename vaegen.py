import torch
import time
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.optim.lr_scheduler import StepLR
from utils import save_image , dotdict
from unet import VAE, GenImage

def forward(model, model_gen, data):
    im = model_gen(data)
    noise = torch.randn_like(im)
    eps = 0.05

    im_in = im#+noise*eps
    im_out, loss_z = model(im_in)
    loss_im = torch.sum((im_in-im_out)**2)*10
    loss = loss_im+torch.sum(loss_z)
    return im_in, im_out, loss, loss_im, torch.sum(loss_z)

def train(args, model, model_gen,  device, train_loader, optimizer, epoch):
    model.eval()
    model_gen.train()
    mean_loss = 0
    mean_loss_im = 0
    mean_loss_z = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
     
        optimizer.zero_grad()
        data = torch.randn_like(data)

        im_in, im_out, loss, loss_im, loss_z = forward(model, model_gen, data)
        
        mean_loss += loss.item()
        mean_loss_im += loss_im.item()
        mean_loss_z += loss_z.item()
        loss.backward()

        optimizer.step()

        if (batch_idx+1) % args.log_interval == 0:
            save_image(im_in[:5], "im/generated_curr.jpg")
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


def test(model, model_gen, device, test_loader):
    model.eval()
    model_gen.eval()
    test_loss = 0
    test_loss_im = 0
    test_loss_z = 0
    ctime = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            data = torch.randn_like(data)
            im_in, im_out, loss, loss_im, loss_z  = forward(model, model_gen, data)

            test_loss += loss
            test_loss_im += loss_im
            test_loss_z += loss_z

    test_loss /= len(test_loader.dataset)
    test_loss_im /= len(test_loader.dataset)
    test_loss_z /= len(test_loader.dataset)

    print('\nTest set {:.4f} Average loss: {:.4f} lossz : {:.4f} lossim : {:.4f} \n'.format(time.time()-ctime,test_loss, test_loss_z, test_loss_im))
    save_image(im_in[:5], "im/generated.jpg")
    save_image(im_out[:5], "im/vae-generated.jpg")

    



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

    model = VAE(1,16,16,4, depth =1).to(device)
    model.load_state_dict(torch.load("vaesquarel.pt", weights_only=True))
    model_gen = GenImage(1,1).to(device)
    optimizer = optim.Adam(model_gen.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, model_gen, device, train_loader, optimizer, epoch)
        test(model, model_gen,  device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
