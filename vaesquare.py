import torch
import time
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.optim.lr_scheduler import StepLR
from utils import save_image , dotdict
from unet import VAE

timestep = 0
def forward(model, data):
    global timestep
    im = data
    noise = torch.randn_like(im)
    eps = torch.rand(data.shape[0], device = data.device)
    eps = eps*0

    im_in = (1-eps[:,None,None,None])*im+noise*eps[:,None,None,None]
    im_out, loss_z = model(im_in)
    loss_im = torch.sum((im_in-im_out)**2, axis = (1,2,3))
    value_z = max(0,min(1, (timestep-200)/1000))
    timestep+=1
    #print(value_z)
    loss = loss_im+value_z*loss_z
    loss = loss*(1-eps)
    loss = torch.sum(loss)
    return im_in, im_out, loss, torch.sum(loss_im), torch.sum(loss_z)

def train(args, model,  device, train_loader, optimizer_enc, optimizer_dec, epoch):
    model.train()
    mean_loss = 0
    mean_loss_im = 0
    mean_loss_z = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
     
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()

        im_in, im_out, loss, loss_im, loss_z = forward(model, data)
        
        mean_loss += loss.item()
        mean_loss_im += loss_im.item()
        mean_loss_z += loss_z.item()
        loss.backward()

        optimizer_enc.step()
        optimizer_dec.step()

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

    
    for temp in [0.01,0.1,0.5,1]:
        im_gen  = model.gen(device, temp = temp)
        save_image(im_gen, "im/vae_generated"+str(temp)+".jpg")
    for i in range(1):
        save_image(im_in[i], "im/vae_input"+str(i)+".jpg")
        save_image(im_out[i], "im/vae_output"+str(i)+".jpg")



def main():
    # Training settings
    args_dict = {'batch_size' : 64, 'test_batch_size' : 1000, 'epochs' : 10, 'lr' : 0.0005, 'gamma' : 0.9, 'no_cuda' :False, 'dry_run':False, 'seed': 1, 'log_interval' : 100, 'save_model' :True}
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

    model = VAE(1,16,16,1, depth =1).to(device)
    optimizer_enc = optim.Adam(model.encoderblocks.parameters(), lr=args.lr)
    optimizer_dec = optim.Adam(model.decoderblocks.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer_enc, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer_enc, optimizer_dec, epoch)
        test(model,  device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "vaesquarel.pt")


if __name__ == '__main__':
    main()
