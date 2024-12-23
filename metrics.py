import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models.inception import Inception_V3_Weights
from scipy import linalg
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm



def preprocess_images(images, dataset):
    # We ensure input shape is [batch_size, channels, height, width]
    if len(images.shape) != 4:
        raise ValueError("Input images must have shape [batch_size, channels, height, width]")

    # CIFAR dataset preprocessing
    if dataset == "CIFAR":
        transform = Compose([
            lambda x: x / 255.0,  # Scale pixel values to [0, 1]
            Resize((299, 299)),
            Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),  # CIFAR normalization
        ])
    elif dataset == "MNIST":
        transform = Compose([
            lambda x: x.repeat(1, 3, 1, 1),  # Convert grayscale to 3 channels
            lambda x: x / 255.0,  # Scale pixel values to [0, 1]
            Resize((299, 299)),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # MNIST normalization
        ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Apply transforms sequentially to the images
    for t in transform.transforms:
        images = t(images)
    
    return images



def calculate_inception_score_first_version(images, batch_size=32, splits=10,dataset="MNIST"):
    model = inception_v3(weights='IMAGENET1K_V1', transform_input=False).eval()
    model = model.to('cuda')  
    images = preprocess_images(images,dataset)

    # Calculate predictions in batches
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to('cuda')  
            preds.append(F.softmax(model(batch), dim=1).cpu())  
    preds = torch.cat(preds, dim=0)

    scores = []
    N = preds.size(0)
    for i in range(splits):
        part = preds[i * (N // splits): (i + 1) * (N // splits), :]
        p_y = part.mean(dim=0)
        kl = part * (torch.log(part) - torch.log(p_y.unsqueeze(0)))
        scores.append(torch.exp(kl.sum(dim=1).mean()))
    return torch.tensor(scores).mean(), torch.tensor(scores).std()


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features 
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
    

def calculate_inception_score(images, model, batch_size=32, splits=10):
    """
    Calculates the Inception Score for a set of images using the given model.
    
    Parameters:
    - images (torch.Tensor): Tensor of images with shape (N, C, H, W).
    - model (nn.Module): InceptionV3 model returning final average pooling features.
    - batch_size (int): Batch size for processing.
    - splits (int): Number of splits for calculating the score.
    
    Returns:
    - mean (float): Mean of the Inception Score.
    - std (float): Standard deviation of the Inception Score.
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    images = images.to(device)

    # DataLoader to handle batches
    dataloader = DataLoader(images, batch_size=batch_size)
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)[-1]  # Get the final average pooling features
            preds.append(F.softmax(output, dim=1).cpu())
    
    preds = torch.cat(preds, dim=0)  # Concatenate all predictions

    scores = []
    N = preds.size(0)
    for i in range(splits):
        part = preds[i * (N // splits): (i + 1) * (N // splits), :]
        p_y = part.mean(dim=0)  # Marginal probability
        kl = part * (torch.log(part) - torch.log(p_y.unsqueeze(0)))  # KL divergence
        scores.append(torch.exp(kl.sum(dim=1).mean()))
    
    return torch.tensor(scores).mean().item(), torch.tensor(scores).std().item()


def get_activation(dataloader, 
                   model, 
                   preprocess, # Preprocessing Transform for InceptionV3
                   device = 'cpu'
                  ):
    """
    Given Dataloader and Model, Generate N X 2048
    Dimensional activation map for N data points
    in dataloader.
    """
    
    # Set model to evaluation Mode
    model.to(device)
    model.eval()
    
    # Save activations
    pred_arr = np.zeros((len(dataloader), 2048))
    
    # Batch Size
    batch_size = dataloader.shape[0]//4
    # Loop over Dataloader
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            # Transform the Batch according to Inceptionv3 specification
            batch = torch.stack([preprocess(img) for img in batch]).to(device)
            
            # Predict
            pred = model(batch).cpu().numpy()
            
            # Store
            pred_arr[i*batch_size : i*batch_size + batch.size(0), :] = pred
            
    return pred_arr

#--------------------------------------------------------------------

def calculate_activation_statistics(dataloader, 
                                    model, 
                                    preprocess, 
                                    device='cpu'
                                   ):
    """
    Get mean vector and covariance matrix of the activation maps.
    """
    
    # Get activation maps
    act = get_activation(dataloader, 
                         model, 
                         preprocess, # Preprocessing Transform for InceptionV3
                         device
                       )
    # Mean
    mu = np.mean(act, axis=0)
    
    # Covariance Metric
    sigma = np.cov(act, rowvar=False)
    
    return mu, sigma

#----------------------------------------------------------------

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    
    """
    Given Mean and Sigma of Real and Generated Data,
    it calculates FID between them using:
     
     d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
     
    """
    # Make sure they have appropriate dims
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Handle various cases
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def get_FID(mnist_dl_real , mnist_dl_generated) :
    # transformations on the images
    transform_inception = transforms.Compose([
    transforms.Lambda(lambda x: (x + 1.0)/2.0), # [-1, 1] => [0, 1]
    transforms.ToPILImage(), 
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB format
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization

    ])
    model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

    model.fc = nn.Identity()

    # Mean and Sigma For Generated Data
    mu2, sigma2 = calculate_activation_statistics( mnist_dl_generated, model, preprocess = transform_inception, device='cuda')
    # Mean and Sigma for Test Data
    mu1, sigma1 = calculate_activation_statistics(mnist_dl_real, model, preprocess = transform_inception, device='cuda')

    # Calculate FID
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f'FID-Score: {fid}')
