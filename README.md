repository created for the course "Probabilistic graphical models" from MVA master.

The main file to run the code is denoiser_basic.py . The NCSN architecture is stored in Unet.py and weights were saved from our last attempts ( denoisiermnist.pt for MNIST and denoisercifar.pt for CIFAR10 ). The sript could be run on test mode only.

Some additional notebooks contain our other experiments : 
- ddpm-from-scratch.ipynb is a notebook that implements DDPM in Pytorch ( a public notebook found on Kaggle we used and made some experiments on )
- toy_langevin_dynamics.ipynb is a notebook that tests score matching ( through an MLP + annealed langevin dynamics ) ( code inspired from : https://github.com/JeongJiHeon/ScoreDiffusionModel/tree/main/NCSN )
