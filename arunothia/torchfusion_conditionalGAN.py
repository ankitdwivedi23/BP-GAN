from torchfusion.gan.learners import *
from torchfusion.gan.applications import *
from torch.optim import *
from torchfusion.datasets import fashionmnist_loader
import torch.cuda as cuda
import torch
import torch.nn as nn
from torchvision import datasets 
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.distributions import Normal
import argparse

parser = argparse.ArgumentParser(description='Load the desired dataset')

parser.add_argument('--BATCH_SIZE', type=int, default=128,
                    help='batch size, default value is 128')

parser.add_argument('--LATENT_SIZE', type=int, default=128,
                    help='Latent size, default value is 128')

parser.add_argument('--SHOW_EVERY', type=int, default=500,
                    help='Show every, default value is 500')

parser.add_argument('--NUM_EPOCHS', type=int, default=100,
                    help='Number of Epochs, default value is 100')

args = parser.parse_args()

G = StandardGenerator(output_size=(3,64,64),latent_size=args.LATENT_SIZE,num_classes=6)
#G = ResGenerator(output_size=(3,64,64),num_classes=6,latent_size=args.LATENT_SIZE, \
#        kernel_size=3,activation=nn.LeakyReLU(),conv_groups=1,attention=False,dropout_ratio=0)
D = StandardProjectionDiscriminator(input_size=(3,64,64),apply_sigmoid=False,num_classes=6)
#D = ResProjectionDiscriminator(input_size=(3,64,64),num_classes=6,kernel_size=3,activation=nn.LeakyReLU(), \
#        attention=True,apply_sigmoid=False,conv_groups=1,dropout_ratio=0)

if cuda.is_available():
    G = nn.DataParallel(G.cuda())
    D = nn.DataParallel(D.cuda())

g_optim = Adam(G.parameters(),lr=0.0002,betas=(0.5,0.999))
d_optim = SGD(D.parameters(), lr=0.005)
#d_optim = Adam(D.parameters(),lr=0.0002,betas=(0.5,0.999))


book_data  = datasets.ImageFolder(root='data/Task2_split/Task2_Split/train',
                               transform=transforms.Compose([
                                   transforms.Resize((64, 64)),
                                   transforms.ToTensor()
                           ]))

book_dataset =  torch.utils.data.DataLoader(book_data,
                                        batch_size=args.BATCH_SIZE,
                                        shuffle=True)


learner = RStandardGanLearner(G,D)

if __name__ == "__main__":
    learner.train(book_dataset,num_classes=6,gen_optimizer=g_optim,disc_optimizer=d_optim,save_outputs_interval=args.SHOW_EVERY, \
        model_dir="./genre-gan",latent_size=args.LATENT_SIZE,num_epochs=args.NUM_EPOCHS,batch_log=False)