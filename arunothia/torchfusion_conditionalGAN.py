from torchfusion.gan.learners import *
from torchfusion.gan.applications import StandardGenerator,StandardProjectionDiscriminator
from torch.optim import Adam
from torchfusion.datasets import fashionmnist_loader
import torch.cuda as cuda
import torch
import torch.nn as nn
from torchvision import datasets 
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.distributions import Normal

G = StandardGenerator(output_size=(3,64,64),latent_size=3*64*64,num_classes=6)
D = StandardProjectionDiscriminator(input_size=(3,64,64),apply_sigmoid=False,num_classes=6)

if cuda.is_available():
    G = nn.DataParallel(G.cuda())
    D = nn.DataParallel(D.cuda())

g_optim = Adam(G.parameters(),lr=0.0002,betas=(0.5,0.999))
d_optim = Adam(D.parameters(),lr=0.0002,betas=(0.5,0.999))


book_data  = datasets.ImageFolder(root='data/Task2_Split/Task2_Split/train',
                               transform=transforms.Compose([
                                   transforms.Resize((64, 64)),
                                   transforms.ToTensor()
                           ]))

book_dataset =  torch.utils.data.DataLoader(book_data,
                                        batch_size=128,
                                        shuffle=True)


learner = RStandardGanLearner(G,D)

if __name__ == "__main__":
    learner.train(book_dataset,num_classes=6,gen_optimizer=g_optim,disc_optimizer=d_optim,save_outputs_interval=500,model_dir="./genre-gan",latent_size=3*64*64,num_epochs=10,batch_log=False)