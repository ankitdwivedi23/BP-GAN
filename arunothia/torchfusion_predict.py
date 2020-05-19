from torchfusion.gan.learners import *
from torchfusion.gan.applications import StandardGenerator
import torch.cuda as cuda
import torch.nn as nn
from torchvision.utils import save_image
import torch
from torch.distributions import Normal
import argparse

parser = argparse.ArgumentParser(description='Load the desired dataset')

parser.add_argument('--LATENT_SIZE', type=int, default=3*64*64,
                    help='Latent size, default value is 3*64*64')

parser.add_argument('--LABEL_TO_GENERATE', type=int, default=1,
                    help='Label to generate, default value is 1')

args = parser.parse_args()

G = StandardGenerator(output_size=(3,64,64),latent_size=args.LATENT_SIZE,num_classes=6)

if cuda.is_available():
    G = nn.DataParallel(G.cuda())

learner = RStandardGanLearner(G,None)
learner.load_generator("results/genre-gan/gen_models/gen_model_10.pth")

if __name__ == "__main__":
    "Define an instance of the normal distribution"
    dist = Normal(0,1)

    #Get a sample latent vector from the distribution
    latent_vector = dist.sample((1,args.LATENT_SIZE))

    #Define the class of the image you want to generate
    label = torch.LongTensor(1).fill_(args.LABEL_TO_GENERATE)

    #Run inference
    image = learner.predict([latent_vector,label])

    #Save generated image
    save_image(image, "results/image{}.jpg".format(args.LABEL_TO_GENERATE))