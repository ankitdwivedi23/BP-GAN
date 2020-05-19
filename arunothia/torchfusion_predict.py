from torchfusion.gan.learners import *
from torchfusion.gan.applications import StandardGenerator
import torch.cuda as cuda
import torch.nn as nn
from torchvision.utils import save_image
import torch
from torch.distributions import Normal


G = StandardGenerator(output_size=(3,64,64),latent_size=3*64*64,num_classes=6)

if cuda.is_available():
    G = nn.DataParallel(G.cuda())

learner = RStandardGanLearner(G,None)
learner.load_generator("./genre-gan/gen_models/gen_model_10.pth")

if __name__ == "__main__":
    "Define an instance of the normal distribution"
    dist = Normal(0,1)

    #Get a sample latent vector from the distribution
    latent_vector = dist.sample((1,3*64*64))

    #Define the class of the image you want to generate
    label = torch.LongTensor(1).fill_(5)

    #Run inference
    image = learner.predict([latent_vector,label])

    #Save generated image
    save_image(image, "image.jpg")