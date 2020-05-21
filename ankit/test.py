import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets 
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from shutil import rmtree

import args
import util
from models import dcgan
from eval import fid_score

def main():
    device, gpu_ids = util.get_available_devices()

    # Number of channels in the training images
    nc = 3
    # Size of z latent vector (i.e. size of generator input)
    nz = 100
    # Size of feature maps in generator
    ngf = 64

    # Arguments
    opt = args.get_setup_args()
    

    def eval_fid(fake_images):
        output_images_path = os.path.join(opt.output_path, opt.version, "test")
        os.makedirs(output_images_path, exist_ok=True)
        print("Saving images generated for testing...")
        for i in range(fake_images.size(0)):
            vutils.save_image(fake_images[i, :, :, :], "{}/{}.jpg".format(output_images_path, i))    
        print("Calculating FID...")
        fid = fid_score.calculate_fid_given_paths((output_images_path, test_images_path), opt.batch_size, device)
        return fid

    test_images_path = os.path.join(opt.data_path, "test")
    model_path = os.path.join(opt.output_path, opt.version)

    test_set = datasets.ImageFolder(root=test_images_path,
                                transform=transforms.Compose([
                                    transforms.Resize((opt.img_size, opt.img_size)),
                                    transforms.ToTensor()
                            ]))

    netG = dcgan.Generator(nc, nz, ngf)
    netG = netG.to(device)
    netG.load_state_dict(torch.load(os.path.join(model_path, "model.pt")))
    netG.eval()

    noise = torch.randn(len(test_set), nz, 1, 1, device=device)
    # Generate fake image batch with G
    fake = netG(noise)

    fid = eval_fid(fake)

    print("FID: {}".format(fid))

if __name__ == '__main__':
    main()


