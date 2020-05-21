import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch

import cgan
import util
import args
from eval import fid_score

def main():
    cuda = True if torch.cuda.is_available() else False
    device, _ = util.get_available_devices()
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # Arguments
    opt = args.get_setup_args()    

    def eval_fid(fake_images):
        output_images_path = os.path.join(opt.output_path, opt.version, "test")
        os.makedirs(output_images_path, exist_ok=True)
        print("Saving images generated for testing...")
        for i in range(fake_images.size(0)):
            save_image(fake_images[i, :, :, :], "{}/{}.jpg".format(output_images_path, i))    
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

    netG = cgan.Generator()
    netG.cuda()
    netG.load_state_dict(torch.load(os.path.join(model_path, "model.pt")))
    netG.eval()

    noise = FloatTensor(np.random.normal(0, 1, (len(test_set), opt.latent_dim)))
    gen_labels = LongTensor(np.random.randint(0, opt.num_classes, len(test_set)))
    
    # Generate fake image batch with G
    gen_imgs = netG(noise, gen_labels)

    fid = eval_fid(gen_imgs)

    print("FID: {}".format(fid))

if __name__ == '__main__':
    main()
