import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def set_random_seed(seed=23):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def main():
    set_random_seed()

    device, gpu_ids = util.get_available_devices()

    # Arguments
    opt = args.get_setup_args()

    # Number of channels in the training images
    nc = opt.channels
    # Size of z latent vector (i.e. size of generator input)
    nz = opt.latent_dim
    # Size of feature maps in generator
    ngf = 64

    def eval_fid(gen_images_path, eval_images_path):        
        print("Calculating FID...")
        fid = fid_score.calculate_fid_given_paths((gen_images_path, eval_images_path), opt.batch_size, device)
        return fid

    def evaluate(source_images_path, keep_images=True):
        dataset = datasets.ImageFolder(root=source_images_path,
                                       transform=transforms.Compose([
                                                 transforms.Resize((opt.img_size, opt.img_size)),
                                                 transforms.ToTensor()
                            ]))
        
        dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            num_workers=opt.num_workers)
        
        output_gen_images_path = os.path.join(opt.output_path, opt.version, opt.eval_mode)
        os.makedirs(output_gen_images_path, exist_ok=True)

        output_source_images_path = source_images_path + "_" + str(opt.img_size)

        source_images_available = True

        if (not os.path.exists(output_source_images_path)):
            os.makedirs(output_source_images_path)
            source_images_available = False

        images_done = 0
        for _, data in enumerate(dataloader, 0):
            images, _ = data
            batch_size = images.size(0)
            noise = torch.randn((batch_size, nz, 1, 1)).to(device)

            gen_images = netG(noise)
            for i in range(images_done, images_done + batch_size):
                vutils.save_image(gen_images[i - images_done, :, :, :], "{}/{}.jpg".format(output_gen_images_path, i), normalize=True)       
                if (not source_images_available):
                    vutils.save_image(images[i - images_done, :, :, :], "{}/{}.jpg".format(output_source_images_path, i), normalize=True)     
            images_done += batch_size
        
        fid = eval_fid(output_gen_images_path, output_source_images_path)
        if (not keep_images):
            print("Deleting images generated for validation...")
            rmtree(output_gen_images_path)
        return fid

    test_images_path = os.path.join(opt.data_path, "test")
    val_images_path = os.path.join(opt.data_path, "val")
    model_path = os.path.join(opt.output_path, opt.version, opt.model_file)

    netG = dcgan.Generator(nc, nz, ngf).to(device)

    if (opt.model_file.endswith(".pt")):
        netG.load_state_dict(torch.load(model_path))
    elif (opt.model_file.endswith(".tar")):
        checkpoint = torch.load(model_path)
        netG.load_state_dict(checkpoint['g_state_dict'])

    netG.eval()

    if opt.eval_mode == "val":
        source_images_path = val_images_path
    else:
        source_images_path = test_images_path

    if opt.eval_mode == "val" or opt.eval_mode == "test":
        print("Evaluating model...")
        fid = evaluate(source_images_path)
        print("FID: {}".format(fid))

if __name__ == '__main__':
    main()