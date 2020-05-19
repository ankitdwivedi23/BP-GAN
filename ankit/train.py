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

    #cuda = True if torch.cuda.is_available() else False
    device, gpu_ids = util.get_available_devices()

    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def eval_fid(fake_images, epoch):
        output_path = os.path.join(output_val_images_path, str(epoch))
        os.makedirs(output_path, exist_ok=True)
        print("Saving images generated for validation...")
        for i in range(fake_images.size(0)):
            vutils.save_image(fake_images[i, :, :, :], "{}/{}.jpg".format(output_path, i))    
        fid = fid_score.calculate_fid_given_paths((output_path, val_images_path), opt.batch_size, device)
        rmtree(output_path)
        return fid

    # Number of channels in the training images
    nc = 3
    # Size of z latent vector (i.e. size of generator input)
    nz = 100
    # Size of feature maps in generator
    ngf = 64
    # Size of feature maps in discriminator
    ndf = 64

    # Arguments
    opt = args.get_setup_args()

    train_images_path = os.path.join(opt.data_path, "train")
    val_images_path = os.path.join(opt.data_path, "val")
    output_model_path = os.path.join(opt.output_path, opt.version)
    output_train_images_path = os.path.join(opt.output_path, opt.version, "train")
    output_grid_images_path = os.path.join(opt.output_path, opt.version, "grid")
    output_val_images_path = os.path.join(opt.output_path, opt.version, "val")

    os.makedirs(output_train_images_path, exist_ok=True)
    os.makedirs(output_grid_images_path, exist_ok=True)
    os.makedirs(output_val_images_path, exist_ok=True)

    train_set = datasets.ImageFolder(root=train_images_path,
                                transform=transforms.Compose([
                                    transforms.Resize((opt.img_size, opt.img_size)),
                                    transforms.ToTensor()
                            ]))

    val_set =  datasets.ImageFolder(root=val_images_path,
                                transform=transforms.Compose([
                                    transforms.Resize((opt.img_size, opt.img_size)),
                                    transforms.ToTensor()
                            ]))

    val_size = len(val_set)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Initialize generator and discriminator
    netG = dcgan.Generator(nc, nz, ngf).to(device)
    netD = dcgan.Discriminator(nc, ndf).to(device)

    # Create batch of latent vectors to visualize
    # the progress of the generator
    grid_noise = torch.randn(64, nz, 1, 1, device=device)
    val_noise = torch.randn(val_size, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


    dataloader = torch.utils.data.DataLoader(train_set,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            num_workers=opt.num_workers)

    # ----------
    #  Training
    # ----------

    G_losses = []
    D_losses = []
    FIDs = []
    best_fid = float("inf")
    iters = 0

    def save_loss_plot(path):
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(path)

    def save_fid_plot(path):
        N = len(FIDs)
        plt.figure(figsize=(10,5))
        plt.title("FID on Validation Set")
        plt.plot(FIDs)
        plt.xlabel("epochs")
        plt.ylabel("FID")
        plt.xlim(0, opt.num_epochs)
        plt.savefig(path)


    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(opt.num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()        
            # Format batch
            real_imgs = data[0].to(device)
            batch_size = real_imgs.size(0)
            label = torch.full((batch_size,), real_label, device=device)
            
            # Forward pass real batch through D
            output = netD(real_imgs).view(-1)
            
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Save Losses for plotting
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Output training stats
            if i % opt.print_every == 0:
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f] [D(x): %.4f] [D(G(z)): %.4f / %.4f]"
                % (epoch, opt.num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            batches_done = epoch * len(dataloader) + i
            
            # Put G in eval mode
            netG.eval()
            
            if (batches_done % opt.eval_interval == 0) or ((epoch == opt.num_epochs-1) and (i == len(dataloader)-1)):
                print("Evaluating at [Epoch %d/%d] [Batch %d/%d]" % (epoch, opt.num_epochs, i, len(dataloader)))
                with torch.no_grad():                
                    fake_grid = netG(grid_noise).detach().cpu()
                    fake_val = netG(val_noise).detach().cpu()
                vutils.save_image(fake_grid.data[:64], "{}/{}.png".format(output_grid_images_path, batches_done), nrow=5, padding=2, normalize=True)
                vutils.save_image(fake.data[:64], "{}/{}.png".format(output_train_images_path, batches_done), nrow=5, padding=2, normalize=True)
                fid = eval_fid(fake_val, epoch)
                print("[Val FID: %.4f]" % (fid))
                FIDs.append(fid)
                if fid < best_fid:
                    print("NEW Best Model found!")
                    best_fid = fid
                    torch.save(netG.state_dict(), os.path.join(output_model_path, "model.pt"))               
            
            # Put G back in train mode
            netG.train()

    print("Saving FID plot...")
    save_fid_plot(os.path.join(opt.output_path, opt.version, "fid_plot.png"))
    print("Done!")

    print("Saving plot showing generator and discriminator loss during training...")
    save_loss_plot(os.path.join(opt.output_path, opt.version, "loss_plot.png"))
    print("Done!")

if __name__ == '__main__':
    main()