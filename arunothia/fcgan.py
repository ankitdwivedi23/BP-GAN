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
from models import fcgan
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

    # Arguments
    opt = args.get_setup_args()

    #cuda = True if torch.cuda.is_available() else False
    device, gpu_ids = util.get_available_devices()

    num_classes = opt.num_classes
    noise_dim = opt.latent_dim + opt.num_classes

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    train_images_path = os.path.join(opt.data_path, "train")
    val_images_path = os.path.join(opt.data_path, "val")
    output_model_path = os.path.join(opt.output_path, opt.version)
    output_train_images_path = os.path.join(opt.output_path, opt.version, "train")
    output_sample_images_path = os.path.join(opt.output_path, opt.version, "sample")

    os.makedirs(output_train_images_path, exist_ok=True)
    os.makedirs(output_sample_images_path, exist_ok=True)

    train_set = datasets.ImageFolder(root=train_images_path,
                                transform=transforms.Compose([
                                    transforms.Resize((opt.img_size, opt.img_size)),
                                    transforms.ToTensor()
                            ]))

    dataloader = torch.utils.data.DataLoader(train_set,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            num_workers=opt.num_workers)

    gen = fcgan.Generator(noise_dim).to(device)
    disc = fcgan.Discriminator(num_classes).to(device)

    gen.apply(weights_init)
    disc.apply(weights_init)

    optimG = optim.Adam(gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimD = optim.Adam(disc.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    adversarial_loss = torch.nn.BCELoss()
    auxiliary_loss = torch.nn.CrossEntropyLoss()

    real_label_val = 1
    real_label_smooth_val = 0.9
    fake_label_val = 0
    c_fake_label = opt.num_classes

    # Probability of adding label noise during discriminator training
    label_noise_prob = 0.05

    # Keep track of losses, accuracy, FID
    G_losses = []
    D_losses = []
    D_acc = []
    FIDs = []
    val_epochs = []

    def print_labels():
        for class_name in train_set.classes:
            print("{} -> {}".format(class_name, train_set.class_to_idx[class_name]))



    def eval_fid(gen_images_path, eval_images_path):        
        print("Calculating FID...")
        fid = fid_score.calculate_fid_given_paths((gen_images_path, eval_images_path), opt.batch_size, device)
        return fid

    
    def validate(keep_images=True):
        val_set = datasets.ImageFolder(root=val_images_path,
                                       transform=transforms.Compose([
                                                 transforms.Resize((opt.img_size, opt.img_size)),
                                                 transforms.ToTensor()
                            ]))
        
        val_loader = torch.utils.data.DataLoader(val_set,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            num_workers=opt.num_workers)
        
        output_images_path = os.path.join(opt.output_path, opt.version, "val")
        os.makedirs(output_images_path, exist_ok=True)

        images_done = 0
        for _, data in enumerate(val_loader, 0):
            batch_size = data[0].size(0)
            noise = torch.randn((batch_size, opt.latent_dim)).to(device)
            labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            labels_onehot = F.one_hot(labels, num_classes)

            noise = torch.cat((noise, labels_onehot.to(dtype=torch.float)), 1)
            gen_images = gen(noise)
            for i in range(images_done, images_done + batch_size):
                vutils.save_image(gen_images[i - images_done, :, :, :], "{}/{}.jpg".format(output_images_path, i))            
            images_done += batch_size
        
        fid = eval_fid(output_images_path, val_images_path)
        if (not keep_images):
            print("Deleting images generated for validation...")
            rmtree(output_images_path)
        return fid


    def sample_images(num_images, batches_done):
        # Sample noise
        z = torch.randn((num_classes * num_images, opt.latent_dim)).to(device)
        # Get labels ranging from 0 to n_classes for n rows
        labels = torch.zeros((num_classes * num_images,), dtype=torch.long).to(device)

        for i in range(num_classes):
            for j in range(num_images):
                labels[i*num_images + j] = i
        
        labels_onehot = F.one_hot(labels, num_classes)
        z = torch.cat((z, labels_onehot.to(dtype=torch.float)), 1)        
        sample_imgs = gen(z)
        vutils.save_image(sample_imgs.data, "{}/{}.png".format(output_sample_images_path, batches_done), nrow=num_images, padding=2, normalize=True)

    
    def save_loss_plot(path):
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(path)

    def save_acc_plot(path):
        plt.figure(figsize=(10,5))
        plt.title("Discriminator Accuracy")
        plt.plot(D_acc)
        plt.xlabel("iterations")
        plt.ylabel("accuracy")
        plt.savefig(path)
    
    def save_fid_plot(FIDs, epochs, path):
        #N = len(FIDs)
        plt.figure(figsize=(10,5))
        plt.title("FID on Validation Set")
        plt.plot(epochs, FIDs)
        plt.xlabel("epochs")
        plt.ylabel("FID")
        #plt.xticks([i * 49 for i in range(1, N+1)])    
        plt.savefig(path)

    
    print("Label to class mapping:")
    print_labels()

    for epoch in range(opt.num_epochs):
        for i, data in enumerate(dataloader, 0):

            images, class_labels = data
            images = images.to(device)
            class_labels = class_labels.to(device)

            batch_size = images.size(0)

            real_label_smooth = torch.full((batch_size,), real_label_smooth_val, device=device)
            real_label = torch.full((batch_size,), real_label_val, device=device)
            fake_label = torch.full((batch_size,), fake_label_val, device=device)

            ############################
            # Train Discriminator
            ###########################
            
            ## Train with all-real batch

            optimD.zero_grad()

            real_pred, real_aux = disc(images)
            
            mask = torch.rand((batch_size,), device=device) <= label_noise_prob
            mask = mask.type(torch.float)            
            noisy_label = torch.mul(1-mask, real_label_smooth) + torch.mul(mask, fake_label)

            d_real_loss = (adversarial_loss(real_pred, noisy_label) + auxiliary_loss(real_aux, class_labels)) / 2

            # Train with fake batch
            noise = torch.randn((batch_size, opt.latent_dim)).to(device)
            gen_class_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            gen_class_labels_onehot = F.one_hot(gen_class_labels, num_classes)

            noise = torch.cat((noise, gen_class_labels_onehot.to(dtype=torch.float)), 1)
            gen_images = gen(noise)
            fake_pred, fake_aux = disc(gen_images.detach())

            mask = torch.rand((batch_size,), device=device) <= label_noise_prob
            mask = mask.type(torch.float)            
            noisy_label = torch.mul(1-mask, fake_label) + torch.mul(mask, real_label_smooth)
            
            c_fake = torch.full(gen_class_labels.shape, c_fake_label, device=device, dtype=torch.long)
            d_fake_loss = (adversarial_loss(fake_pred, noisy_label) + auxiliary_loss(fake_aux, c_fake)) / 2

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            # Calculate discriminator accuracy
            pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            gt = np.concatenate([class_labels.data.cpu().numpy(), gen_class_labels.data.cpu().numpy()], axis=0)
            d_acc = np.mean(np.argmax(pred, axis=1) == gt)

            d_loss.backward()
            optimD.step()

            ############################
            # Train Generator
            ###########################

            optimG.zero_grad()

            validity, aux_scores = disc(gen_images)
            g_loss = 0.5 * (adversarial_loss(validity, real_label) + auxiliary_loss(aux_scores, gen_class_labels))

            g_loss.backward()
            optimG.step()

            # Save losses and accuracy for plotting
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
            D_acc.append(d_acc)

            # Output training stats
            if i % opt.print_every == 0:
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f, acc:  %d%%] [G loss: %.4f]"
                % (epoch, opt.num_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
                )

            batches_done = epoch * len(dataloader) + i
            
            # Generate and save sample images
            if (batches_done % opt.sample_interval == 0) or ((epoch == opt.num_epochs-1) and (i == len(dataloader)-1)):
                # Put G in eval mode
                gen.eval()
                
                with torch.no_grad():                
                    sample_images(opt.num_sample_images, batches_done)
                vutils.save_image(gen_images.data[:36], "{}/{}.png".format(output_train_images_path, batches_done), nrow=6, padding=2, normalize=True)
                
                # Put G back in train mode
                gen.train()
            
        # Save model checkpoint
        if (epoch !=0 and epoch % opt.checkpoint_epochs == 0):
            print("Checkpoint at epoch {}".format(epoch))
            print("Saving generator model...")
            torch.save(gen.state_dict(), os.path.join(output_model_path, "model_checkpoint_{}.pt".format(epoch)))
            print("Saving G & D loss plot...")
            save_loss_plot(os.path.join(opt.output_path, opt.version, "loss_plot_{}.png".format(epoch)))
            print("Saving D accuracy plot...")
            save_acc_plot(os.path.join(opt.output_path, opt.version, "accuracy_plot_{}.png".format(epoch)))
            print("Validating model...")
            fid = validate(keep_images=False)
            print("Validation FID: {}".format(fid))
            FIDs.append(fid)
            val_epochs.append(epoch)
            print("Saving FID plot...")
            save_fid_plot(FIDs, val_epochs, os.path.join(opt.output_path, opt.version, "fid_plot_{}.png".format(epoch)))

    
    print("Saving final generator model...")
    torch.save(gen.state_dict(), os.path.join(output_model_path, "model.pt"))
    print("Done!")

    print("Saving final G & D loss plot...")
    save_loss_plot(os.path.join(opt.output_path, opt.version, "loss_plot.png"))
    print("Done!")

    print("Saving final D accuracy plot...")
    save_acc_plot(os.path.join(opt.output_path, opt.version, "accuracy_plot.png"))
    print("Done!")

    print("Validating final model...")
    fid = validate()
    FIDs.append(fid)
    val_epochs.append(epoch)
    print("Saving final FID plot...")
    save_fid_plot(FIDs, val_epochs, os.path.join(opt.output_path, opt.version, "fid_plot"))

if __name__ == '__main__':
    main()