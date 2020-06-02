import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch import autograd
from torchvision import datasets 
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from shutil import rmtree

import args
import util
from models import acgan
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

    # WGAN hyperparams

    # lower and upper clip value for disc. weights
    clip_value = 0.01
    # number of training steps for discriminator per iter
    n_critic = 5
    # Gradient penalty lambda hyperparameter
    lambda_gp = 10

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
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))

    dataloader = torch.utils.data.DataLoader(train_set,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            num_workers=opt.num_workers)

    gen = acgan.Generator(noise_dim).to(device)
    disc = acgan.Discriminator(num_classes).to(device)

    gen.apply(weights_init)
    disc.apply(weights_init)

    #optimG = optim.Adam(gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    #optimD = optim.Adam(disc.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    optimG = optim.RMSprop(gen.parameters(), lr=opt.lr)
    optimD = optim.RMSprop(disc.parameters(), lr=opt.lr)

    #adversarial_loss = torch.nn.BCELoss()
    auxiliary_loss = torch.nn.CrossEntropyLoss()

    real_label_val = 1
    real_label_low = 0.75
    real_label_high = 1.0
    fake_label_val = 0

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

        output_source_images_path = val_images_path + "_" + str(opt.img_size)

        source_images_available = True

        if (not os.path.exists(output_source_images_path)):
            os.makedirs(output_source_images_path)
            source_images_available = False

        images_done = 0
        for _, data in enumerate(val_loader, 0):
            images, labels = data
            batch_size = images.size(0)
            noise = torch.randn((batch_size, opt.latent_dim)).to(device)
            labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            labels_onehot = F.one_hot(labels, num_classes)

            noise = torch.cat((noise, labels_onehot.to(dtype=torch.float)), 1)
            gen_images = gen(noise)
            for i in range(images_done, images_done + batch_size):
                vutils.save_image(gen_images[i - images_done, :, :, :], "{}/{}.jpg".format(output_images_path, i), normalize=True)       
                if (not source_images_available):
                    vutils.save_image(images[i - images_done, :, :, :], "{}/{}.jpg".format(output_source_images_path, i), normalize=True)     
            images_done += batch_size
        
        fid = eval_fid(output_images_path, output_source_images_path)
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
        plt.close()

    def save_acc_plot(path):
        plt.figure(figsize=(10,5))
        plt.title("Discriminator Accuracy")
        plt.plot(D_acc)
        plt.xlabel("iterations")
        plt.ylabel("accuracy")
        plt.savefig(path)
        plt.close()
    
    def save_fid_plot(FIDs, epochs, path):
        #N = len(FIDs)
        plt.figure(figsize=(10,5))
        plt.title("FID on Validation Set")
        plt.plot(epochs, FIDs)
        plt.xlabel("epochs")
        plt.ylabel("FID")
        #plt.xticks([i * 49 for i in range(1, N+1)])    
        plt.savefig(path)
        plt.close()

    
    def calc_gradient_penalty(netD, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
        alpha = alpha.view(batch_size, 3, opt.img_size, opt.img_size)
        alpha = alpha.to(device)

        fake_data = fake_data.view(batch_size, 3, opt.img_size, opt.img_size)
        interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

        interpolates = interpolates.to(device)
        interpolates.requires_grad_(True)   

        disc_interpolates, _ = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)                              
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
        return gradient_penalty

    
    print("Label to class mapping:")
    print_labels()

    for epoch in range(1, opt.num_epochs + 1):
        for i, data in enumerate(dataloader, 0):

            images, class_labels = data
            images = images.to(device)
            class_labels = class_labels.to(device)

            batch_size = images.size(0)

            real_label_smooth = (real_label_low - real_label_high) * torch.rand((batch_size,), device=device) + real_label_high
            real_label = torch.full((batch_size,), real_label_val, device=device)
            fake_label = torch.full((batch_size,), fake_label_val, device=device)

            ############################
            # Train Discriminator
            ###########################
            
            ## Train with all-real batch

            optimD.zero_grad()

            real_pred, real_aux = disc(images)
            
            d_real_aux_loss = auxiliary_loss(real_aux, class_labels)

            # Train with fake batch
            noise = torch.randn((batch_size, opt.latent_dim)).to(device)
            gen_class_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            gen_class_labels_onehot = F.one_hot(gen_class_labels, num_classes)

            noise = torch.cat((noise, gen_class_labels_onehot.to(dtype=torch.float)), 1)
            gen_images = gen(noise).detach()
            fake_pred, fake_aux = disc(gen_images)

            #d_fake_aux_loss = auxiliary_loss(fake_aux, gen_class_labels)

            gradient_penalty = calc_gradient_penalty(disc, images, gen_images)

            # Total discriminator loss
            d_aux_loss = d_real_aux_loss
            d_loss = fake_pred.mean() - real_pred.mean() + gradient_penalty + d_aux_loss
            
            # Calculate discriminator accuracy
            pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            gt = np.concatenate([class_labels.data.cpu().numpy(), gen_class_labels.data.cpu().numpy()], axis=0)
            d_acc = np.mean(np.argmax(pred, axis=1) == gt)

            d_loss.backward()
            optimD.step()

            # Clip weights of discriminator
            for p in disc.parameters():
                p.data.clamp_(-clip_value, clip_value)

            if i % n_critic == 0:
                ############################
                # Train Generator
                ###########################

                optimG.zero_grad()

                gen_images = gen(noise)

                gen_pred, aux_scores = disc(gen_images)
                g_aux_loss = auxiliary_loss(aux_scores, gen_class_labels)
                g_loss = g_aux_loss - gen_pred.mean()

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
        if (epoch != opt.num_epochs and epoch % opt.checkpoint_epochs == 0):
            print("Checkpoint at epoch {}".format(epoch))
            print("Saving generator model...")
            torch.save(gen.state_dict(), os.path.join(output_model_path, "model_checkpoint_{}.pt".format(epoch)))
            print("Saving G & D loss plot...")
            save_loss_plot(os.path.join(opt.output_path, opt.version, "loss_plot_{}.png".format(epoch)))
            print("Saving D accuracy plot...")
            save_acc_plot(os.path.join(opt.output_path, opt.version, "accuracy_plot_{}.png".format(epoch)))
            
            print("Validating model...")
            gen.eval()
            with torch.no_grad():
                fid = validate(keep_images=False)
            print("Validation FID: {}".format(fid))
            FIDs.append(fid)
            val_epochs.append(epoch)
            print("Saving FID plot...")
            save_fid_plot(FIDs, val_epochs, os.path.join(opt.output_path, opt.version, "fid_plot_{}.png".format(epoch)))
            gen.train()

    
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
    gen.eval()    
    with torch.no_grad():
        fid = validate()
    print("Final Validation FID: {}".format(fid))
    FIDs.append(fid)
    val_epochs.append(epoch)
    print("Saving final FID plot...")
    save_fid_plot(FIDs, val_epochs, os.path.join(opt.output_path, opt.version, "fid_plot"))

if __name__ == '__main__':
    main()