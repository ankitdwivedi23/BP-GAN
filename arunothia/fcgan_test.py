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

    device, gpu_ids = util.get_available_devices()

    # Arguments
    opt = args.get_setup_args()

    num_classes = opt.num_classes
    noise_dim = opt.latent_dim + opt.num_classes

    train_images_path = os.path.join(opt.data_path, "train")
    output_train_images_path = train_images_path + "_" + str(opt.img_size)
    output_sample_images_path = os.path.join(opt.output_path, opt.version, "sample_eval")
    output_nn_pixel_images_path = os.path.join(opt.output_path, opt.version, "nn_eval_pixel")
    output_nn_inception_images_path = os.path.join(opt.output_path, opt.version, "nn_eval_inception")

    os.makedirs(output_sample_images_path, exist_ok=True)
    os.makedirs(output_nn_pixel_images_path, exist_ok=True)
    #os.makedirs(output_nn_inception_images_path, exist_ok=True)

    def get_nn_pixels(sample_images, train_images):
        nn = [None]*len(sample_images)
        pdist = torch.nn.PairwiseDistance(p=2)
        N, C, H, W = train_images.shape
        for i in range(len(sample_images)):
            sample_image = sample_images[i].unsqueeze(0)
            sample_image = torch.cat(N*[sample_image])
            distances = pdist(sample_image.view(-1, C*H*W), train_images.view(-1, C*H*W))
            min_index = torch.argmin(distances)
            nn[i] = train_images[min_index]
        
        r = torch.stack(nn, dim=0).squeeze().to(device)
        return r
    
    def get_nn_inception(sample_activations, train_activations, train_images):
        nn = [None]*len(sample_activations)
        pdist = torch.nn.PairwiseDistance(p=2)
        N = train_activations.size(0)
        for i in range(len(sample_activations)):
            sample_act = sample_activations[i].unsqueeze(0)
            sample_act = torch.cat(N*[sample_act])
            distances = pdist(sample_act, train_activations)
            min_index = torch.argmin(distances)
            nn[i] = train_images[min_index]
        
        r = torch.stack(nn, dim=0).squeeze().to(device)
        return r
    
    def get_nearest_neighbour_pixels(sample_images, num_images, train_images, train_labels):
        all_nn = []
        for i in range(num_classes):
            train_imgs = train_images[train_labels[:] == i]
            nearest_n = get_nn_pixels(sample_images[i*num_images:(i+1)*num_images], train_imgs)
            class_nn = torch.stack([sample_images[i*num_images:(i+1)*num_images], nearest_n], dim=0).squeeze().view(-1, 3, opt.img_size, opt.img_size).to(device)
            all_nn.append(class_nn)
        #r = torch.stack(nn, dim=0).squeeze().view(-1, 3, opt.img_size, opt.img_size).to(device)
        #print(r.shape)
        return all_nn
    

    def get_nearest_neighbour_inception(sample_images, num_images, train_images, train_labels):
        print("Getting sample activations...")
        sample_activations = fid_score.get_activations_given_path(output_sample_images_path, opt.batch_size, device)
        sample_activations = torch.from_numpy(sample_activations).type(torch.FloatTensor).to(device)

        print("Getting train activations...")
        train_activations = fid_score.get_activations_given_path(output_train_images_path, opt.batch_size, device)
        train_activations = torch.from_numpy(train_activations).type(torch.FloatTensor).to(device)

        all_nn = []
        for i in range(num_classes):
            train_imgs = train_images[train_labels[:] == i]
            train_act = train_activations[train_labels[:] == i]
            nearest_n = get_nn_inception(sample_activations[i*num_images:(i+1)*num_images], train_act, train_images)
            class_nn = torch.stack([sample_images[i*num_images:(i+1)*num_images], nearest_n], dim=0).squeeze().view(-1, 3, opt.img_size, opt.img_size).to(device)
            all_nn.append(class_nn)
        #r = torch.stack(nn, dim=0).squeeze().view(-1, 3, opt.img_size, opt.img_size).to(device)
        #print(r.shape)
        return all_nn

    def get_onehot_labels(num_images):
        labels = torch.zeros(num_images, 1).to(device)
        for i in range(num_classes - 1):
            temp = torch.ones(num_images, 1).to(device) + i
            labels = torch.cat([labels, temp], 0)
            
        labels_onehot = torch.zeros(num_images * num_classes, num_classes).to(device)
        labels_onehot.scatter_(1, labels.to(torch.long), 1)

        return labels_onehot

    def sample_images(num_images):
        # Sample noise - declared once at the top to maintain consistency of samples
        z = torch.randn((num_classes * num_images, opt.latent_dim)).to(device)

        '''
        labels = torch.zeros((num_classes * num_images,), dtype=torch.long).to(device)

        for i in range(num_classes):
            for j in range(num_images):
                labels[i*num_images + j] = i
        
        labels_onehot = F.one_hot(labels, num_classes)        
        '''

        labels_onehot = get_onehot_labels(num_images)       
        z = torch.cat((z, labels_onehot.to(dtype=torch.float)), 1)        
        sample_imgs = gen(z)
        for i in range(len(sample_imgs)):
            vutils.save_image(sample_imgs[i], "{}/{}.png".format(output_sample_images_path, i), normalize=True)

        train_set = datasets.ImageFolder(root=train_images_path,
                                transform=transforms.Compose([
                                    transforms.Resize((opt.img_size, opt.img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))

        
        '''
        source_images_available = True

        if (not os.path.exists(output_train_images_path)):
            os.makedirs(output_train_images_path)
            source_images_available = False
        
        
        
        
        if (not source_images_available):
            train_loader = torch.utils.data.DataLoader(train_set,
                                            batch_size=1,
                                            num_workers=opt.num_workers)
        else:
            train_loader = torch.utils.data.DataLoader(train_set,
                                            batch_size=opt.batch_size,
                                            num_workers=opt.num_workers)
        '''

        train_loader = torch.utils.data.DataLoader(train_set,
                                            batch_size=opt.batch_size,
                                            num_workers=opt.num_workers)

        train_images = torch.FloatTensor().to(device)
        train_labels = torch.LongTensor().to(device)

        print("Loading train images...")

        for i, data in enumerate(train_loader, 0):
            img, label = data
            img = img.to(device)
            label = label.to(device)
            train_images = torch.cat([train_images, img], 0)
            train_labels = torch.cat([train_labels, label], 0)
            #if (not source_images_available):
            #    vutils.save_image(img, "{}/{}.jpg".format(output_train_images_path, i), normalize=True)

        print("Estimating nearest neighbors in pixel space, this takes a few minutes...")
        nearest_neighbour_imgs_list = get_nearest_neighbour_pixels(sample_imgs, num_images, train_images, train_labels)
        for label, nn_imgs in enumerate(nearest_neighbour_imgs_list):
            vutils.save_image(nn_imgs.data, "{}/{}.png".format(output_nn_pixel_images_path, label), nrow=num_images, padding=2, normalize=True)
        print("Saved nearest neighbors.")

        '''
        print("Estimating nearest neighbors in feature space, this takes a few minutes...")
        nearest_neighbour_imgs_list = get_nearest_neighbour_inception(sample_imgs, num_images, train_images, train_labels)
        for label, nn_imgs in enumerate(nearest_neighbour_imgs_list):
            vutils.save_image(nn_imgs.data, "{}/{}.png".format(output_nn_inception_images_path, label), nrow=num_images, padding=2, normalize=True)
        print("Saved nearest neighbors.")
        '''

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
            images, labels = data
            batch_size = images.size(0)
            noise = torch.randn((batch_size, opt.latent_dim)).to(device)
            labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            labels_onehot = F.one_hot(labels, num_classes)

            noise = torch.cat((noise, labels_onehot.to(dtype=torch.float)), 1)
            gen_images = gen(noise)
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

    gen = fcgan.Generator(noise_dim).to(device)

    if (opt.model_file.endswith(".pt")):
        gen.load_state_dict(torch.load(model_path))
    elif (opt.model_file.endswith(".tar")):
        checkpoint = torch.load(model_path)
        gen.load_state_dict(checkpoint['g_state_dict'])

    gen.eval()

    if opt.eval_mode == "val":
        source_images_path = val_images_path
    else:
        source_images_path = test_images_path

    if opt.eval_mode == "val" or opt.eval_mode == "test":
        print("Evaluating model...")
        fid = evaluate(source_images_path)
        print("FID: {}".format(fid))
    elif opt.eval_mode == "nn":
        sample_images(opt.num_sample_images)

if __name__ == '__main__':
    main()


