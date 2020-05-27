import os
import argparse
import random
import torch
import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default="data", help="path to root directory of full images dataset")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--random_seed", type=int, default=23, help="random seed for train-test split")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for dataloader")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
    args = parser.parse_args()
    
    dataset = datasets.ImageFolder(root=args.input_root,
                                   transform=transforms.Compose([
                                        transforms.Resize((args.img_size, args.img_size)),
                                        transforms.ToTensor()
                            ]))

    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers)
    

    nimages = 0
    mean = 0.0
    var = 0.0
    for _, batch_target in tqdm(enumerate(dataloader)):
        batch = batch_target[0]
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0) 
        var += batch.var(2).sum(0)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)

    print("Mean: {}".format(mean))
    print("Standard Deviation: {}".format(std))

if __name__ == '__main__':
    main()