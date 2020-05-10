import argparse
import os
import random
import torch
import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from PIL import ImageFile
from shutil import copyfile, rmtree
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
Input images should be arranged in this way:

inputroot/label1/xxx.png
inputroot/label1/xxy.png
inputroot/label1/xxz.png

inputroot/label2/xxx.png
inputroot/label2/xxy.png
inputroot/label2/xxz.png

Output images will be saved in this way:

outputroot/train/label1/xxx.png
outputroot/train/label1/xxy.png
outputroot/train/label1/xxz.png

and similarly for test
'''

def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default="data", help="path to root directory of full images dataset")
    parser.add_argument("--output_root", type=str, default="data_split", help="path to root directory for storing train-test split")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--random_seed", type=int, default=23, help="random seed for train-test split")
    args = parser.parse_args()

    output_train = os.path.join(args.output_root, "train/")
    output_test = os.path.join(args.output_root, "test/")

    os.makedirs(output_train, exist_ok=True)
    os.makedirs(output_test, exist_ok=True)

    for label in tqdm(os.listdir(args.input_root)):        
        label_folder = os.path.join(args.input_root, label)

        # Create dummy folder to enable loading images using ImageFolder
        dummy_folder = os.path.join(label_folder, "dummy")
        
        os.makedirs(dummy_folder, exist_ok=True)
        os.makedirs(os.path.join(output_train, label), exist_ok=True)
        os.makedirs(os.path.join(output_test, label), exist_ok=True)

        for f in os.listdir(label_folder):
            src = os.path.join(label_folder,f)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(dummy_folder, f)
            if not os.path.exists(dst):
                copyfile(src, dst)

        dataset = datasets.ImageFolder(root=label_folder,
                                        transform=transforms.Compose([
                                        transforms.Resize((args.img_size, args.img_size)),
                                        transforms.ToTensor()
                                ]))

        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=len(dataset),
                                                shuffle=True,
                                                num_workers=1)


        for i, (imgs, _) in enumerate(dataloader):
            X = imgs.numpy()

            X_train, X_test = \
                train_test_split(X,
                                test_size=0.2,
                                random_state=args.random_seed,
                                shuffle=True)
            
            imgs_train = torch.from_numpy(X_train)
            imgs_test = torch.from_numpy(X_test)
            
            for i in range(imgs_train.size(0)):
                output_path = os.path.join(output_train, label)
                os.makedirs(output_path, exist_ok=True)
                vutils.save_image(imgs_train[i, :, :, :], "{}/{}.jpg".format(output_path, i))
            
            for i in range(imgs_test.size(0)):
                output_path = os.path.join(output_test, label)
                os.makedirs(output_path, exist_ok=True)
                vutils.save_image(imgs_test[i, :, :, :], "{}/{}.jpg".format(output_path, i))
        
        # Delete dummy folder
        rmtree(dummy_folder)

if __name__ == '__main__':
    main()    
    


        





