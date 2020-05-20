import argparse

def get_setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, help="model name or version")
    parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for dataloader")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--eval_interval", type=int, default=400, help="number of batches between evaluation")
    parser.add_argument("--print_every", type=int, default=50, help="number of iterations between printing training stats")
    parser.add_argument("--data_path", type=str, default="data", help="path to root data directory")
    parser.add_argument("--output_path", type=str, default="results", help="path to directory for storing model output")
    return parser.parse_args()