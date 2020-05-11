from setup import *
from discriminator import *
from generator import *
from train import *
from load import *

# Parser to choose dataset and other config values

parser = argparse.ArgumentParser(description='Load the desired dataset')
parser.add_argument('--NUM_TRAIN', type=int, default=50000,
                    help='num_train value, default value is 50000')

parser.add_argument('--BATCH_SIZE', type=int, default=128,
                    help='batch size, default value is 128')

parser.add_argument('--DATA_TYPE', default='MNIST',
                    help='data type, default value is MNIST')

parser.add_argument('--NOISE_DIM', type=int, default=96,
                    help='Noise Dimension, default value is 96')

parser.add_argument('--SHOW_EVERY', type=int, default=250,
                    help='Show every, default value is 250')

parser.add_argument('--NUM_EPOCHS', type=int, default=10,
                    help='Number of Epochs, default value is 10')

parser.add_argument('--IMG_RESIZE', type=int, default=28,
                    help='Size to which image inputs are reshaped, default value is 28')

parser.add_argument('--NUM_CHANNELS', type=int, default=1,
                    help='Number of channels in the input images, default value is 1')

args = parser.parse_args()

loader_train, loader_val = load_data(args.DATA_TYPE, args.NUM_TRAIN, args.BATCH_SIZE, args.IMG_RESIZE, 
                                     args.NUM_CHANNELS)

# Make the discriminator
D = discriminator(args.IMG_RESIZE, args.NUM_CHANNELS).type(dtype)

# Make the generator
G = generator(args.NOISE_DIM, args.IMG_RESIZE, args.NUM_CHANNELS).type(dtype)

# Use the function you wrote earlier to get optimizers for the Discriminator and the Generator
D_solver = get_optimizer(D)
G_solver = get_optimizer(G)

# Run it!
'''
run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, loader_train, loader_val,
                args.SHOW_EVERY, args.BATCH_SIZE, args.NOISE_DIM, args.NUM_EPOCHS, args.IMG_RESIZE, args.NUM_CHANNELS)
'''

run_a_gan(D, G, D_solver, G_solver, ls_discriminator_loss, ls_generator_loss, loader_train, loader_val,
                args.SHOW_EVERY, args.BATCH_SIZE, args.NOISE_DIM, args.NUM_EPOCHS, args.IMG_RESIZE, args.NUM_CHANNELS)

''''

D_DC = build_dc_classifier(args.IMG_RESIZE, args.NUM_CHANNELS).type(dtype) 
D_DC.apply(initialize_weights)
G_DC = build_dc_generator(args.NOISE_DIM).type(dtype)
G_DC.apply(initialize_weights)

D_DC_solver = get_optimizer(D_DC)
G_DC_solver = get_optimizer(G_DC)

run_a_gan(D_DC, G_DC, D_DC_solver, G_DC_solver, ls_discriminator_loss, ls_generator_loss, loader_train, loader_val,
                args.SHOW_EVERY, args.BATCH_SIZE, args.NOISE_DIM, args.NUM_EPOCHS, args.IMG_RESIZE, args.NUM_CHANNELS)

'''