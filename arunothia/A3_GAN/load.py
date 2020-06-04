from setup import *

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

def load_data(data_type, NUM_TRAIN=50000, batch_size=128, img_resize=28, num_channels=1):

    NUM_VAL = int(NUM_TRAIN/10)

    if data_type == "MNIST":
    
        mnist_train = dset.MNIST('../data/MNIST_data', train=True, download=True,
                                transform=T.ToTensor())
        loader_train = DataLoader(mnist_train, batch_size=batch_size,
                                sampler=ChunkSampler(NUM_TRAIN, 0))

        mnist_val = dset.MNIST('../data/MNIST_data', train=True, download=True,
                                transform=T.ToTensor())
        loader_val = DataLoader(mnist_val, batch_size=batch_size,
                                sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

        imgs = loader_train.__iter__().next()[0].view(batch_size, 784).numpy().squeeze()
    
    elif data_type == "BOOK":

        book_data = datasets.ImageFolder(root='../data/book-dataset/',
                                            transform=transforms.Compose([
                                            #transforms.Grayscale(num_output_channels=num_channels),
                                            transforms.Resize((img_resize, img_resize)),
                                            transforms.ToTensor()
                                    ]))

        loader_train = DataLoader(book_data, batch_size=batch_size,
                                sampler=ChunkSampler(NUM_TRAIN, 0))

        loader_val = DataLoader(book_data, batch_size=batch_size,
                                sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

        imgs = loader_train.__iter__().next()[0].view(batch_size*num_channels, img_resize*img_resize).numpy().squeeze()   

    else:
        print("NO DATASET CHOSEN")
        return None, None

    show_images(imgs, num_channels).savefig('../results/input-data')
    return loader_train, loader_val

def show_data_split():

    book_data = datasets.ImageFolder(root='../data/Task2_split/Task2_Split/train',
                                            transform=transforms.Compose([
                                            transforms.ToTensor()
                                    ]))
    
    dataloader = DataLoader(book_data, batch_size=500, shuffle=True, num_workers=4)
    class_names = book_data.classes
    
    inputs, classes = next(iter(dataloader))

    class_limit = [0, 0, 0, 0, 0 ,0]
    inp = [[], [], [], [], [], []]
    cl = []

    for e,x in enumerate(inputs):
        if class_limit[classes[e]] >= 6:
            continue
        class_limit[classes[e]] += 1
        inp[classes[e]].append(x)
        cl.append(classes[e])
    
    flat_input = [None]*36

    for i in range(6):
        for j in range(6):
            flat_input[j*6+i] = inp[i][j]
    
    out = torchvision.utils.make_grid(flat_input, nrow=6)
    #imshow(out, title=[class_names[x] for x in range(6)]).savefig('../results/data_split_')
    imshow(out).savefig('../results/data_split')
