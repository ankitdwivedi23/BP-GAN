from setup import *

def generator(noise_dim, img_resize=28, num_channels=1):
    """
    Build and return a PyTorch model implementing the architecture above.
    """
    model = nn.Sequential(
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            nn.Linear(noise_dim, 1024, bias=True),
            nn.ReLU(True),
            nn.Linear(1024, 1024, bias=True),
            nn.ReLU(True),
            nn.Linear(1024, img_resize*img_resize*num_channels, bias=True),
            nn.Tanh()

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    )
    return model

def build_dc_generator(noise_dim):
    """
    Build and return a PyTorch model implementing the DCGAN generator using
    the architecture described above.
    """
    return nn.Sequential(
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        nn.Linear(in_features=noise_dim, out_features=1024, bias=True),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=1024),
        nn.Linear(in_features=1024, out_features=6272, bias=True),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=6272),
        Unflatten(C=128, H=7, W=7),
        nn.ConvTranspose2d(in_channels=128, out_channels=64, 
                            kernel_size=(4,4), stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(num_features=64),
        nn.ConvTranspose2d(in_channels=64, out_channels=1, 
                            kernel_size=(4,4), stride=2, padding=1),
        nn.Tanh(),
        Flatten()

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    )



def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = bce_loss(logits_fake, torch.ones_like(logits_fake))
    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = 0.5 * (((scores_fake-1)*(scores_fake-1)).mean())
    return loss