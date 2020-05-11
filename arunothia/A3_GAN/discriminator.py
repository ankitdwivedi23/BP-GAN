from setup import *

def discriminator(img_resize=28, num_channels=1):
    """
    Build and return a PyTorch model implementing the architecture above.
    """
    model = nn.Sequential(
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            nn.Linear(img_resize*img_resize*num_channels, 256, bias=True),
            nn.LeakyReLU(0.01, True),
            nn.Linear(256, 256, bias=True),
            nn.LeakyReLU(0.01, True),
            nn.Linear(256, 1, bias=True)
            
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    )
    return model



def build_dc_classifier(img_resize=28, num_channels=1):
    """
    Build and return a PyTorch model for the DCGAN discriminator implementing
    the architecture above.
    """
    return nn.Sequential(
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        Unflatten(C=num_channels, H=img_resize, W=img_resize),
        nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=(5,5), stride=1),
        nn.LeakyReLU(0.01, True),
        nn.MaxPool2d(kernel_size=(2,2), stride=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=1),
        nn.LeakyReLU(0.01, True),
        nn.MaxPool2d(kernel_size=(2,2), stride=2),
        Flatten(),
        nn.Linear(in_features=1024, out_features=1024, bias=True),
        nn.LeakyReLU(0.01, True),
        nn.Linear(in_features=1024, out_features=1, bias=True)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    )

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """

    logits_stack = torch.stack([logits_real, logits_fake], dim=0)
    target_stack = torch.stack([torch.ones_like(logits_real), 
                               torch.zeros_like(logits_fake)], dim=0)
    loss = 2 * bce_loss(logits_stack, target_stack)
    return loss



def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    real_term = (scores_real - 1) * (scores_real - 1)
    fake_term = scores_fake * scores_fake 
    
    loss = 0.5 * (real_term.mean() + fake_term.mean())

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss