"""
This module is responsible for the definition of the VAE model
"""

# pytorch
import torch
# neural nets
import torch.nn as nn
# short hand for convolutional functions
import torch.nn.functional as F
# initial number of filters
init_channels = 64
# color channels
image_channels = 3
# number of features to consider
latent_dim = 100


class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()

        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels,
            kernel_size=4, stride=2, padding=2
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2,
            kernel_size=4, stride=2, padding=2
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4,
            kernel_size=4, stride=2, padding=2
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=init_channels*8,
            kernel_size=4, stride=2, padding=2
        )
        self.enc5 = nn.Conv2d(
            in_channels=init_channels*8, out_channels=1024,
            kernel_size=4, stride=2, padding=2
        )
        self.fc1 = nn.Linear(1024, 2048)
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_log_var = nn.Linear(2048, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 1024)
        # decoder 
        self.dec1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=init_channels*8,
            kernel_size=3, stride=2
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4,
            kernel_size=3, stride=2
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2,
            kernel_size=3, stride=2
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=init_channels,
            kernel_size=3, stride=2
        )
        self.dec5 = nn.ConvTranspose2d(
            in_channels=init_channels, out_channels=image_channels,
            kernel_size=4, stride=2
        )

    def reparameterize(self, mu, log_var):
        """
        Generate a sample by taking a sample from the distribution of
        possibilities as defined by the given mu and variance
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample

    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        # dataset is broken up into batches. Get batch size
        batch, _, _, _ = x.shape
        # apply pooling to reduce the kernel size to 1 effectively
        # flattenting our images. Reshape those images into an output
        # for each image in the batch
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        # view z in a different format. (no data copy)
        z = z.view(-1, 1024, 1, 1)

        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        reconstruction = torch.sigmoid(self.dec5(x))
        return reconstruction, mu, log_var
