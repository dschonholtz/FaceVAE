import torch
import torch.optim as optim
import torch.nn as nn
import model
import matplotlib
from torch.utils.data import DataLoader
# make a grid of images
from torchvision.utils import make_grid
from engine import train, validate
from utils import save_reconstructed_images, image_to_vid, save_loss_plot
from utils import transform
from prepare_data import prepare_dataset
from dataset import LFWDataset

# uses ggplot instead of default plotter
matplotlib.style.use('ggplot')

# define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# a list to save all the reconstructed images in PyTorch grid format
grid_images = []

# initialize the model
model = model.ConvVAE().to(device)
# define the learning parameters
lr = 0.0001
epochs = 200
batch_size = 64
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')

# initialize the transform
transform = transform()
# prepare the training and validation data loaders
train_data, valid_data = prepare_dataset(
    root_path='../input/catsNdogs/'
)
trainset = LFWDataset(train_data, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
validset = LFWDataset(valid_data, transform=transform)
validloader = DataLoader(validset, batch_size=batch_size)

train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, trainloader, trainset, device, optimizer, criterion
    )
    valid_epoch_loss, recon_images = validate(
        model, validloader, validset, device, criterion
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    # save the reconstructed images from the validation loop
    save_reconstructed_images(recon_images, epoch+1)
    # convert the reconstructed images to PyTorch image grid format
    image_grid = make_grid(recon_images.detach().cpu())
    grid_images.append(image_grid)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")

# save the reconstructions as a .gif file
image_to_vid(grid_images)
# save the loss plots to disk
save_loss_plot(train_loss, valid_loss)
print('TRAINING COMPLETE')
