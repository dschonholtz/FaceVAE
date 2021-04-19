# allow us to eaily read/write images
import imageio
# array manipulation for image manipulation
import numpy as np
# Give us vision transforms for images
import torchvision.transforms as transforms
# plot any data we might need to
import matplotlib.pyplot as plt
# save outputs
from torchvision.utils import save_image

# transform images to something compatible with the pill lib
to_pil_image = transforms.ToPILImage()


def transform():
    """
    Convert the list of default pytorch tranforms into
    tensors, chained by the compose command
    """
    tensor_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    return tensor_transform


def image_to_vid(images):
    """List of images to animaged gif"""
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave('../outputs/generated_images.gif', imgs)


def save_reconstructed_images(recon_images, epoch):
    """Save generated images in output"""
    save_image(recon_images.cpu(), f"../outputs/output{epoch}.jpg")


def save_loss_plot(train_loss, valid_loss):
    """
    Save loss plot of training and validation
    This is nice, because it will be easy to push the saved image
    to the UI.
    """
    # width and height of the figure in inches: ಠ_ಠ
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../outputs/loss.jpg')
    plt.show()
