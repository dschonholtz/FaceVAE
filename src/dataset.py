# import open cv for image processing
import cv2
# import torch to load out dataset
from torch.utils.data import Dataset


class LFWDataset(Dataset):
    """
    A class to help us manage our data in our ML pipeline
    """
    def __init__(self, data_list, transform):
        """
        Initiallizes the LFWDataset
        :param data_list []: List of np array encoded images
        :param transform: The image transform to apply to the images
        """
        self.data = data_list
        self.transform = transform

    def __len__(self):
        """

        :return: Returns the length of the dataset
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Gets an item in the dataset at a specific index in the dataset
        :param index: Index in the list of data to fetch
        :return:  Transformed image from dataset
        """
        # read an image at a index and store it locally
        image = cv2.imread(self.data[index])
        # Resize it to a uniform size of 64x 64
        image = cv2.resize(image, (64, 64))
        # convert the image from blue green red to red green blue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply the given transform
        image = self.transform(image)
        # return
        return image
