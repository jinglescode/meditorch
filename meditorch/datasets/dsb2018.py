import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
from torch.utils.data import Dataset


class DataScienceBowl2018Nuclei(Dataset):
    
    """`DataScienceBowl2018Nuclei <https://www.kaggle.com/c/data-science-bowl-2018/>`_ Dataset.
        
    This dataset is taken from Data Science Bowl <https://datasciencebowl.com/>, for the challenge, Spot Nuclei. Speed Cures <https://datasciencebowl.com/spot-nuclei-speed-cures/>. 
    Download the dataset from Kaggle <https://www.kaggle.com/c/data-science-bowl-2018/>.
    This dataset contains a large number of segmented nuclei images. The images were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (brightfield vs. fluorescence). The dataset is designed to challenge an algorithm's ability to generalize across these variations.
    """

    def __init__(self, file_path, img_width=128, img_height=128, img_channels=3, ignoreMask=False):

        self.data = None
        self.targets = None
        
        # Get train and test IDs
        image_ids = next(os.walk(file_path))[1]

        # Get and resize train images and masks
        self.data = np.zeros((len(image_ids), img_channels, img_height, img_width), dtype=np.uint8)
        self.targets = np.zeros((len(image_ids), 1, img_height, img_width), dtype=np.bool)
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(image_ids), total=len(image_ids)):
            path = file_path + id_
            img = imread(path + '/images/' + id_ + '.png')[:,:,:img_channels]
            img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
            img = np.rollaxis(img, 2, 0)
            img = img.astype(np.float32)
            self.data[n] = img
            
            if not ignoreMask:
                mask = np.zeros((img_height, img_width, 1), dtype=np.bool)
                for mask_file in next(os.walk(path + '/masks/'))[2]:
                    mask_ = imread(path + '/masks/' + mask_file)
                    mask_ = np.expand_dims(resize(mask_, (img_height, img_width), mode='constant', 
                                                  preserve_range=True), axis=-1)
                    mask = np.maximum(mask, mask_)
                mask = np.rollaxis(mask, 2, 0)
                mask = mask.astype(np.float32)
                self.targets[n] = mask

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self):
        return len(self.data)
    
    
    def display_sample(self, index):
        img = np.rollaxis(self.data[index], 0, 3)
        imshow(img)
        plt.show()
        mask = np.squeeze(np.rollaxis(self.targets[index], 0, 3))
        imshow(mask)
        plt.show()
        print("x", self.data[index].shape)
        print("y", self.targets[index].shape)
