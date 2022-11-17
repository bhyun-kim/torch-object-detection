

from glob import glob 

import os.path as osp

import cv2 

from torch.utils.data import Dataset

class COCODataset(Dataset):
    def __init__(self, root_dir, classes=None, transform=None):
        """
        Args: 
            root_dir (str): Root directory to dataset 
            classes (list): class list contain strings
            transform (torch.transforms)
        
        """

    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        return sample
