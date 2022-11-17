

from glob import glob 

import os.path as osp

import cv2 
import xmltodict

from torch.utils.data import Dataset

class PASCALVOCDataset(Dataset):
    def __init__(self, root_dir, classes=None, transform=None):
        """
        Args: 
            root_dir (str): Root directory to dataset 
            classes (list): class list contain strings
            transform (torch.transforms)
        
        """

        self.root_dir = root_dir
        self.img_list = glob(osp.join(root_dir, 'JPEGImages', '*.jpg'))
        self.transform = transform

        if classes:
            self.classes = classes
        else : 
            # PASCAL VOC classes
            self.classes = [
                'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor'
                ]
            
        #train / val split 
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        """Return image and bbox information 
        Args: 
            idx (int): index of data
        
        Returns: 
            img (np.arr)
            bboxes (list): contains [class_idx, xmin, ymin, xmax, ymax]
        """

        img_path = self.img_list[idx]
        ann_path = img_path.replace('JPEGImages', 'Annotations')
        ann_path = ann_path.replace('.jpg', '.xml')

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with open(ann_path) as ann_file:
            ann = ann_file.read()
            ann_dict = xmltodict.parse(ann)

        objects = ann_dict['annotation']['object']

        bboxes = []
        class_indices = []

        if isinstance(objects, list):
            pass
        else : 
            objects = [objects]

        for obj in objects :
            class_idx = self.classes.index(obj['name'])
            bbox = obj['bndbox']
            xmin, ymin = int(bbox['xmin']), int(bbox['ymin'])
            xmax, ymax = int(bbox['xmax']), int(bbox['ymax'])
            
            class_indices.append(class_idx)
            bboxes.append([xmin, ymin, xmax, ymax])
            
        sample = {'image': img, 'class':class_indices, 'bbox': bboxes}

        if self.transform: 
            sample = self.transform(sample)

        return sample
