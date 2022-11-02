

from glob import glob 

import os.path as osp

import cv2 
import xmltodict

from torch.utils.data import Dataset

class PASCALVOCDataset(Dataset):
    def __init__(self, root_dir, classes='default'):

        self.root_dir = root_dir
        self.img_list = glob(osp.join(root_dir, 'JPEGImages', '*.jpg'))

        if classes == 'default':
            self.classes = [
                'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor'
                ]
        else : 
            self.classes = classes

        #train / val split 
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

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

        if isinstance(objects, list):
            pass
        else : 
            objects = [objects]

        for obj in objects :
            class_idx = self.classes.index(obj['name'])
            bbox = obj['bndbox']
            xmin, ymin = int(bbox['xmin']), int(bbox['ymin'])
            xmax, ymax = int(bbox['xmax']), int(bbox['ymax'])

            bboxes.append([class_idx, xmin, ymin, xmax, ymax])
            
        return img, bboxes
