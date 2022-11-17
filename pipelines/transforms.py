import numpy as np

import torch

import cv2

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        """
        Args: 
            sample (dict): 
                {'image': np.arr, 'class': int, 'bbox': [xmin, ymin, xmax, ymax]}
        Returns: 
            sample (dict): same as input
        """

        image, class_indices, bboxes = sample['image'], sample['class'], sample['bbox']

        # image resize
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        bboxes_out = []
        for bbox in bboxes:

            # FIXME
            xmin, ymin, xmax, ymax = bbox

            xmin = int(xmin * new_w / w)
            xmax = int(xmax * new_w / w)

            ymin = int(ymin * new_h / h)
            ymax = int(ymax * new_h / h)

            bboxes_out.append([xmin, ymin, xmax, ymax])

        return {'image': image, 'class':class_indices, 'bbox': bboxes_out}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        """
        Args: 
            sample (dict): 
                {'image': np.arr, 'class': int, 'bbox': [xmin, ymin, xmax, ymax]}
        Returns: 
            sample (dict): same as input
        """

        image, class_indices, bboxes = sample['image'], sample['class'], sample['bbox']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        bboxes_out = []
        for bbox in bboxes:

            # FIXME
            xmin, ymin, xmax, ymax = bbox

            xmin -= left
            xmax -= left

            ymin -= top
            ymax -= top

            bboxes_out.append([xmin, ymin, xmax, ymax])

        return {'image': image, 'class':class_indices, 'bbox': bboxes_out}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}