# from https://github.com/amdegroot/ssd.pytorch
import torch
import cv2
import numpy as np
from numpy import random
import imgaug.augmenters as iaa


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def remove_empty_boxes(boxes, labels):
    """Removes bounding boxes of W or H equal to 0 and its labels

    Args:
        boxes   (ndarray): NP Array with bounding boxes as lines
                           * BBOX[x1, y1, x2, y2]
        labels  (labels): Corresponding labels with boxes

    Returns:
        ndarray: Valid bounding boxes
        ndarray: Corresponding labels
    """
    del_boxes = []
    for idx, box in enumerate(boxes):
        if box[0] == box[2] or box[1] == box[3]:
            del_boxes.append(idx)

    return np.delete(boxes, del_boxes, 0), np.delete(labels, del_boxes)


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
            if boxes is not None:
                boxes, labels = remove_empty_boxes(boxes, labels)
        return img, boxes, labels


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32) * 255

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)

        image = (image - self.mean) / self.std
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, imshape):
        w, h = imshape
        self.shape = (w, h)

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, self.shape)
        return image, boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        # guard against no boxes
        if boxes is not None and boxes.shape[0] == 0:
            return image, boxes, labels
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.max() < min_iou or overlap.min() > max_iou:
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes

## Data augmentation methods used in "A Method of Data Augmentation for Classifying Road Damage Considering Influence on Classification Accuracy" ##

class AddRandomPixelValue(object):
    def __call__(self, image, boxes, classes):
        if random.randint(2) and (1 or 2) in classes:
            image = image + random.uniform(-40,40)
            image[image > 255] = 255
            image[image < 0] = 0
        return image, boxes, classes
    
class AverageNeighborBlur(object):
    def __call__(self, image, boxes, classes):
        if random.randint(2) and (1 or 2 or 4) in classes:
            image = cv2.blur(image,(5,5))
        return image, boxes, classes
    
class GaussianBlur(object):
    def __call__(self, image, boxes, classes):
        if random.randint(2) and (2) in classes:
            image = cv2.GaussianBlur(image,(5,5),0)
        return image, boxes, classes
    
class IvertPixels(object):
    def __call__(self, image, boxes, classes):
        if random.randint(2) and (1 or 4) in classes:
            image = 255-image
        return image, boxes, classes
    
class RandomScalePixelValue(object):
    def __call__(self, image, boxes, classes):
        if random.randint(2) and (1 or 2) in classes:
            scale = random.uniform(0.5, 1.5)
            image = scale*image
            image[image > 255] = 255
            image[image < 0] = 0
        return image, boxes, classes
    
class EmbossAndMerge(object):
    def __call__(self, image, boxes, classes):
        if random.randint(2) and (2) in classes:
            seq = iaa.Sequential([
                iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))
            ])
            image = seq(image=image)
            return image, boxes, classes
        
class DropOut(object):
    def __call__(self, image, boxes, classes):
        if random.randint(2) and (2) in classes:
            seq = iaa.Sequential([
                iaa.Dropout(p=(0.02, 0.02)),
            ])
            image = seq(image=image)
            return image, boxes, classes
    
class D00Transforms(object):
    def __call__(self, image, boxes, classes):
        if random.randint(2) and (1) in classes:
            # AddRandomPixelValue()
            transforms.append(iaa.Add((-40, 40)))
            # AverageNeighborBlur(image, boxes, classes)
            transforms.append(iaa.AverageBlur(k=(2, 11)))
            # GaussianBlur(image, boxes, classes)
            transforms.append(iaa.GaussianBlur(sigma=(0.0, 3.0)))
            # IvertPixels(image, boxes, classes)
            transforms.append(iaa.Invert(1))
            # RandomScalePixelValue(image, boxes, classes)
            transforms.append(iaa.Multiply((0.5, 1.5)))
            seq = iaa.Sequential(transforms)
            image = seq(image=image)
        return image, boxes, classes

class D10Transforms(object):
    def __call__(self, image, boxes, classes):
        if random.randint(2) and (2) in classes:
            transforms = []
            # AddRandomPixelValue()
            transforms.append(iaa.Add((-40, 40)))
            # AverageNeighborBlur(image, boxes, classes)
            transforms.append(iaa.AverageBlur(k=(2, 11)))
            #8: Emboss an image and merge with the original image
            transforms.append(iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)))
            # GaussianBlur(image, boxes, classes)
            transforms.append(iaa.GaussianBlur(sigma=(0.0, 3.0)))
            # RandomScalePixelValue(image, boxes, classes)
            transforms.append(iaa.Multiply((0.5, 1.5)))
            seq = iaa.Sequential(transforms)
            image = seq(image=image)
        return image, boxes, classes
    
class D40Transforms(object):
    def __call__(self, image, boxes, classes):
        if random.randint(2) and (4) in classes:
            transforms = []
            #3: AverageNeighborBlur(image, boxes, classes)
            transforms.append(iaa.AverageBlur(k=(2, 11)))
            #4: Convert 2% of all pixels to black pixels and drop the information
            transforms.append(iaa.Dropout(p=0.02))
            #9: Flip horizontally (allready done)
            #IvertPixels(image, boxes, classes)
            transforms.append(iaa.Invert(1))
            #14: Add pepper noises at 5% of all pixels
            transforms.append(iaa.Pepper(0.05))
            seq = iaa.Sequential(transforms)
            image = seq(image=image)
        return image, boxes, classes
        
        
class ClassSpesificTransforms(object):
    def __call__(self, image, boxes, classes):
        if random.randint(2):
            transforms = []
            if (1 or 2) in classes:
                #AddRandomPixelValue
                transforms.append(iaa.Add((-40, 40)))
            if (1 or 2 or 4) in classes:
                #AverageNeighborBlur
                transforms.append(iaa.AverageBlur(k=(2, 11)))
            if (4) in classes:
                #4: Convert 2% of all pixels to black pixels and drop the information (class 4)
                transforms.append(iaa.Dropout(p=0.02))
            if (2) in classes:
                #8: Emboss an image and merge with the original image (class 2)
                transforms.append(iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)))
            #9: Flip horizontally (allready done, for class 4 otherwise)
            if (2) in classes:
                #image, boxes, classes = GaussianBlur(image, boxes, classes)
                transforms.append(iaa.GaussianBlur(sigma=(0.0, 3.0)))
            if (1 or 4) in classes:
                #image, boxes, classes = IvertPixels(image, boxes, classes)
                transforms.append(iaa.Invert(1))
            if (1 or 2) in classes:
                #image, boxes, classes = RandomScalePixelValue(image, boxes, classes)
                transforms.append(iaa.Multiply((0.5, 1.5)))
            if (4) in classes:
                #14: Add pepper noises at 5% of all pixels (class 4)
                transforms.append(iaa.Pepper(0.05))
            seq = iaa.Sequential(transforms)
            image = seq(image=image)
        return image, boxes, classes

class ClassSpesificTransforms2(object):
    def __call__(self, image, boxes, classes):
        if random.randint(2):
            class_trans = random.choice(classes)
            if class_trans == 1:
                aug = iaa.OneOf([
                    iaa.Add((-40, 40)),
                    iaa.AverageBlur(k=(2, 11)),
                    iaa.Invert(1),
                    iaa.Multiply((0.5, 1.5))
                ])
                image = aug(image=image)

            if class_trans == 2:
                aug = iaa.OneOf([
                    iaa.Add((-40, 40)),
                    iaa.AverageBlur(k=(2, 11)),
                    iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
                    iaa.GaussianBlur(sigma=(0.0, 3.0))
                ])
                image = aug(image=image)
            if class_trans == 4:
                aug = iaa.OneOf([
                    iaa.AverageBlur(k=(2, 11)),
                    iaa.Dropout(p=0.02),
                    #9: Flip horizontally (allready done, for class 4 otherwise)
                    iaa.Invert(1),
                    iaa.Pepper(0.05)
                ])
                image = aug(image=image)
            
        return image, boxes, classes    

    
"""class ClassSpesificTransforms(object):
    def __call__(self, image, boxes, classes):
        if random.randint(2): 
            if (1 or 2) in classes:
                #image, boxes, classes = AddRandomPixelValue()
                image = image + random.uniform(-40,40)
                image[image > 255] = 255
                image[image < 0] = 0
            if (1 or 2 or 4) in classes:
                #image, boxes, classes = AverageNeighborBlur(image, boxes, classes)
                image = cv2.blur(image,(5,5))
            #4: Convert 2% of all pixels to black pixels and drop the information (class 4)
            #8: Emboss an image and merge with the original image (class 2)
            
            #9: Flip horizontally (allready done, for class 4 otherwise)
            if (2) in classes:
                #image, boxes, classes = GaussianBlur(image, boxes, classes)
                image = cv2.GaussianBlur(image,(5,5),0)
            if (1 or 4) in classes:
                #image, boxes, classes = IvertPixels(image, boxes, classes)
                image = 255-image
            if (1 or 2) in classes:
                #image, boxes, classes = RandomScalePixelValue(image, boxes, classes)
                scale = random.uniform(0.5, 1.5)
                image = scale*image
                image[image > 255] = 255
                image[image < 0] = 0
            #14: Add pepper noises at 5% of all pixels (class 4)
        return image, boxes, classes"""


