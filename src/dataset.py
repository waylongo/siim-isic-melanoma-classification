import os
import cv2
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensor
from configs import *

class MelanomaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms=None, meta_features=None):
        """
        Class initialization
        Args:
            df (pd.DataFrame): DataFrame with data description
            imfolder (str): folder with images
            train (bool): flag of whether a training dataset is being initialized or testing one
            transforms: image transformation method to be applied
        """

        self.df = df
        self.imfolder = imfolder
        self.train = train
        self.transforms = transforms
        self.meta_features = meta_features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        im_path = os.path.join(self.imfolder, self.df.iloc[index]['image_name'] + '.jpg')
        x = cv2.imread(im_path)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        meta = np.array(self.df.iloc[index][self.meta_features].values, dtype=np.float32)

        if self.transforms:
            augmented = self.transforms(image=x)
            x = augmented['image']

        if self.train:
            y = self.df.loc[index]['target']
            return (x, meta), y
        else:
            return (x, meta)
    
    def get_labels(self):
        return list(self.df['target'].values)


class Microscope(A.ImageOnlyTransform):
    def __init__(self, p: float = 0.5, always_apply=False):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        if random.random() < self.p:
            circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8),
                                (img.shape[0]//2, img.shape[1]//2),
                                random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15),
                                (0, 0, 0),
                                -1)

            mask = circle - 255
            img = np.multiply(img, mask)

        return img


class HairRemove(A.ImageOnlyTransform):
    def apply(self, img, **params):
        if random.random() < self.p:
            # convert image to grayScale
            grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # kernel for morphologyEx
            kernel = cv2.getStructuringElement(1, (17, 17))

            # apply MORPH_BLACKHAT to grayScale image
            blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

            # apply thresholding to blackhat
            _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

            # inpaint with original image and threshold image
            img = cv2.inpaint(img, threshold, 1, cv2.INPAINT_TELEA)

        return img


class ColorConstancy(A.ImageOnlyTransform):
    def apply(self, img, power=6, gamma=None, **params):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_dtype = img.dtype

        if gamma is not None:
            img = img.astype('uint8')
            look_up_table = np.ones((256,1), dtype='uint8') * 0
            for i in range(256):
                look_up_table[i][0] = 255*pow(i/255, 1/gamma)
            img = cv2.LUT(img, look_up_table)

        img = img.astype('float32')
        img_power = np.power(img, power)
        rgb_vec = np.power(np.mean(img_power, (0,1)), 1/power)
        rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
        rgb_vec = rgb_vec/rgb_norm
        rgb_vec = 1/(rgb_vec*np.sqrt(3))
        img = np.multiply(img, rgb_vec)

        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

        return img.astype(img_dtype)




def get_tta_transform(tta_idx):

    if tta_idx == 0: # original
        tta_test_transform = A.Compose([
            ColorConstancy(p=1),
            A.Normalize(),
            ToTensor(),
        ])
    elif tta_idx == 1: # horizontal flip
        tta_test_transform = A.Compose([
            ColorConstancy(p=1),
            A.HorizontalFlip(p=1),
            A.Normalize(),
            ToTensor(),
        ])
    elif tta_idx == 2: # vertical flip
        tta_test_transform = A.Compose([
            ColorConstancy(p=1),
            A.VerticalFlip(p=1),
            A.Normalize(),
            ToTensor(),
        ])
    elif tta_idx == 3: # random rotate 90 degree
        tta_test_transform = A.Compose([
            ColorConstancy(p=1),
            A.RandomRotate90(p=1),
            A.Normalize(),
            ToTensor(),
        ])
    elif tta_idx == 4: # flip and transpose
        tta_test_transform = A.Compose([
            ColorConstancy(p=1),
            A.Flip(p=0.5),
            A.Transpose(p=1),
            A.Normalize(),
            ToTensor(),
        ])
    else: # train transform
        tta_test_transform = A.Compose([
            ColorConstancy(p=1),
            A.OneOf([
                A.Flip(p=0.5),
                A.IAAFliplr(p=0.5),
                A.Transpose(p=0.5),
                A.IAAFlipud(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.Rotate(limit=365, p=0.75),
                A.ShiftScaleRotate(p=0.75),
            ], p=0.75),
            A.Blur(blur_limit=3, p=0.5),
            A.Normalize(),
            ToTensor(),
        ])

    return tta_test_transform


train_transform = A.Compose([
    ColorConstancy(p=1),
    A.OneOf([
        A.Flip(p=0.5),
        A.IAAFliplr(p=0.5),
        A.Transpose(p=0.5),
        A.IAAFlipud(p=0.5),
    ], p=0.5),
    A.OneOf([
        A.Rotate(limit=365, p=0.5),
        A.ShiftScaleRotate(p=0.5),
    ], p=0.5),
    # A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.25),
    A.Blur(blur_limit=3, p=0.5),
    A.Cutout(num_holes=6, max_h_size=int(IMG_SIZE * 0.125), max_w_size=int(IMG_SIZE * 0.125), p=0.5),
    A.Normalize(),
    ToTensor(),
])


