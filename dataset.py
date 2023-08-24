from typing import Callable
import os
import os.path
from os.path import exists
import pandas as pd
import numpy as np
import cv2
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations
from albumentations.pytorch import ToTensorV2

PRE__MEAN = [0.485, 0.456, 0.406]
PRE__STD = [0.229, 0.224, 0.225]

def ApplyWeightedRandomSampler(dataset_csv):
    dataframe = pd.read_csv(dataset_csv) # head: image_path, label
    class_counts = dataframe.label.value_counts()

    sample_weights = [1/class_counts[i] for i in dataframe.label.values]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataframe), replacement=True)
    return sampler

class TrainDataset(Dataset):

    def __init__(self, csv_file, input_shape=(224, 224)):
        #self.image_dir = image_dir
        self.dataframe = pd.read_csv(csv_file)
        self.composed_transformations = albumentations.Compose([
            albumentations.Resize(height=256, width=256),
            albumentations.RandomCrop(height=input_shape[0], width=input_shape[0]),
            albumentations.HorizontalFlip(),
            albumentations.RandomGamma(gamma_limit=(80, 180)), # 0.5, 1.5
            albumentations.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
            albumentations.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            albumentations.Normalize(PRE__MEAN, PRE__STD, always_apply=True),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.dataframe)

    def get_labels(self):
        return self.dataframe.iloc[:, 1]

    def __getitem__(self, idx):

        img_path = self.dataframe.iloc[idx, 0]
        label_str = self.dataframe.iloc[idx, 1]

        image = cv2.imread(img_path)
        if image is None:
            raise Exception('Error: Image is None.')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = 1 if label_str == 'bonafide' else 0
        map_x = torch.ones((14,14)) if label == 1 else torch.zeros((14, 14))

        image = self.composed_transformations(image = image)['image']

        return {
            "images": image,
            "labels": torch.tensor(label, dtype = torch.float),
            "map": map_x
        }

class TestDataset(Dataset):

    def __init__(self, csv_file, input_shape=(224, 224)):
        self.dataframe = pd.read_csv(csv_file)
        self.composed_transformations = albumentations.Compose([
            albumentations.Resize(height=input_shape[0], width=input_shape[1]),
            albumentations.Normalize(PRE__MEAN, PRE__STD, always_apply=True),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label_str = self.dataframe.iloc[idx, 1]

        image = cv2.imread(img_path)
        if image is None:
            raise Exception('Error: Image is None.')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = 1 if label_str == 'bonafide' else 0

        image = self.composed_transformations(image=image)['image']

        return {
            "images": image,
            "labels": torch.tensor(label, dtype = torch.float),
            "img_path": img_path
        }
