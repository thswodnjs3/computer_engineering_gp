import albumentations as A

from glob import glob
import pandas as pd
import numpy as np 
import cv2

import os
import random

import torch
from torch.utils.data import ConcatDataset
from sklearn.metrics import accuracy_score
import time

# make variables
device = torch.device('cuda')
A_list = {
    # 'NO': None,
    'HF': A.HorizontalFlip(p=1),
    'Affine': A.Affine((45, -45), p=1)
    # 'VF': A.VerticalFlip(p=1),
    # 'BF': [A.HorizontalFlip(p=1), A.VerticalFlip(p=1)],
    # '90CAR': [A.CenterCrop(int(512*0.9), int(512*0.9), p=1), A.Resize(512, 512, p=1)],
    # '95CAR': [A.CenterCrop(int(512*0.95), int(512*0.95), p=1), A.Resize(512, 512, p=1)],
    # 'R': A.Rotate(limit=360, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    # 'SH12': A.ShiftScaleRotate(shift_limit=0.12, scale_limit=0, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    # 'SC20': A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    # 'SCSH': A.ShiftScaleRotate(shift_limit=0.12, scale_limit=0.2, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    # 'SHR': A.ShiftScaleRotate(shift_limit=0.12, scale_limit=0, rotate_limit=360, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    # 'SCR': A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=360, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    # 'SCSHR': A.ShiftScaleRotate(shift_limit=0.12, scale_limit=0.2, rotate_limit=360, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    # 'RRC': A.RandomResizedCrop(height=512, width=512, scale=(0.3, 1.0), p=1),
    # 'GN': A.GaussNoise(p=1),
    # 'ISO': A.ISONoise(p=1),
    # 'GD': A.GridDistortion(p=1),
    # 'P': A.Perspective(p=1),
    # 'GB': A.GaussianBlur(p=1),
    # 'MB': A.MotionBlur(p=1),
    # 'IC': A.ImageCompression(p=1),
    # 'CJ': A.ColorJitter(p=1),
    # 'RF': A.RandomFog(p=1),
    # 'RG': A.RandomGamma(p=1),
    # 'RR': A.RandomRain(p=1),
    # 'RSh': A.RandomShadow(p=1),
    # 'RSn': A.RandomSnow(p=1),
    # 'RSF': A.RandomSunFlare(p=1)
}
train_y = pd.read_csv("open/train_df.csv")
train_y['good'] = ['good' if x=='good' else 'bad' for x in list(train_y['state'])]

classes = sorted(np.unique(train_y['class']))
goods = sorted(np.unique(train_y['good']))
states = sorted(np.unique(train_y['state']))