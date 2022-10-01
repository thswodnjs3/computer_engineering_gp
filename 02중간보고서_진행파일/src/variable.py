import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import f1_score

size = 512

device = torch.device('cuda')
# device = torch.device('cpu')
csv = pd.read_csv('./open/train_df.csv', index_col=0)
classes = sorted(np.unique(csv["class"]))
states = {class_:sorted(np.unique(csv[csv["class"]==class_]["state"])) for class_ in classes}

A_list = {
    'no_augmentation': None,
    'HF': A.HorizontalFlip(p=1), 'VF': A.VerticalFlip(p=1), 'BF': [A.HorizontalFlip(p=1), A.VerticalFlip(p=1)],
    '90CAR': [A.CenterCrop(int(size*0.9), int(size*0.9), p=1), A.Resize(size, size, p=1)],
    '95CAR': [A.CenterCrop(int(size*0.95), int(size*0.95), p=1), A.Resize(size, size, p=1)],
    'R': A.Rotate(limit=360, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    'SH12': A.ShiftScaleRotate(shift_limit=0.12, scale_limit=0, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    'SC20': A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    'SCSH': A.ShiftScaleRotate(shift_limit=0.12, scale_limit=0.2, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    'SHR': A.ShiftScaleRotate(shift_limit=0.12, scale_limit=0, rotate_limit=360, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    'SCR': A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=360, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    'SCSHR': A.ShiftScaleRotate(shift_limit=0.12, scale_limit=0.2, rotate_limit=360, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    'RRC': A.RandomResizedCrop(height=512, width=512, scale=(0.3, 1.0), p=1),
    'GN': A.GaussNoise(p=1),
    'ISO': A.ISONoise(p=1),
    'GD': A.GridDistortion(p=1),
    'P': A.Perspective(p=1),
    'GB': A.GaussianBlur(p=1),
    'MB': A.MotionBlur(p=1),
    'IC': A.ImageCompression(p=1),
    'CJ': A.ColorJitter(p=1),
    'RF': A.RandomFog(p=1),
    'RG': A.RandomGamma(p=1),
    'RR': A.RandomRain(p=1),
    'RSh': A.RandomShadow(p=1),
    'RSn': A.RandomSnow(p=1),
    'RSF': A.RandomSunFlare(p=1)
}

criterion_list = {
    'CE': nn.CrossEntropyLoss(),
    'NLL': nn.NLLLoss(),
    'BCE': nn.BCELoss()
}

scaler_list = {
    'GS': torch.cuda.amp.GradScaler()
}
                                                                                        

def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score