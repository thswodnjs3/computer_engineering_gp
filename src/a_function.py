import cv2
import numpy as np
import os
import pickle

from glob import glob
from sklearn.metrics import f1_score

def img_load(path, size):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, (size, size))
    return img

def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score

def load_pickle(path):
    output = None
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            output = pickle.load(fr)
    return output

def store_pickle(path, data):
    with open(path, 'wb') as fw:
        pickle.dump(data, fw)
        
def make_imgs(section, size, train_y=None, condition=None, target=None, condition2=None, target2=None):
    if condition is None:
        png = sorted(glob(f'./open/{section}/*.png'))
        imgs = [img_load(m, size) for m in png]
    elif condition2 is None:
        png = list(train_y[train_y[condition]==target]['file_name'])
        imgs = [img_load(f'./open/train/{m}', size) for m in png]
    else:
        png = list(train_y[(train_y[condition]==target)&(train_y[condition2]==target2)]['file_name'])
        imgs = [img_load(f'./open/train/{m}', size) for m in png]
    return imgs

def make_labels(train_y,
                condition=None, target=None, output=None,
                condition2=None, target2=None):
    if condition is None:
        dataframe = train_y[target]
    elif condition2 is None:
        dataframe = train_y[train_y[condition]==target][output]
    else:
        dataframe = train_y[(train_y[condition]==target)&(train_y[condition2]==target2)][output]
    label_unique = sorted(np.unique(dataframe))
    label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}
    train_labels = [label_unique[k] for k in dataframe]
    
    return train_labels, label_unique