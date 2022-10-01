import cv2
import numpy as np
import os
import pickle
import shutil
import time
from tqdm.notebook import tqdm

import pandas as pd

import torch

from IPython.display import clear_output

def export_original_data(path):
    original_data = []

    for class_ in tqdm(sorted(os.listdir(path))):
        class_path = path+'/'+class_
        if os.path.isdir(class_path)==False:
            continue
        
        for criteria in os.listdir(class_path):
            criteria_path = class_path+'/'+criteria
            
            for state in os.listdir(criteria_path):
                state_path = criteria_path+'/'+state
                
                for image in sorted(os.listdir(state_path)):
                    image_path = state_path+'/'+image
                    
                    img = cv2.imread(image_path)
                    original_data.append((image_path, img))
                
    original_pd = pd.DataFrame()
    original_pd['path'] = [x[0] for x in original_data]
    original_pd['img'] = [x[1] for x in original_data]
    original_pd['size'] = [x[1].shape for x in original_data]
    
    with open("./pickle/original_pd.pickle", "wb") as fw:
        pickle.dump(original_pd, fw)
    return original_pd

def export_contest_data(path, criteria):
    result = []
    path = path+'/'+criteria
    
    if criteria=='train':
        train_pd = pd.read_csv('./open/train_df.csv', index_col=0)

    for image in tqdm(os.listdir(path)):
        image_path = path+'/'+image
        img = cv2.imread(image_path)
        
        if criteria=='train':
            class_ = train_pd[train_pd['file_name']==image]['class'].values[0]
            state = train_pd[train_pd['file_name']==image]['state'].values[0]
        
            result.append((image, class_, state, img))
        else:
            result.append((image, img))
    
    with open("./pickle/contest_"+criteria+"_data.pickle", "wb") as fw:
        pickle.dump(result, fw)
    
    return result

def make_mapping(original_pd, contest_train_data, contest_test_data):
    name_mapping = []
    already_exist = []

    for contest_name, contest_class, contest_state, contest_image in tqdm(contest_train_data):
        for original_idx, original_row in original_pd[original_pd['path'].str.contains(contest_class) & original_pd['path'].str.contains(contest_state)].iterrows():
            original_path, original_image, original_size = original_row.values
            if np.array_equal(contest_image, original_image):
                name_mapping.append((contest_name, original_path))
                already_exist.append(original_path)
                break

    name_mapping_paths = [x[1] for x in name_mapping]

    for contest_name, contest_image in tqdm(contest_test_data):
        for original_idx, original_row in original_pd[original_pd['path'].isin(name_mapping_paths)==False].iterrows():
            original_path, original_image, original_size = original_row.values
            if np.array_equal(contest_image, original_image):
                name_mapping.append((contest_name, original_path))
                already_exist.append(original_path)
                break
    return name_mapping, already_exist

class make_dataset():
    def __init__(self, csv_path, src_path, dest_path):
        self.data = pd.read_csv(csv_path, index_col=0)
        self.src_path = src_path
        self.dest_path = dest_path
        
    def make_folder(self):
        if os.path.isdir(self.dest_path)==False:
            os.mkdir(self.dest_path)

        for c in self.data.columns:
            if c in ['class', 'state', 'label']:
                class_path = self.dest_path + '/' + c
                if os.path.isdir(class_path)==False:
                    os.mkdir(class_path)
                for feature in self.data[c].unique():
                    feature_path = class_path + '/' + feature
                    if os.path.isdir(feature_path)==False:
                        os.mkdir(feature_path)
                    
    def sort_image(self, feature, verbose=True):
        src_path = self.src_path
        dest_path = self.dest_path

        print("Sorting...")
        for label in tqdm(self.data[feature].unique(), desc=feature):

            for name in tqdm(self.data[self.data[feature]==label]['file_name'],desc=label, leave=False):
                from_ = src_path + '/' + name
                to = dest_path + '/' + feature + '/' + label + '/' + name

                if os.path.isfile(from_):
                    shutil.copyfile(from_, to)

        if verbose:
            clear_output()

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

def collate_fn(samples):
    images, labels = zip(*samples)
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels