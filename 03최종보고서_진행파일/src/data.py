import albumentations as A
import cv2
import numpy as np
import os
import pandas as pd
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader, ConcatDataset

csv = pd.read_csv('./open/train_df.csv', index_col=0)

classes = sorted(np.unique(csv["class"]))
states = {class_:[] for class_ in classes}
for class_ in classes:
    states[class_] += [state for state in sorted(np.unique(csv[csv["class"]==class_]["state"]))]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.3, 0.3, 0.3), (0.3, 0.3, 0.3))
])

def img_load(path, size):
    img = cv2.imread(path)[:,:,::-1]
    if size[0] is None:
        pass
    else:
        img = cv2.resize(img, size)
    return img

def make_paths(path):
    classes = [class_ for class_ in sorted(os.listdir(path)) if os.path.isdir(f"{path}/{class_}")]

    states = {key:[] for key in classes}
    temp = {class_:sorted(os.listdir(f"{path}/{class_}/{criteria}")) for class_ in classes for criteria in ['train', 'test']}
    for key in temp.keys():
        states[key] += temp[key]
    states = {key:sorted(list(set(states[key]))) for key in states.keys()}
    train_paths = sorted([f"{path}/{class_}/{criteria}/{state}/{image}" for class_ in classes \
        for criteria in ['train', 'test'] for state in sorted(os.listdir(f"{path}/{class_}/{criteria}"))\
        for image in sorted(os.listdir(f"{path}/{class_}/{criteria}/{state}")) if int(image.split('.')[0])<20000])
    test_paths = sorted([f"{path}/{class_}/{criteria}/{state}/{image}" for class_ in classes \
        for criteria in ['train', 'test'] for state in sorted(os.listdir(f"{path}/{class_}/{criteria}"))\
        for image in sorted(os.listdir(f"{path}/{class_}/{criteria}/{state}")) if int(image.split('.')[0])>=20000])
    
    return train_paths, test_paths

class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, transforms=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transforms=transforms
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = self.img_paths[idx]
        if self.transforms is not None:
            if type(self.transforms)==list:
                self.transforms = A.Compose(self.transforms)
            img = self.transforms(image=img)['image']
        img = transform(img)
        label = self.labels[idx]
        return img, label
    
def make_dataset(size, train_paths, test_paths, class_, augmentation=None, batch_size=32):
    
    # # 2 Stage
    # train_labels = csv[csv["class"]==class_]["state"]
    
    # label_unique = sorted(np.unique(train_labels))
    # label_to_num = {key:value for key,value in zip(label_unique, range(len(label_unique)))}
    # num_to_label = {value:key for key,value in zip(label_unique, range(len(label_unique)))}

    # train_labels = [label_to_num[x.split('/')[4]] for x in train_paths if x.split('/')[2]==class_ and x.split('/')[4] in label_unique]
    # test_labels = [label_to_num[x.split('/')[4]] for x in test_paths if x.split('/')[2]==class_ and x.split('/')[4] in label_unique]
    
    # train_png = [x for x in train_paths if x.split('/')[2]==class_ and x.split('/')[4] in label_unique]
    # test_png = [x for x in test_paths if x.split('/')[2]==class_ and x.split('/')[4] in label_unique]
    
    
    # 3 Stage
    train_labels = [x if x=='good' else 'bad' for x in csv[csv['class']=='bottle']['state']]
    
    label_unique = sorted(np.unique(train_labels))
    label_to_num = {key:value for key,value in zip(label_unique, range(len(label_unique)))}
    num_to_label = {value:key for key,value in zip(label_unique, range(len(label_unique)))}
    
    train_labels = [label_to_num['good'] if y.split('/')[4]=='good' else label_to_num['bad'] for y in [x for x in train_paths if x.split('/')[2]==class_]]
    test_labels = [label_to_num['good'] if y.split('/')[4]=='good' else label_to_num['bad'] for y in [x for x in test_paths if x.split('/')[2]==class_]]
    
    train_png = [x for x in train_paths if x.split('/')[2]==class_]
    test_png = [x for x in test_paths if x.split('/')[2]==class_]
    
    train_imgs = [img_load(m, size) for m in train_png]
    test_imgs = [img_load(n, size) for n in test_png]
    
    train_dataset = Custom_dataset(np.array(train_imgs), np.array(train_labels), transforms=None)
    test_dataset = Custom_dataset(np.array(test_imgs), np.array(test_labels), transforms=None)
    
    if augmentation is not None:
        train_datasets = []
        train_datasets.append(train_dataset)
        
        train_labels_bad = [label for label in train_labels if num_to_label[label] != 'good']
        train_png_bad = [png for idx, png in enumerate(train_imgs) if num_to_label[train_labels[idx]] != 'good']
        
        for aug in augmentation:
            train_datasets += [Custom_dataset(np.array(train_png_bad), np.array(train_labels_bad), aug)]
            
        train_dataset = ConcatDataset(train_datasets)
        print(f'class: {class_}, original dataset: {len(train_imgs)}, augmented data: {len(train_png_bad)*len(augmentation)}, augmented dataset : {len(train_dataset)}')
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
    
    return train_loader, test_loader, len(label_unique), num_to_label