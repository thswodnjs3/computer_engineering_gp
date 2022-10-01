import cv2
import numpy as np
import os
import pickle
import random

from collections import Counter
from glob import glob
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from src.module import Custom_dataset
from src.variable import csv, A_list

def img_load(path, size):
    img = cv2.imread(path)[:,:,::-1]
    if size is None:
        pass
    else:
        img = cv2.resize(img, (size, size))
    return img


def make_paths(path, mode):
    classes = [class_ for class_ in sorted(os.listdir(path)) if os.path.isdir(f"{path}/{class_}")]        
    
    if mode=="image":
        criterias = ['train', 'test']
        temp = {class_:sorted(os.listdir(f"{path}/{class_}/{criteria}")) for class_ in classes for criteria in criterias}
    elif mode=="mask":
        criterias = ["ground_truth"]
        temp = {class_:sorted(os.listdir(f"{path}/{class_}/{criteria}")) for class_ in classes for criteria in criterias}
    
    states = {key:[] for key in classes}
    for key in temp.keys():
        states[key] += temp[key]
    states = {key:sorted(list(set(states[key]))) for key in states.keys()}
    
    if mode=="image":
        train_paths = sorted([f"{path}/{class_}/{criteria}/{state}/{image}"\
            for class_ in classes \
            for criteria in criterias\
            for state in sorted(os.listdir(f"{path}/{class_}/{criteria}"))\
            for image in sorted(os.listdir(f"{path}/{class_}/{criteria}/{state}"))\
            if int(image.split('.')[0])<20000])
        
        test_paths = sorted([f"{path}/{class_}/{criteria}/{state}/{image}"\
            for class_ in classes \
            for criteria in criterias\
            for state in sorted(os.listdir(f"{path}/{class_}/{criteria}"))\
            for image in sorted(os.listdir(f"{path}/{class_}/{criteria}/{state}"))\
            if int(image.split('.')[0])>=20000])
        
        return train_paths, test_paths
    elif mode=="mask":
        mask_paths = sorted([f"{path}/{class_}/{criteria}/{state}/{mask}"\
            for class_ in classes\
            for criteria in criterias\
            for state in sorted(os.listdir(f"{path}/{class_}/{criteria}"))\
            for mask in sorted(os.listdir(f"{path}/{class_}/{criteria}/{state}"))])
        return mask_paths

def load_pickle(path):
    status = False
    if os.path.isfile(path):
        status = True
        with open(path, 'rb') as fr:
            output = pickle.load(fr)
    else:
        status = False
        output = None
    return status, output

def save_pickle(target, path):
    with open(path, 'wb') as fw:
        pickle.dump(target, fw)
        
def make_dataset(mode, size, train_paths, test_paths, class_, batch_size=32, toTensor=True, normalize=[(0.3, 0.3, 0.3), (0.3, 0.3, 0.3)],
                 augmentation=None, augmented_report=False, augmented_target='bad', augmented_ratio=0, show_dataratio=False, test_aug=False, weighted_loss=False):
    
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
    if mode=='class':
        train_labels = csv['class']
    elif mode=='good':
        train_labels = [x if x=='good' else 'bad' for x in csv[csv['class']==class_]['state']]
    elif mode=='state':
        train_labels = [x for x in csv[csv['class']==class_]['state'] if x!='good']
    
    label_unique = sorted(np.unique(train_labels))
    label_to_num = {key:value for key,value in zip(label_unique, range(len(label_unique)))}
    num_to_label = {value:key for key,value in zip(label_unique, range(len(label_unique)))}
    
    if mode=='class':
        status1, train_labels = load_pickle('./pickle/train_labels_class.pickle')
        status2, test_labels = load_pickle('./pickle/test_labels_class.pickle')
        if status1==False:
            train_labels = [label_to_num[x.split('/')[2]] for x in train_paths]
            save_pickle(train_labels, './pickle/train_labels_class.pickle')
        if status2==False:
            test_labels = [label_to_num[x.split('/')[2]] for x in test_paths]
            save_pickle(test_labels, './pickle/test_labels_class.pickle')
            
        status3, train_imgs = load_pickle('./pickle/train_imgs_class.pickle')
        status4, test_imgs = load_pickle('./pickle/test_imgs_class.pickle')
        if status3==False:
            train_png = [x for x in train_paths]
            train_imgs = [img_load(m, size) for m in train_png]
            save_pickle(train_imgs, './pickle/train_imgs_class.pickle')
        if status4==False:
            test_png = [x for x in test_paths]
            test_imgs = [img_load(n, size) for n in test_png]
            save_pickle(test_imgs, './pickle/test_imgs_class.pickle')
    else:
        status1, train_labels = load_pickle(f'./pickle/train_labels_{class_}_{mode}.pickle')
        status2, test_labels = load_pickle(f'./pickle/test_labels_{class_}_{mode}.pickle')
        if status1==False:
            if mode=='good':
                train_labels = [label_to_num['good'] if y.split('/')[4]=='good' else label_to_num['bad'] for y in [x for x in train_paths if x.split('/')[2]==class_]]
            elif mode=='state':
                train_labels = [label_to_num[x.split('/')[4]] for x in train_paths if x.split('/')[2]==class_ and x.split('/')[4]!='good']
            save_pickle(train_labels, f'./pickle/train_labels_{class_}_{mode}.pickle')
        if status2==False:
            if mode=='good':
                test_labels = [label_to_num['good'] if y.split('/')[4]=='good' else label_to_num['bad'] for y in [x for x in test_paths if x.split('/')[2]==class_]]
            elif mode=='state':
                test_labels = [label_to_num[x.split('/')[4]] for x in test_paths if x.split('/')[2]==class_ and x.split('/')[4]!='good']
            save_pickle(test_labels, f'./pickle/test_labels_{class_}_{mode}.pickle')
        
        status3, train_imgs = load_pickle('./pickle/train_imgs_{class_}_{mode}.pickle')
        status4, test_imgs = load_pickle('./pickle/test_imgs_{class_}_{mode}.pickle')
        if status3==False:
            if mode=='good':
                train_png = [x for x in train_paths if x.split('/')[2]==class_]
            elif mode=='state':
                train_png = [x for x in train_paths if x.split('/')[2]==class_ and x.split('/')[4]!='good']
            train_imgs = [img_load(m, size) for m in train_png]
            save_pickle(train_imgs, f'./pickle/train_imgs_{class_}_{mode}.pickle')
        if status4==False:
            if mode=='good':
                test_png = [x for x in test_paths if x.split('/')[2]==class_]
            elif mode=='state':
                test_png = [x for x in test_paths if x.split('/')[2]==class_ and x.split('/')[4]!='good']
            test_imgs = [img_load(n, size) for n in test_png]
            save_pickle(test_imgs, f'./pickle/test_imgs_{class_}_{mode}.pickle')
    
    if show_dataratio:
        ratio = Counter([num_to_label[x] for x in train_labels])
        for idx, k in enumerate(ratio.keys()):
            if idx+1==len(ratio):
                print(f'{k}: {ratio[k]}')
            else:
                print(f'{k}: {ratio[k]}', end=', ')
    
    train_dataset = Custom_dataset('train', np.array(train_imgs), np.array(train_labels), transforms=None, toTensor=toTensor, normalize=normalize)
    test_dataset = Custom_dataset('test', np.array(test_imgs), np.array(test_labels), transforms=None, toTensor=toTensor, normalize=normalize)
    
    if augmentation is not None and len(augmentation)!=0:
        if class_ is not None:
            train_datasets = []
            train_datasets.append(train_dataset)
            
            train_labels_good = [label for label in train_labels if num_to_label[label]=='good']
            train_png_good = [png for idx, png in enumerate(train_imgs) if num_to_label[train_labels[idx]]=='good']
            
            train_labels_bad = [label for label in train_labels if num_to_label[label]!='good']
            train_png_bad = [png for idx, png in enumerate(train_imgs) if num_to_label[train_labels[idx]]!='good']
            
            for aug in augmentation:
                train_datasets += [Custom_dataset('train', np.array(train_png_bad), np.array(train_labels_bad), transforms=aug, toTensor=toTensor, normalize=normalize)]
            if augmented_target=='both':
                for _ in range(augmented_ratio):
                    train_datasets += [Custom_dataset('train', np.array(train_png_good), np.array(train_labels_good), transforms=random.choice(augmentation), toTensor=toTensor, normalize=normalize)]
                
            train_dataset = ConcatDataset(train_datasets)
            if augmented_report:
                if mode=='good':
                    print(f'class: {class_}, original dataset: {len(train_imgs)}[good:{len(train_labels_good)},bad:{len(train_labels_bad)}], augmented data: {(len(train_labels_good)*augmented_ratio)+(len(train_labels_bad)*len(augmentation))}[good:{len(train_labels_good)*augmented_ratio}, bad:{len(train_labels_bad)*len(augmentation)}], augmented dataset: {len(train_imgs)+(len(train_labels_good)*augmented_ratio)+(len(train_labels_bad)*len(augmentation))}, good ratio:{(len(train_labels_good)+(len(train_labels_good)*augmented_ratio))*100/(len(train_imgs)+(len(train_labels_good)*augmented_ratio)+(len(train_labels_bad)*len(augmentation))):.2f}%')
                elif mode=='state':
                    ratio = Counter(train_labels_bad)
                    for idx, k in enumerate(ratio.keys()):
                        if idx==0:
                            print(f'Original: {len(train_labels_bad)}[', end='')
                        if idx+1==len(ratio):
                            print(f'{num_to_label[k]}: {ratio[k]}]', end=' ')
                        else:
                            print(f'{num_to_label[k]}: {ratio[k]}', end=', ')
                    for idx, k in enumerate(ratio.keys()):
                        if idx==0:
                            print(f'augmented: {len(train_labels_bad)*len(augmentation)}[', end='')
                        if idx+1==len(ratio):
                            print(f'{num_to_label[k]}: {ratio[k]*len(augmentation)}]')
                        else:
                            print(f'{num_to_label[k]}: {ratio[k]*len(augmentation)}', end=', ')
        elif class_ is None:
            train_datasets = []
            train_datasets.append(train_dataset)
            for aug in augmentation:
                train_datasets += [Custom_dataset('train', np.array(train_imgs), np.array(train_labels), transforms=aug, toTensor=toTensor, normalize=normalize)]
            train_dataset = ConcatDataset(train_datasets)
            
    samples = []
    
    if weighted_loss and mode=='good':
        if num_to_label[0]=='good':
            samples.append((num_to_label[0], len(train_labels_good)*(augmented_ratio+1)))
            samples.append((num_to_label[1], len(train_labels_bad)*(len(augmentation)+1)))
        elif num_to_label[1]=='good':
            samples.append((num_to_label[0], len(train_labels_bad)*(len(augmentation)+1)))
            samples.append((num_to_label[1], len(train_labels_good)*(augmented_ratio+1)))
        
    if test_aug:
        if mode=='class':
            status5, loaded = load_pickle('./pickle/test_datasets_class.pickle')
        else:
            status5, loaded = load_pickle(f'./pickle/test_datasets_{class_}_{mode}.pickle')
        if status5==False:
            test_datasets = []
            test_datasets.append(test_dataset)
            
            test_datasets += [Custom_dataset('test', np.array(test_imgs), np.array(test_labels),
                                            transforms=[A_list[key] for key in A_list.keys() if key not in ['no_augmentation', '80CAR', '85CAR', '90CAR', '95CAR']], toTensor=toTensor, normalize=normalize)]
            test_dataset = ConcatDataset(test_datasets)
            if mode=='class':
                save_pickle(test_datasets, './pickle/test_datasets_class.pickle')
            else:
                save_pickle(test_datasets, f'./pickle/test_datasets_{class_}_{mode}.pickle')
        else:
            test_dataset = ConcatDataset(loaded)
                
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    
    return train_loader, test_loader, len(label_unique), num_to_label, samples