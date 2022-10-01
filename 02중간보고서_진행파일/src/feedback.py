import albumentations as A

import warnings
warnings.filterwarnings('ignore')

from glob import glob
import pandas as pd
import pickle
import numpy as np 
from tqdm import tqdm
import cv2

import os
import timm
import random

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
import time

import matplotlib.pyplot as plt

from IPython.display import display
from sklearn.metrics import confusion_matrix

device = torch.device('cuda')

train_png = sorted(glob('./open/train/*.png'))
test_png = sorted(glob('./open/test/*.png'))

train_y = pd.read_csv("./open/train_df.csv")

classes = sorted(np.unique(train_y["class"]))

A_list = {
    'HF': A.HorizontalFlip(p=1), 'VF': A.VerticalFlip(p=1), 'BF': [A.HorizontalFlip(p=1), A.VerticalFlip(p=1)],
    '90CAR': [A.CenterCrop(int(512*0.9), int(512*0.9), p=1), A.Resize(512, 512, p=1)],
    '95CAR': [A.CenterCrop(int(512*0.95), int(512*0.95), p=1), A.Resize(512, 512, p=1)],
    'R': A.Rotate(limit=360, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    'SH12': A.ShiftScaleRotate(shift_limit=0.12, scale_limit=0, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    'SC20': A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    'SCSH': A.ShiftScaleRotate(shift_limit=0.12, scale_limit=0.2, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    'SHR': A.ShiftScaleRotate(shift_limit=0.12, scale_limit=0, rotate_limit=360, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    'SCR': A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=360, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
    'SCSHR': A.ShiftScaleRotate(shift_limit=0.12, scale_limit=0.2, rotate_limit=360, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
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

def img_load(path, size):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, (size, size))
    return img

def load_pickle(path):
    output = None
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            output = pickle.load(fr)
    else:
        pass
    return output

def store_pickle(data, path):
    with open(path, 'wb') as fw:
        pickle.dump(data, fw)

def make_train_dataset(stage, mode, class_, size):
    path = f'./pickle/{stage}-train-dataset-{mode}-{class_}.pickle'
    load = load_pickle(path)
    if load is None:
        if stage=='1-stage':
            train_names = list(train_y['file_name'])
            train_labels = list(train_y['label'])
        elif stage=='2-stage':
            if mode=="class":
                train_names = list(train_y['file_name'])
                train_labels = list(train_y["class"])
            elif mode=='state':
                target = sorted([(f, path.split('\\')[-1]) for path, dir, file in os.walk(f'./open(original)/{class_}') if 'ground_truth' not in path.split('\\') for f in file if f.endswith('.png') and int(f.split('.')[0])<20000], key=lambda x:x[0])
                train_names = [img for img, _ in target]
                train_labels = [label for _, label in target]
        elif stage=='3-stage':
            if mode=="class":
                train_names = list(train_y['file_name'])
                train_labels = list(train_y["class"])
            elif mode=="good":
                train_names = pd.Series([name for name in list(train_y[train_y['class']==class_]['file_name'])])
                train_labels = pd.Series([state if state=='good' else 'bad' for state in list(train_y[train_y['class']==class_]["state"])])
            elif mode=="state":
                train_names = pd.Series([train_y.loc[index]['file_name'] for index in list(train_y[train_y['class']==class_].index) if train_y.loc[index]['state']!='good'])
                train_labels = pd.Series([state for state in train_y[train_y['class']==class_]["state"] if state!="good"])
        
        label_unique = sorted(np.unique(train_labels))
        label_to_num = {key:value for key,value in zip(label_unique, range(len(label_unique)))}
        num_to_label = {value:key for key,value in zip(label_unique, range(len(label_unique)))}
        
        train_labels = [label_to_num[k] for k in train_labels]
        train_imgs = [img_load(m, size) for m in train_png if m.split('\\')[-1] in train_names]
        
        num_classes = len(label_unique)
        
        dump = (train_imgs, train_labels, label_to_num, num_to_label, num_classes)
        
        store_pickle(dump, path)
    else:
        train_imgs, train_labels, label_to_num, num_to_label, num_classes = load
    return train_imgs, train_labels, label_to_num, num_to_label, num_classes

def make_test_dataset_public(size):
    path = './pickle/test-dataset-public.pickle'
    load = load_pickle(path)
    if load is None:
        test_imgs = [img_load(n, size) for n in test_png]
        store_pickle(test_imgs, path)
    else:
        test_imgs = load
    return test_imgs

def make_test_dataset_local(size):
    path = './pickle/test-dataset-local.pickle'
    load = load_pickle(path)
    if load is None:
        target = sorted([(f'{path}\\{f}', path.split('\\')[1], path.split('\\')[3]) for path, dir, file in os.walk('./open(original)') if 'ground_truth' not in path for f in file if f.endswith('.png') and int(f.split('.')[0])>=20000], key=lambda x:x[0])
        imgs = [img_load(m, size) for m in [x for x, _, _ in target]]
        labels = [f'{y}-{z}'for _,y,z in target]
        sets = (imgs, labels)
        store_pickle(sets, path)
    else:
        imgs, labels = load
    data_img = []
    data_img += imgs
    data_img += [A.OneOf([tf if type(tf)!=list else A.Compose(tf) for tf in [A_list[key] for key in A_list.keys()]], p=1)(image=img)['image'] for img in imgs]
    
    data_label = []
    data_label += labels
    data_label += labels
        
    return data_img, data_label

def make_test_dataset_custom(stage, mode, class_, size):
    path = f'./pickle/{stage}-test-dataset-{mode}-{class_}.pickle'
    load = load_pickle(path)
    criteria = '\\'
    if load is None:
        if stage=='1-stage':
            target = sorted([(f, f"{path.split(criteria)[1]}-{path.split(criteria)[-1]}") for path, dir, file in os.walk('./open(original)') if 'ground_truth' not in path.split('\\') for f in file if f.endswith('.png') and int(f.split('.')[0])>=20000], key=lambda x:x[0])
            test_names = [img for img, _ in target]
            test_labels = [label for _, label in target]
        elif stage=='2-stage':
            if mode=='class':
                target = sorted([(f, path.split(criteria)[1]) for path, dir, file in os.walk('./open(original)') if 'ground_truth' not in path.split('\\') for f in file if f.endswith('.png') and int(f.split('.')[0])>=20000], key=lambda x:x[0])
                test_names = [img for img, _ in target]
                test_labels = [label for _, label in target]
            elif mode=='state':
                target = sorted([(f, path.split(criteria)[-1]) for path, dir, file in os.walk(f'./open(original)/{class_}') if 'ground_truth' not in path.split('\\') for f in file if f.endswith('.png') and int(f.split('.')[0])>=20000], key=lambda x:x[0])
                test_names = [img for img, _ in target]
                test_labels = [label for _, label in target]
        elif stage=='3-stage':
            if mode=='class':
                target = sorted([(f, path.split(criteria)[1]) for path, dir, file in os.walk('./open(original)') if 'ground_truth' not in path.split('\\') for f in file if f.endswith('.png') and int(f.split('.')[0])>=20000], key=lambda x:x[0])
                test_names = [img for img, _ in target]
                test_labels = [label for _, label in target]
            elif mode=='good':
                target = sorted([(f, 'good') if path.split(criteria)[-1]=='good' else (f, 'bad') for path, dir, file in os.walk(f'./open(original)/{class_}') if 'ground_truth' not in path.split('\\') for f in file if f.endswith('.png') and int(f.split('.')[0])>=20000], key=lambda x:x[0])
                test_names = [img for img, _ in target]
                test_labels = [label for _, label in target]
            elif mode=='state':
                target = sorted([(f, path.split(criteria)[-1]) for path, dir, file in os.walk(f'./open(original)/{class_}') if 'ground_truth' not in path.split('\\') for f in file if f.endswith('.png') and int(f.split('.')[0])>=20000 and path.split('\\')[-1]!='good'], key=lambda x:x[0])
                test_names = [img for img, _ in target]
                test_labels = [label for _, label in target]
            
        label_unique = sorted(np.unique(test_labels))
        label_to_num = {key:value for key,value in zip(label_unique, range(len(label_unique)))}
        num_to_label = {value:key for key,value in zip(label_unique, range(len(label_unique)))}
        
        test_labels = [label_to_num[k] for k in test_labels]
        
        test_imgs = [img_load(m, size) for m in test_png if m.split(criteria)[-1] in test_names]
        
        dump = (test_imgs, test_labels, label_to_num, num_to_label)
        
        store_pickle(dump, path)
    else:
        test_imgs, test_labels, label_to_num, num_to_label = load
    
    return test_imgs, test_labels, label_to_num, num_to_label

class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, transforms=None, mode='train', normalize=(0, 1)):
        self.img_paths = img_paths
        self.labels = labels
        self.mode=mode
        self.transforms = transforms
        self.normalize = normalize
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = self.img_paths[idx]
        if self.transforms is not None:
            if type(self.transforms)==list:
                self.transforms = A.Compose(self.transforms)
            img = self.transforms(image=img)['image']
        img = transforms.ToTensor()(img)
        if self.normalize[0] is not None:
            m, v = self.normalize
            tf = transforms.Normalize((m, m, m), (v, v, v))
            img = tf(img)
        if self.mode=='test':
            pass
        
        label = self.labels[idx]
        return img, label
    
class Network(nn.Module):
    def __init__(self, num_classes):
        super(Network, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x

def make_dataloader(stage, mode, class_, batch_size, size, num_aug, normalize, train_aug):
    train_imgs, train_labels, _, num_to_label_tr, num_classes = make_train_dataset(stage, mode, class_, size)
    test_imgs, test_labels, _, num_to_label_te = make_test_dataset_custom(stage, mode, class_, size)
    
    train_bad_imgs = []
    train_bad_labels = []
    
    train_dataset = Custom_dataset(np.array(train_imgs), np.array(train_labels), mode='train', normalize=normalize)
    
    if train_aug:
        train_datasets = []
        train_datasets += [train_dataset]
        if stage=='1-stage':
            train_datasets += [Custom_dataset(np.array(train_imgs), np.array(train_labels), transforms=A.OneOf([tf if type(tf)!=list else A.Compose(tf) for tf in [A_list[key] for key in A_list.keys()]], p=1), mode='train', normalize=normalize)]
        elif stage=='2-stage':
            if mode=='class':
                train_datasets += [Custom_dataset(np.array(train_imgs), np.array(train_labels), transforms=A.OneOf([tf if type(tf)!=list else A.Compose(tf) for tf in [A_list[key] for key in A_list.keys()]], p=1), mode='train', normalize=normalize)]
            elif mode=='good':
                for _ in range(num_aug):
                    train_bad_imgs = [i for i, l in zip(train_imgs, train_labels) if num_to_label_tr[l]!='good']
                    train_bad_labels = [l for i, l in zip(train_imgs, train_labels) if num_to_label_tr[l]!='good']
                    for aug in [tf if type(tf)!=list else A.Compose(tf) for tf in [A_list[key] for key in A_list.keys()]]:
                        train_datasets += [Custom_dataset(np.array(train_bad_imgs), np.array(train_bad_labels), transforms=aug, mode='train', normalize=normalize)]
                    # train_datasets += [Custom_dataset(np.array(train_bad_imgs), np.array(train_bad_labels), transforms=A.OneOf([tf if type(tf)!=list else A.Compose(tf) for tf in [A_list[key] for key in A_list.keys() if key not in ['no_augmentation']]], p=1), mode='train')]
        elif stage=='3-stgae':
            pass
        train_dataset = ConcatDataset(train_datasets)
        
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    
    test_datasets = []
    test_datasets += [Custom_dataset(np.array(test_imgs), np.array(test_labels), mode='test', normalize=normalize)]
    test_datasets += [Custom_dataset(np.array(test_imgs), np.array(test_labels), transforms=A.OneOf([tf if type(tf)!=list else A.Compose(tf) for tf in [A_list[key] for key in A_list.keys()]], p=1), mode='test', normalize=normalize)]
    
    test_dataset = ConcatDataset(test_datasets)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    
    # for idx, data in enumerate(train_dataset):
    #     i, l = data
    #     if idx%8==0:
    #         plt.figure(figsize=(32,16))
    #     plt.subplot(1, 8, (idx%8)+1)
    #     plt.axis('off')
    #     plt.title(num_to_label_te[l])
    #     plt.imshow(transforms.ToPILImage()(i))
    #     if idx%8==7:
    #         plt.show()
    # plt.show()
    
    print(f'\033[46mMode: {mode}, Class: {class_}, Train Dataset: {len(train_labels)}+{len(train_dataset)-len(train_labels)}(={len(train_bad_imgs)*num_aug}), Test Dataset: {len(test_labels)}+{len(test_dataset)-len(test_labels)}\033[0m')
    
    return train_loader, test_loader, num_classes, num_to_label_tr

def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score

def trainer(epochs, train_loader, test_loader, num_classes, lr=1e-4, show_cf_matrix=False, num_to_label=None, class_=None):
    model = Network(num_classes).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    best_accuracy = 0

    for epoch in range(epochs):
        train_start=time.time()
        train_loss = 0
        train_pred=[]
        train_y=[]
        model.train()
        for batch in (train_loader):
            optimizer.zero_grad()
            x = torch.tensor(batch[0], dtype=torch.float32, device=device)
            y = torch.tensor(batch[1], dtype=torch.long, device=device)
            with torch.cuda.amp.autocast():
                pred = model(x)
            loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()/len(train_loader)
            train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
            train_y += y.detach().cpu().numpy().tolist()
        
        train_f1 = score_function(train_y, train_pred)

        Train_time = time.time() - train_start
        
        test_start = time.time()
        test_loss = 0
        test_pred = []
        test_y = []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x = torch.tensor(batch[0], dtype = torch.float32, device = device)
                y = torch.tensor(batch[1], dtype=torch.long, device=device)
                with torch.cuda.amp.autocast():
                    pred = model(x)
                loss = criterion(pred, y)
                
                test_loss += loss.item()/len(test_loader)
                test_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                test_y += y.detach().cpu().numpy().tolist()
                
            test_f1 = score_function(test_y, test_pred)
            
            Test_time = time.time() - test_start
            
            if test_f1 > best_accuracy:
                best_stats = (train_loss, train_f1, test_loss, test_f1, test_pred, test_y)
                best_accuracy = test_f1
                torch.save(model, './pt/model.pt')
            print(f'Epoch: {epoch+1}/{epochs}, Train loss: {train_loss:.5f}, Train f1: {train_f1:.5f}, Train time: {Train_time:.0f}s/{Train_time*(epochs-epoch-1):.0f}s, Test loss: {test_loss:.5f}, Test f1: {test_f1:.5f}, Test time: {Test_time:.0f}s')
    
    if test_f1 != best_accuracy:
        train_loss, train_f1, test_loss, test_f1, test_pred, test_y = best_stats
        model = torch.load('./pt/model.pt')
        statement = "\033[41mLast Epoch is not the best weight\033[0m"
        print(statement)
        print(f'Train loss: {train_loss:.5f}, Train f1 : {train_f1:.3f}, Test loss : {test_loss:.5f}, \033[44mTest f1 : {test_f1:.2f}, Best: {best_accuracy:.2f}\033[0m')
        
    if show_cf_matrix:
        pd.options.display.float_format = '{:.3f}'.format
        cf = confusion_matrix(test_y, test_pred)
        df = pd.DataFrame(cf)
        df.rename(columns={col:f'predict:{num_to_label[col]}' for col in list(df.columns)}, inplace=True)
        df.rename(index={row:f'target:{num_to_label[row]}' for row in list(df.index)}, inplace=True)
        df.columns.name = class_
        
        df['precision'] = [0 for i in range(len(list(df.index)))]
        df['recall'] = [0 for i in range(len(list(df.index)))]
        
        for idx, col in enumerate(list(df.index)):
            df.loc[col, 'precision'] = df.iloc[idx, idx]/df.iloc[0:len(df.index), idx].sum()
            df.loc[col, 'recall'] = df.iloc[idx, idx]/df.iloc[idx, 0:len(df.index)].sum()
        df.fillna(0)
        display(df)
        
    return model
 
def helper(stage, mode, class_, batch_size, size, epochs, num_aug, lr, normalize, show_cf_matrix=False, train_aug=True):
    train_loader, test_loader, num_classes, num_to_label = make_dataloader(stage, mode, class_, batch_size, size, num_aug, normalize, train_aug)
    model = trainer(epochs, train_loader, test_loader, num_classes, lr=lr, show_cf_matrix=show_cf_matrix, num_to_label=num_to_label, class_=class_)
    return model, num_to_label

def tester(stage, models, size, mode, file_name, show_cf_matrix):
    answers = []
    
    if mode=='submission':
        test_data = make_test_dataset_public(size)
    elif mode=='practice':
        test_data, test_label = make_test_dataset_local(size)
        
    class_total = []
    state_total = []
    
    class_model, class_num_to_label = models['class']
    for idx, img in enumerate(test_data):
        img = np.transpose(img, (2, 0, 1))
        class_model.to(device)
        class_model.eval()
        img = torch.tensor(img, dtype=torch.float32, device=device)
        class_pred = class_model(torch.unsqueeze(img, 0))
        class_pred = class_pred.argmax(1).detach().cpu().numpy().tolist()[0]
        class_total.append(class_num_to_label[class_pred])
        answers.append(f'{class_num_to_label[class_pred]}')
        
        # state_model, state_num_to_label = models['state'][class_num_to_label[class_pred]]
        # state_model.to(device)
        # state_model.eval()
        # state_pred = state_model(torch.unsqueeze(img, 0))
        # state_pred = state_pred.argmax(1).detach().cpu().numpy().tolist()[0]
        # state_total.append(state_num_to_label[state_pred])
        
        # answers.append(f'{class_num_to_label[class_pred]}-{state_num_to_label[state_pred]}')
        
        # if mode=='practice':
        #     if idx<5:
        #         img = transforms.ToPILImage()(img)
        #         plt.title(f'Predict: {class_num_to_label[class_pred]}-{state_num_to_label[state_pred]}, Answer: {test_label[idx]}', fontsize=15)
        #         plt.imshow(img)
        #         plt.show()
    if mode=='practice':
        print(score_function(test_label, answers))
    if mode=='practice' and show_cf_matrix:
        print("*"*70,"Result","*"*70)
        pd.options.display.float_format = '{:.3f}'.format
        if stage=='1-stage':
            cf = confusion_matrix(answers, test_label)
            df = pd.DataFrame(cf)
            df.rename(columns={col:f'predict:{class_num_to_label[col]}' for col in list(df.columns)}, inplace=True)
            df.rename(index={row:f'target:{class_num_to_label[row]}' for row in list(df.index)}, inplace=True)
            df['precision'] = [0 for i in range(len(list(df.index)))]
            df['recall'] = [0 for i in range(len(list(df.index)))]
            
            for idx, col in enumerate(list(df.index)):
                df.loc[col, 'precision'] = df.iloc[idx, idx]/df.iloc[0:len(df.index), idx].sum()
                df.loc[col, 'recall'] = df.iloc[idx, idx]/df.iloc[idx, 0:len(df.index)].sum()
            df.fillna(0)
            display(df)
        elif stage=='2-stage':
            class_answer = [t.split('-')[0] for t in test_label]
            class_total = [a.split('-')[0] for a in answers]
            
            cf = confusion_matrix(class_answer, class_total)
            df = pd.DataFrame(cf)
            df.rename(columns={col:f'predict:{class_num_to_label[col]}' for col in list(df.columns)}, inplace=True)
            df.rename(index={row:f'target:{class_num_to_label[row]}' for row in list(df.index)}, inplace=True)
            df['precision'] = [0 for i in range(len(list(df.index)))]
            df['recall'] = [0 for i in range(len(list(df.index)))]
            
            for idx, col in enumerate(list(df.index)):
                df.loc[col, 'precision'] = df.iloc[idx, idx]/df.iloc[0:len(df.index), idx].sum()
                df.loc[col, 'recall'] = df.iloc[idx, idx]/df.iloc[idx, 0:len(df.index)].sum()
            df.fillna(0)
            display(df)
        
        # for class_ in classes:
        #     states = [(t.split('-')[1], a.split('-')[1]) for t, a in zip(test_label, answers) if t.split('-')[0]==class_ and a.split('-')[0]==class_]
        #     state_answer = list(map(lambda x:x[0], states))
        #     state_total = list(map(lambda x:x[1], states))
            
        #     cf = confusion_matrix(state_answer, state_total)
        #     df = pd.DataFrame(cf)
        #     df.rename(columns={col:f"predict:{models['state'][class_][1][col]}" for col in list(df.columns)}, inplace=True)
        #     df.rename(index={row:f"target:{models['state'][class_][1][row]}" for row in list(df.index)}, inplace=True)
        #     df.columns.name = class_
        #     df['precision'] = [0 for i in range(len(list(df.index)))]
        #     df['recall'] = [0 for i in range(len(list(df.index)))]
            
        #     for idx, col in enumerate(list(df.index)):
        #         df.loc[col, 'precision'] = df.iloc[idx, idx]/df.iloc[0:len(df.index), idx].sum()
        #         df.loc[col, 'recall'] = df.iloc[idx, idx]/df.iloc[idx, 0:len(df.index)].sum()
        #     df.fillna(0)
        #     display(df)
        
        # print(f"Class: {score_function(class_total, [y.split('-')[0] for y in test_label]):.4f}, State: {score_function(state_total, [y.split('-')[1] for y in test_label]):.4f}, Total: {score_function(test_label, answers):.4f}")
        # with open(f"./txt/{file_name}-class.txt", "w") as f:
        #     for a, c in zip(answers, class_total):
        #         f.write(f"{a.split('-')[0]:<15} {c}")
        # with open(f"./txt/{file_name}-state.txt", "w") as f:
        #     for a, s in zip(answers, state_total):
        #         f.write(f"{a.split('-')[1]:<15} {s}")
        
        return class_total, state_total, answers, test_label
    elif mode=='practice':
        return class_total, state_total, answers, test_label
    elif mode=='submission':
        return answers

def submission(answers, file_name):
    sub = pd.read_csv('./open/sample_submission.csv', index_col=0)
    sub['label'] = answers
    sub.to_csv(f'./open/{file_name}.csv')