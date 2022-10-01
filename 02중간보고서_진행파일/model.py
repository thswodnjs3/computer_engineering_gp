import albumentations as A
import cv2
import numpy as np
import os
import pickle
import pandas as pd
import random
import time

from collections import Counter
from glob import glob
from tqdm import tqdm

import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

device = torch.device('cuda')

csv = pd.read_csv('./open/train_df.csv', index_col=0)

classes = [class_ for class_ in sorted(os.listdir("./open(original)")) if os.path.isdir("./open(original)"+'/'+class_)]

states = {}
train_paths = []
test_paths = []
for class_ in classes:
    states[class_] = []
    for criteria in ['train', 'test']:
        criteria_path = "./open(original)"+'/'+class_+'/'+criteria
        for state in sorted(os.listdir(criteria_path)):
            states[class_].append(state)
            state_path = criteria_path+'/'+state
            for image in sorted(os.listdir(state_path)):
                image_path = state_path+'/'+image
                if int(image.split('.')[0])<20000:
                    train_paths.append(image_path)
                else:
                    test_paths.append(image_path)
        states[class_] = list(set(states[class_]))

def img_load(path, size):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, size)
    return img

def make_img_data(label, criteria, paths=None, size=None, fix=None, verbose_dataset_making=True):
    if size is None:
        if label=='class':
            if os.path.isfile("./pickle/"+criteria+"_imgs("+label+").pickle"):
                with open("./pickle/"+criteria+"_imgs("+label+").pickle", "rb") as fr:
                    imgs = pickle.load(fr)
            else:
                pngs = sorted(paths)
                if verbose_dataset_making:
                    imgs = [img_load(m, size) for m in tqdm(pngs)]
                else:
                    imgs = [img_load(m, size) for m in pngs]
                with open("./pickle/"+criteria+"_imgs("+label+").pickle", "wb") as fw:
                    pickle.dump(imgs, fw)
        elif label=='state':
            if os.path.isfile(f"./pickle/{label}-{fix}-imgs-{criteria}.pickle"):
                with open(f"./pickle/{label}-{fix}-imgs-{criteria}.pickle", "rb") as fr:
                    imgs = pickle.load(fr)
            else:
                pngs = sorted([x for x in paths if fix in x])
                if verbose_dataset_making:
                    imgs = [img_load(m, size) for m in tqdm(pngs)]
                else:
                    imgs = [img_load(m, size) for m in pngs]
                with open(f"./pickle/{label}-{fix}-imgs-{criteria}.pickle", "wb") as fw:
                    pickle.dump(imgs, fw)
    else:
        if os.path.isfile(f"./pickle/size_{str(size[0])}-{label}-{fix}-imgs-{criteria}.pickle"):
            with open(f"./pickle/size_{str(size[0])}-{label}-{fix}-imgs-{criteria}.pickle", "rb") as fr:
                imgs = pickle.load(fr)
        else:
            pngs = sorted([x for x in paths if fix in x])
            if verbose_dataset_making:
                imgs = [img_load(m, size) for m in tqdm(pngs)]
            else:
                imgs = [img_load(m, size) for m in pngs]
            with open(f"./pickle/size_{str(size[0])}-{label}-{fix}-imgs-{criteria}.pickle", "wb") as fw:
                pickle.dump(imgs, fw)
                
    return imgs

def augmenting(augmentation, imgs, labels, value_to_key, multiply_none, multiply, rules):
    setting = []
    
    transform_list = [[x for x in set_] for set_ in augmentation]
    
    imgs_good = [img for img, state in zip(imgs, labels) if value_to_key[state]=="good"]
    labels_good = [state for state in labels if value_to_key[state]=="good"]
    
    original_good = len(imgs_good)
    
    setting += [Custom_dataset(imgs_good, labels_good, transforms=None, mode='train')]
    if multiply_none:
        setting += [Custom_dataset(imgs_good, labels_good, transforms=aug, mode='train') for set_ in transform_list for aug in set_]
    else:
        setting += [Custom_dataset(imgs_good, labels_good, transforms=aug, mode='train') for aug in [random.choice(random.choice(transform_list))] for i in range(multiply-1)]
    
    augmenting_good = len(ConcatDataset(setting)) - original_good
    
    imgs_other = [img for img, state in zip(imgs, labels) if value_to_key[state]!="good"]
    labels_other = [state for state in labels if value_to_key[state]!="good"]
    
    original_other = len(imgs_other)
    
    setting += [Custom_dataset(imgs_other, labels_other, transforms=None, mode='train')]
    if rules is not None:
        for rule in rules.keys():
            imgs_rule = [img for img, state in zip(imgs, labels) if value_to_key[state]!="good" and value_to_key[state]==rule]
            labels_rule = [state for state in labels if value_to_key[state]!="good" and value_to_key[state]==rule]
            setting += [Custom_dataset(imgs_rule, labels_rule, transforms=aug, mode='train') for set_ in transform_list for aug in set_ if type(aug) not in rules[rule]]
        imgs_other = []
        labels_other = []
        for rule in rules.keys():
            imgs_other += [img for img, state in zip(imgs, labels) if value_to_key[state]!="good" and value_to_key[state]!=rule]
            labels_other += [state for state in labels if value_to_key[state]!="good" and value_to_key[state]!=rule]
            
    setting += [Custom_dataset(imgs_other, labels_other, transforms=aug, mode='train') for set_ in transform_list for aug in set_]
    
    augmenting_other = len(ConcatDataset(setting)) - original_other - augmenting_good - original_good
    
    return setting, original_good, augmenting_good, original_other, augmenting_other
            
def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score

class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, transforms=None, mode='train'):
        self.img_paths = img_paths
        self.labels = labels
        self.mode=mode
        self.transforms=transforms
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = self.img_paths[idx]
        if self.mode=='train':
            if self.transforms is not None:
                if type(self.transforms)==list:
                    self.transforms = A.Compose(self.transforms)
                img = self.transforms(image=img)['image']
        img = transforms.ToTensor()(img)
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

class myModel():
    def __init__(self):
        self.classes = classes
        
        self.states = states
        self.train_paths = train_paths
        self.test_paths = test_paths
        
    def make_dataset(self, label, fix=None, size=None, batch_size=32, augmentation=None, multiply_none=False, multiply=2, exceptions=[],
                     verbose_dataset_making=True):
        if len(exceptions)!=0:
            rule = {}
            for exception in exceptions:
                rule[exception[0]] = exception[1:]
        else:
            rule = None
            
        self.label_name = label
        self.fix_name = fix
        self.augmenting = False if augmentation is None else True
        
        if label=='class':
            self.label = sorted(self.classes)
        elif label=='state':
            self.label = sorted(self.states[fix])
        
        label_value_to_key = {value:key for key,value in zip(self.label, range(len(self.label)))}
        label_key_to_value = {key:value for key,value in zip(self.label, range(len(self.label)))}
        
        if label=='class':
            train_imgs = make_img_data('class','train',paths=self.train_paths,size=size,verbose_dataset_making=verbose_dataset_making)
            train_labels = [label_key_to_value[x.split('/')[2]] for x in self.train_paths]
            test_imgs = make_img_data('class', 'test',paths=self.test_paths,size=size,verbose_dataset_making=verbose_dataset_making)
            test_labels = [label_key_to_value[x.split('/')[2]] for x in self.test_paths]
        elif label=='state':
            train_imgs = make_img_data('state','train',paths=self.train_paths,size=size,fix=fix)
            train_labels = [label_key_to_value[x.split('/')[4]] for x in [x for x in self.train_paths if fix in x]]
            test_imgs = make_img_data('state', 'test',paths=self.test_paths,size=size,fix=fix)
            test_labels = [label_key_to_value[x.split('/')[4]] for x in [x for x in self.test_paths if fix in x]]
        
        if label=="state" and augmentation is not None:
            dataset, self.original_good, self.augmenting_good, self.original_bad, self.augmenting_bad = \
                augmenting(augmentation, train_imgs, train_labels, label_value_to_key, multiply_none, multiply, rule)
            self.train_dataset = ConcatDataset(dataset)
        else:
            self.train_dataset = Custom_dataset(train_imgs, train_labels, mode='train')
        self.test_dataset = Custom_dataset(test_imgs, test_labels, mode='test')
            
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size)
        self.test_loader = DataLoader(self.test_dataset, shuffle=True, batch_size=batch_size)
        
    def make_model(self, optimizer='default', criterion='default', scaler='default'):
        self.model = Network(num_classes=len(set(self.label))).to(device)
        
        if optimizer=='default':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        if criterion=='default':
            self.criterion = nn.CrossEntropyLoss()
        if scaler=='default':
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.train_count=0
        self.test_count=0
        
    def train(self, epochs=10, verbose=True):
        for epoch in range(epochs):
            self.train_count=0
            start=time.time()
            train_loss = 0
            train_pred=[]
            train_y=[]
            self.model.train()
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                x = torch.tensor(batch[0], dtype=torch.float32, device=device)
                y = torch.tensor(batch[1], dtype=torch.long, device=device)
                self.train_count+=x.shape[0]
                with torch.cuda.amp.autocast():
                    pred = self.model(x)
                loss = self.criterion(pred, y)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                train_loss += loss.item()/len(self.train_loader)
                train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                train_y += y.detach().cpu().numpy().tolist()
            
            train_f1 = score_function(train_y, train_pred)

            TIME = time.time() - start
            if verbose:
                print(f'epoch : {epoch+1}/{epochs}    time : {TIME:.0f}s/{TIME*(epochs-epoch-1):.0f}s')
                print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}')
            
    def eval(self, verbose=True):
        test_loss = 0
        test_pred=[]
        test_y=[]
        self.model.eval()
        with torch.no_grad():
            start=time.time()
            for batch in self.test_loader:
                x = torch.tensor(batch[0], dtype=torch.float32, device=device)
                y = torch.tensor(batch[1], dtype=torch.long, device=device)
                self.test_count+=x.shape[0]
                with torch.cuda.amp.autocast():
                    pred = self.model(x)
                loss = self.criterion(pred, y)
                
                test_loss += loss.item()/len(self.test_loader)
                test_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                test_y += y.detach().cpu().numpy().tolist()
            
            test_f1 = score_function(test_y, test_pred)

            TIME = time.time() - start
            
            if verbose:
                if self.augmenting:
                    print(f'class : {self.fix_name:<12}    TEST    loss : {test_loss:.2f}    f1 : {test_f1:.2f}    time : {TIME:.2f}s    num good : {self.original_good}    num bad : {self.original_bad}    aug good : {self.augmenting_good}    aug bad : {self.augmenting_bad}')
                else:
                    print(f'class : {self.fix_name:<12}    TEST    loss : {test_loss:.2f}    f1 : {test_f1:.2f}    time : {TIME:.2f}s    num_train : {self.train_count}    num_test : {self.test_count}')
        return test_f1
    
    def save_weight(self, name=None):
        if name is None:
            if self.label=="class":
                torch.save(self.model.state_dict(), "./weight/"+self.label+".pt")
            elif self.label=="state":
                torch.save(self.model.state_dict(), "./weight/"+self.fix+"-"+self.label+".pt")
        else:
            torch.save(self.model.state_dict(), "./weight/"+name+".pt")
        
    def load_weight(self, model=None, name=None):
        if name is None:
            if self.label_name=="class":
                name = self.label_name
            elif self.label_name=="state":
                name = self.fix_name+'-'+self.label_name
        
        if model is None:
            self.model.load_state_dict(torch.load("./weight/"+name+".pt"))
        else:
            self.model = model
            self.model.load_state_dict(torch.load("./weight/"+name+".pt"))
    
    def return_model(self):
        return self.model

def training(resize=(128,128), batch_size=32, train_on=True, verbose_dataset_making=True):
    scores = []
    
    model = myModel()
    for c in sorted(np.unique(csv['class'])):
        model.make_dataset(label="state",fix=c,size=resize, batch_size=batch_size, verbose_dataset_making=verbose_dataset_making)
        model.make_model()
        model.train(verbose=False)
        model.save_weight(name=c+'-'+'state_no_aug')
        model.load_weight(name=c+'-'+'state_no_aug')
        score = model.eval()
        
        scores.append(score)
    return scores