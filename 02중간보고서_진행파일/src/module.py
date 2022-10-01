import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import warnings

from IPython.display import display
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset

from src.variable import device, criterion_list, scaler_list, score_function

warnings.filterwarnings('ignore')

toTensor_tf = transforms.ToTensor()

class Custom_dataset(Dataset):
    def __init__(self, mode, img_paths, labels, transforms=None, toTensor=True, normalize=[(0.3, 0.3, 0.3), (0.3, 0.3, 0.3)], mask_paths=None):
        self.mode = mode
        self.img_paths = img_paths
        self.labels = labels
        self.transforms=transforms
        self.toTensor = toTensor
        self.normalize = normalize
        if mask_paths is not None:
            self.mask_on = True
            self.mask_paths = mask_paths
        else:
            self.mask_on = False
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = self.img_paths[idx]
        if self.transforms is not None:
            if type(self.transforms)==list:
                if self.mode=='train':
                    self.transforms = A.Compose(self.transforms)
                elif self.mode=='test':
                    self.transforms = A.OneOf([tf if type(tf)!=list else A.Compose(tf) for tf in self.transforms], p=1)
            img = self.transforms(image=img)['image']
        if self.toTensor:
            img = toTensor_tf(img)
        if self.normalize is not None:
            normalize_tf = transforms.Normalize(self.normalize[0], self.normalize[1])
            img = normalize_tf(img)
        
        if self.mask_on:
            mask = self.mask_paths[idx]
            if self.transforms is not None:
                if type(self.transforms)==list:
                    if self.mode=='train':
                        self.transforms = A.Compose(self.transforms)
                    elif self.mode=='test':
                        self.transforms = A.OneOf([tf if type(tf)!=list else A.Compose(tf) for tf in self.transforms], p=1)
                mask = self.transforms(image=mask)['image']
            if self.toTensor:
                mask = toTensor_tf(mask)
            if self.normalize is not None:
                normalize_tf = transforms.Normalize(self.normalize[0], self.normalize[1])
                mask = normalize_tf(mask)
        
        label = self.labels[idx]
        return img, label
    
class Network(nn.Module):
    def __init__(self, num_classes, mask_on=False):
        super(Network, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        
        self.mask_on = mask_on
        self.added = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.added.bias.data = torch.ones(self.added.bias.data.shape)
        self.added.weight.data = torch.zeros(self.added.weight.data.shape)
        
        self.models = {}
        for idx, layer in enumerate(self.model.children()):
            self.models[idx+1] = layer
            
    def forward(self, x, y=None):
        if self.mask_on:
            for key in self.models.keys():
                if key!=3:
                    x = self.models[key](x)
                else:
                    x = self.models[key](x) + self.added(y)
        else:
            x = self.model(x)
        return x
    
class MyModel():
    def __init__(self, class_, train_loader, test_loader, num_to_label):
        self.class_ = class_ if class_ is not None else 'None'
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_to_label = num_to_label
    
    def set_model(self, model=None, num_classes=None, mask_on=False):
        if model is None:
            self.model = Network(num_classes, mask_on=mask_on)
        else:
            self.model = model
        self.model.to(device)
        return self.model

    def train(self, epochs=10,
              criterion='CE', scaler='GS', optimizer='default',
              lr=1e-4, verbose=True, show_predict=False, show_cf_matrix=False, visualize=False,
              return_score=False, print_last=False,
              weighted_loss=False, samples=[], bad_strength=2, good_penalty=0.5,
              mask_on=False, mask_data=None):
        if weighted_loss and len(samples)!=0 and self.class_!='None':
            sample = [s[1] for s in samples]
            normed_weight = [1 - (x/sum(sample)) for x in sample]
            normed_weight = [n*good_penalty if s[0]=='good' else n*bad_strength for s,n in zip(samples, normed_weight)]
            normed_weights = torch.HalfTensor(normed_weight).to(device)
            criterion = nn.CrossEntropyLoss(normed_weights)
        else:
            criterion = criterion_list[criterion]
        scaler = scaler_list[scaler]
        best_accuracy = 0
        if optimizer=='default':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            start=time.time()
            train_loss = 0
            train_pred=[]
            train_y=[]
            self.model.train()
            for batch in self.train_loader:
                optimizer.zero_grad()
                x = torch.tensor(batch[0], dtype=torch.float32, device=device)
                y = torch.tensor(batch[1], dtype=torch.long, device=device)
                with torch.cuda.amp.autocast():
                    pred = self.model(x)
                loss = criterion(pred, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()/len(self.train_loader)
                train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                train_y += y.detach().cpu().numpy().tolist()
            
            train_f1 = score_function(train_y, train_pred)
            train_time = time.time() - start
            
            start=time.time()
            test_loss = 0
            test_pred=[]
            test_y=[]
            self.model.eval()
            with torch.no_grad():
                for batch in self.test_loader:
                    x = torch.tensor(batch[0], dtype=torch.float32, device=device)
                    y = torch.tensor(batch[1], dtype=torch.long, device=device)
                    with torch.cuda.amp.autocast():
                        pred = self.model(x)
                    loss = criterion(pred, y)
                    
                    test_loss += loss.item()/len(self.test_loader)
                    test_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                    test_y += y.detach().cpu().numpy().tolist()
                
            test_f1 = score_function(test_y, test_pred)
            test_time = time.time() - start
            
            if test_f1 > best_accuracy:
                best_stats = (train_loss, train_f1, test_loss, test_f1, test_pred, test_y)
                best_accuracy = test_f1
                torch.save(self.model, './pt/model.pt')
            
            if verbose:
                print(f'Class: {self.class_}, Epoch: {epoch+1}/{epochs}, Time: {train_time:.0f}s/{train_time*(epochs-epoch-1):.0f}s, Train loss: {train_loss:.5f}, Train f1 : {train_f1:.3f}, Test loss : {test_loss:.5f}, \033[44mTest f1 : {test_f1:.2f}\033[0m, Test time : {test_time:.2f}s')
                
        if test_f1 != best_accuracy:
            train_loss, train_f1, test_loss, test_f1, test_pred, test_y = best_stats
            self.model = torch.load('./pt/model.pt')
            statement = "\033[41mLast Epoch is not the best weight\033[0m"
            
        if print_last:
            if test_f1 != best_accuracy:
                print(statement)
            print(f'Class: {self.class_}, Train loss: {train_loss:.5f}, Train f1 : {train_f1:.3f}, Time: {train_time*epochs:.0f}s, Test loss : {test_loss:.5f}, \033[44mTest f1 : {test_f1:.2f}, Best: {best_accuracy:.2f}\033[0m, Test time : {test_time:.2f}s')
                
        if show_predict:
            real_label = [self.num_to_label[r] for r in test_y]
            predict_label = [self.num_to_label[p] for p in test_pred]
            predict = pd.DataFrame(columns=['real', 'predict'], index=range(1, len(test_y)+1))
            predict.columns.name = self.class_
            predict['real'] = real_label
            predict['predict'] = predict_label
            display(predict)
            
        if show_cf_matrix:
            pd.options.display.float_format = '{:.3f}'.format
            cf = confusion_matrix(test_y, test_pred)
            df = pd.DataFrame(cf)
            df.rename(columns={col:f'predict:{self.num_to_label[col]}' for col in list(df.columns)}, inplace=True)
            df.rename(index={row:f'target:{self.num_to_label[row]}' for row in list(df.index)}, inplace=True)
            df.columns.name = self.class_
            
            df['precision'] = [0 for i in range(len(list(df.index)))]
            df['recall'] = [0 for i in range(len(list(df.index)))]
            
            for idx, col in enumerate(list(df.index)):
                df.loc[col, 'precision'] = df.iloc[idx, idx]/df.iloc[0:len(df.index), idx].sum()
                df.loc[col, 'recall'] = df.iloc[idx, idx]/df.iloc[idx, 0:len(df.index)].sum()
            df.fillna(0)
            display(df)
        
        if visualize:
            row = 6
            targets = y.detach().cpu().numpy().tolist()
            predict = pred.argmax(1).detach().cpu().numpy().tolist()
            for idx, (t, p) in enumerate(zip(targets, predict)):
                toPIL = transforms.ToPILImage()
                image = toPIL(x[idx])
                if (idx+1)%row==1:
                    plt.figure(figsize=(28,32))
                plt.subplot(1, row, (idx%row)+1)
                plt.axis('off')
                plt.title(f'{self.class_}, real:{self.num_to_label[t]}, predict:{self.num_to_label[p]}')
                plt.imshow(image)
                if (idx+1)%row==0:
                        plt.show()
            plt.show()
        
        self.model = torch.load('./pt/model.pt')
        if return_score:
            return test_f1
        
    def test(self, data_class, data_good, data_state, model_class, model_good, model_state):
        test_pred=[]
        test_y=[]
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                x = torch.tensor(batch[0], dtype=torch.float32, device=device)
                y = torch.tensor(batch[1], dtype=torch.long, device=device)
                with torch.cuda.amp.autocast():
                    pred = self.model(x)
                test_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                test_y += y.detach().cpu().numpy().tolist()
        test_f1 = score_function(test_y, test_pred)