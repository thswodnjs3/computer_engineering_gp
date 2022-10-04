import cv2
import gc
import warnings
import numpy as np
import os
import pandas as pd
import pickle
import time
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings('ignore')

device = torch.device('cuda')

csv = pd.read_csv('./open/train_df.csv', index_col=0)

path = "./open(original)"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.3, 0.3, 0.3), (0.3, 0.3, 0.3))
])

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

def img_load(path, size):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, size)
    return img
            
def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score

def make_dataset(label, fix, size, batch_size=32):
    train_labels = csv[csv["class"]==fix][label]
    label_unique = sorted(np.unique(train_labels))
    if os.path.isfile(f"./pickle/size-{size[0]}_class-{fix}_state_train.pickle") and os.path.isfile(f"./pickle/size-{size[0]}_class-{fix}_state_test.pickle"):
        with open(f"./pickle/size-{size[0]}_class-{fix}_state_train.pickle", "rb") as fr:
            train_dataset = pickle.load(fr)
        with open(f"./pickle/size-{size[0]}_class-{fix}_state_test.pickle", "rb") as fr:
            test_dataset = pickle.load(fr)
    else:
        label_to_num = {key:value for key,value in zip(label_unique, range(len(label_unique)))}
        num_to_label = {value:key for key,value in zip(label_unique, range(len(label_unique)))}

        train_labels = [label_to_num[x.split('/')[4]] for x in train_paths if x.split('/')[2]==fix and x.split('/')[4] in label_unique]
        test_labels = [label_to_num[x.split('/')[4]] for x in test_paths if x.split('/')[2]==fix and x.split('/')[4] in label_unique]
        
        train_png = [x for x in train_paths if x.split('/')[2]==fix and x.split('/')[4] in label_unique]
        test_png = [x for x in test_paths if x.split('/')[2]==fix and x.split('/')[4] in label_unique]

        train_imgs = [img_load(m, size) for m in train_png]
        test_imgs = [img_load(n, size) for n in test_png]
        
        train_dataset = Custom_dataset(np.array(train_imgs), np.array(train_labels))
        test_dataset = Custom_dataset(np.array(test_imgs), np.array(test_labels))
        
        with open(f"./pickle/size-{size[0]}_class-{fix}_state_train.pickle", "wb") as fw:
            pickle.dump(train_dataset, fw)
        with open(f"./pickle/size-{size[0]}_class-{fix}_state_test.pickle", "wb") as fw:
            pickle.dump(test_dataset, fw)
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    
    return train_loader, test_loader, len(label_unique)

class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, transforms=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transforms=transforms
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = transform(self.img_paths[idx])
        label = self.labels[idx]
        return img, label

class Network(nn.Module):
    def __init__(self, num_classes):
        super(Network, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x

class myModel2():
    def __init__(self):
        pass
    
    def setting(self, size=(128, 128), batch_size=32):
        self.train_loaders = []
        self.test_loaders = []
        self.num_classes = []
        self.classes = []
        for class_ in sorted(np.unique(csv["class"])):
            a, b, c = make_dataset("state", class_, size=size, batch_size=batch_size)
            self.train_loaders.append(a)
            self.test_loaders.append(b)
            self.num_classes.append(c)
            self.classes.append(class_)

    def train(self,epochs=10):
        scores = []
        total_pred = []
        total_true = []

        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler() 
        for tr, te, c, cl in zip(self.train_loaders, self.test_loaders, self.num_classes, self.classes):
            model_ = Network(c).to(device)
            optimizer = torch.optim.Adam(model_.parameters(), lr=1e-3)
            start=time.time()
            for epoch in range(epochs):
                model_.train()
                for batch in (tr):
                    optimizer.zero_grad()
                    x = torch.tensor(batch[0], dtype=torch.float32, device=device)
                    y = torch.tensor(batch[1], dtype=torch.long, device=device)
                    with torch.cuda.amp.autocast():
                        pred = model_(x)
                    loss = criterion(pred, y)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    gc.collect()
                    torch.cuda.empty_cache()
                print(f"{epoch+1}")
            TIME = time.time() - start
            model_.eval()
            with torch.no_grad():
                error = 0
                y_pred=[]
                y_true=[]
                for batch in (te):
                    x = torch.tensor(batch[0], dtype = torch.float32, device = device)
                    y = torch.tensor(batch[1], dtype=torch.long, device=device)
                    with torch.cuda.amp.autocast():
                        pred = model_(x)
                    error = criterion(pred, y)
                    error += error.item()/len(tr)
                    
                    y_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                    y_true += y.detach().cpu().numpy().tolist()
                    
                    total_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                    total_true += y.detach().cpu().numpy().tolist()
                f1 = score_function(y_true, y_pred)
            print(f'{cl}    loss : {loss:.5f}    f1 : {f1:.5f}    time : {TIME:.0f}s')
            scores.append(f1)
        last_f1 = score_function(total_true, total_pred)
        
        return scores, last_f1