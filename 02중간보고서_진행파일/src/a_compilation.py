import albumentations as A
import numpy as np
import os
import pandas as pd
import random
import time
import torch
import torch.nn as nn

from collections import Counter
from sklearn.model_selection import train_test_split
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import ConcatDataset, DataLoader

from src.a_variable import A_list, train_y, classes
from src.a_function import img_load, score_function, load_pickle, store_pickle, make_imgs, make_labels
from src.a_class import Custom_dataset, Network

def make_data(stage, size):
    path_tr = f'./save/{stage}-myimg-train-{size}.npy'
    path_te = f'./save/{stage}-myimg-test-{size}.npy'
    if os.path.isfile(path_tr) and os.path.isfile(path_te):
        train_imgs = np.load(path_tr)
        test_imgs = np.load(path_te)
    else:
        train_imgs = make_imgs(section='train', size=size)
        test_imgs = make_imgs(section='test', size=size)
        np.save(path_tr, np.array(train_imgs))
        np.save(path_te, np.array(test_imgs))
    labels, label_unique = make_labels(train_y, target='label')
    return train_imgs, labels, label_unique, test_imgs

def make_train_loader(train_imgs, train_labels,
                      val_imgs, val_labels,
                      batch_size,
                      aug_state='Off', over_standard=100, under_multiple=3):
    augs = [A_list[key] for key in A_list.keys()]
        
    if aug_state=='On':
        train_ratio = Counter(train_labels)
        
        over_label = [key for key in train_ratio.keys() if train_ratio[key]>over_standard]
        
        over_set = [(img, label) for img, label in zip(train_imgs, train_labels) if label in over_label]
        over_img = [x[0] for x in over_set]
        over_label = [x[1] for x in over_set]
        # over_datasets = [Custom_dataset(np.array(over_img), np.array(over_label), mode='train')]
        over_datasets = []
        over_datasets += [Custom_dataset(np.array(over_img), np.array(over_label), transforms=random.choice([None, A.OneOf(augs, p=1)]), mode='train')]
        
        under_set = [(img, label) for img, label in zip(train_imgs, train_labels) if label not in over_label]
        under_img = [x[0] for x in under_set]
        under_label = [x[1] for x in under_set]
        under_datasets = [Custom_dataset(np.array(under_img), np.array(under_label), mode='train')]
        under_datasets += [Custom_dataset(np.array(under_img), np.array(under_label), transforms=A.HorizontalFlip(p=1), mode='train')]
        for _ in range(under_multiple-1):
            under_datasets += [Custom_dataset(np.array(under_img), np.array(under_label), transforms=A.Affine((45, -45), p=1), mode='train')]
        
        train_datasets = []
        train_datasets += over_datasets
        train_datasets += under_datasets
        train_dataset = ConcatDataset(train_datasets)
        
        val_datasets = []
        val_datasets += [Custom_dataset(np.array(val_imgs), np.array(val_labels), transforms=random.choice([None, A.OneOf(augs, p=1)]), mode='test')]
        # val_datasets = [Custom_dataset(np.array(val_imgs), np.array(val_labels), mode='test')]
        # for aug in augs:
        #     val_datasets += [Custom_dataset(np.array(val_imgs), np.array(val_labels), transforms=aug, mode='test')]
        val_dataset = ConcatDataset(val_datasets)
        
    print(f'Augmentation Result - Train: {len(train_labels)} -> {len(train_dataset)}({len(over_label)}x1+{len(under_label)}x{under_multiple+1}) | Val: {len(val_labels)} -> {len(val_dataset)}(x1)')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
    
    return train_loader, val_loader

def make_test_loader(test_imgs, batch_size):
    test_dataset = Custom_dataset(np.array(test_imgs), np.array(["tmp"]*len(test_imgs)), mode='test')
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return test_loader

def make_hyper(label_unique, device, lr=1e-3, wd = 2e-2, load_model=False, num=0):
    model_ = Network(len(label_unique)).to(device)
    if load_model:
        model_.load_state_dict(torch.load((f'./save/best_model_{num}.pth'))['state_dict'])
    # optimizer = torch.optim.Adam(model_.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model_.parameters(), lr=lr, weight_decay = wd)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    return model_, optimizer, criterion, scaler

def training(epochs,
             model_, train_loader, val_loader,
             optimizer, criterion, scaler,
             device, early_stop, num=0, load_best=False):
    best=0
    best_loss = 0
    if load_best:
        output = load_pickle(f'./pickle/best_{num}th.pickle')
        if output is not None:
            best, best_loss = output
    early_stopping = 0
    for epoch in range(epochs):
        t_start=time.time()
        t_loss = 0
        t_pred=[]
        t_answer=[]
        model_.train()
        for batch in (train_loader):
            optimizer.zero_grad()
            x = torch.tensor(batch[0], dtype=torch.float32, device=device)
            y = torch.tensor(batch[1], dtype=torch.long, device=device)
            with torch.cuda.amp.autocast():
                pred = model_(x)
            loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            t_loss += loss.item()/len(train_loader)
            t_pred += pred.argmax(1).detach().cpu().numpy().tolist()
            t_answer += y.detach().cpu().numpy().tolist()
        t_time = time.time() - t_start
        t_f1 = score_function(t_answer, t_pred)
        
        state_dict=model_.state_dict()
        
        v_start=time.time()
        v_loss = 0
        v_pred = []
        v_answer = []
        model_.eval()
        for batch in val_loader:
            x = torch.tensor(batch[0], dtype=torch.float32, device=device)
            y = torch.tensor(batch[1], dtype=torch.long, device=device)
            with torch.cuda.amp.autocast():
                pred = model_(x)
            loss = criterion(pred, y)
            
            v_loss += loss.item()/len(train_loader)
            v_pred += pred.argmax(1).detach().cpu().numpy().tolist()
            v_answer += y.detach().cpu().numpy().tolist()
        v_time = time.time() - v_start
        v_f1 = score_function(v_answer, v_pred)
        
        if (v_f1 > best) or (v_f1==best and v_loss < best_loss):
            print("-"*50, "Best model updated!", "-"*50)
            best = v_f1
            best_loss = v_loss
            early_stopping = 0
            store_pickle(f'./pickle/best_{num}th.pickle', (best, best_loss))
            torch.save({'epoch':epoch,
                        'state_dict':state_dict,
                        'optimizer':optimizer.state_dict(),
                        'scaler':scaler.state_dict(),
                    }, f'./save/best_model_{num}.pth')
        else:
            early_stopping += 1
        
        print(f'Epoch: {epoch+1}/{epochs}, Train Time: {t_time:.0f}s/{t_time*(epochs-epoch-1):.0f}s, Train loss: {t_loss:.5f}, Train f1: {t_f1:.5f}, Val Time: {v_time:.0f}s/{v_time*(epochs-epoch-1):.0f}s, Val loss: {v_loss:.5f}, Val f1: {v_f1:.5f}')
        if early_stopping == early_stop:
            break
    return model_

def testing(model_, test_loader, device, stage):
    model_.eval()
    f_pred = []

    with torch.no_grad():
        for batch in (test_loader):
            x = torch.tensor(batch[0], dtype = torch.float32, device = device)
            with torch.cuda.amp.autocast():
                pred_first = model_(x)
            pred_list = pred_first.argmax(1).detach().cpu().numpy().tolist()
            
            if stage=='1-stage':
                f_pred.extend(pred_list)
    return f_pred

def make_submission(label_unique, f_pred,
                    stage, size, aug_state, under_multiple):
    label_decoder = {val:key for key, val in label_unique.items()}
    f_pred = [label_decoder[result] for result in f_pred]
    submission = pd.read_csv("open/sample_submission.csv")
    submission["label"] = f_pred
    submission.to_csv(f"./open/{stage}_size-{size}_aug-{aug_state}_multiple-{under_multiple}_yoon_ensemble.csv", index = False)