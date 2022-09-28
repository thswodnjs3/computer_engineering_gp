import pandas as pd
import time
import timm
import torch
import torch.nn as nn
import warnings

from IPython.display import display
from sklearn.metrics import f1_score

from src.data import make_paths
from src.data import make_dataset

def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score

warnings.filterwarnings('ignore')

device = torch.device('cuda')

train_paths, test_paths = make_paths(path="./open(original)")

class Network(nn.Module):
    def __init__(self, num_classes):
        super(Network, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class MyModel():
    def __init__(self):
        pass
    
    def setting(self, size, class_, augmentation=None, batch_size=32):
        self.class_ = class_
        self.train_loader, self.test_loader, self.num_classes, self.num_to_label = \
            make_dataset(size=(size,size), train_paths=train_paths, test_paths=test_paths, class_=class_, augmentation=augmentation, batch_size=batch_size)
    
    def make_model(self, model=None):
        if model is None:
            self.model = Network(self.num_classes)
        else:
            self.model = model
        self.model.to(device)
        return self.model

    def train(self, epochs=10, verbose=True, show_predict=False):
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
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
            
            if verbose:
                print(f'Epoch: {epoch+1}/{epochs}, Time: {train_time:.0f}s/{train_time*(epochs-epoch-1):.0f}s, Train loss: {train_loss:.5f}, Train f1 : {train_f1:.3f}, Test loss : {test_loss:.5f}, Test f1 : {test_f1:.2f}, Test time : {test_time:.2f}s')
                
            if epoch+1==epochs and show_predict:
                real_label = [self.num_to_label[r] for r in test_y]
                predict_label = [self.num_to_label[p] for p in test_pred]
                predict = pd.DataFrame(columns=['real', 'predict'], index=range(1, len(test_y)+1))
                predict['real'] = real_label
                predict['predict'] = predict_label
                display(predict)
                