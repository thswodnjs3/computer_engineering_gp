import timm
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset

class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, mode='train', transforms=None):
        self.img_paths = img_paths
        self.labels = labels
        self.mode=mode
        self.transforms=transforms
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = self.img_paths[idx]
        if self.mode=='train':
            norm = transforms.Normalize(mean = [0.433038, 0.403458, 0.394151],
                                        std = [0.181572, 0.174035, 0.163234])
        if self.mode=='test':
            norm = transforms.Normalize(mean = [0.418256, 0.393101, 0.386632],
                                        std = [0.195055, 0.190053, 0.185323])
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        img = transforms.ToTensor()(img)
        img = norm(img)
        label = self.labels[idx]
        return img, label
    def get_labels(self):
        return self.labels
    
class Network(nn.Module):
    def __init__(self, num_classes, mode='train'):
        super(Network, self).__init__()
        self.mode = mode
        if self.mode == 'train':
            self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes, drop_path_rate = 0.2)
        if self.mode == 'test':
            self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes, drop_path_rate = 0)
        
    def forward(self, x):
        x = self.model(x)
        return x