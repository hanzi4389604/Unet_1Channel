#!/usr/bin/env python
# coding: utf-8

# In[83]:


import torch
import os
import glob
from torch.utils.data import Dataset
import torch.nn as nn
from torch import optim
from PIL import Image
import torchvision.transforms as T

transform = T.Compose([
    T.Resize(256),
    T.ToTensor(),]) 


class DATA_Loader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
   
       
    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        # label_path
        label_path = image_path.replace('image', 'label')
        # training image and label
        image=Image.open(image_path)
        image = transform(image)
        label=Image.open(label_path)
        label = transform(label)
        if label.max() > 1:
            label = label / 255
        return image, label
 
    def __len__(self):
        return len(self.imgs_path)


class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels):    
        super().__init__()
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
#        print('TripleConv',self.triple_conv(x).shape)
        return self.triple_conv(x)
 
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):    
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  ###
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
#        print('DoubleConv++',self.double_conv(x).shape)
        return self.double_conv(x)




class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
#        print('After/Maxpool_double_conv',self.maxpool_conv(x).shape)
        return self.maxpool_conv(x)
 


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):   
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) 
        self.conv = DoubleConv(in_channels, out_channels)
 
    def forward(self, x1, x2):    ###
        x1 = self.up(x1)
#        print('x up',x1.shape)
#        print('x cat',x2.shape)
        x = torch.cat([x2, x1], dim=1)
#        print('important',x.shape)
        return self.conv(x)
 
 

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
 
    def forward(self, x):
        return self.conv(x)
    
    
    
### Network structure
 
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):   
        super(UNet, self).__init__()
        self.n_channels = n_channels   
        self.n_classes = n_classes   
        self.bilinear = bilinear
 
        self.inc = TripleConv(n_channels, 64)
        self.down1 = Down(64, 128)    
        self.down2 = Down(128, 256)   
        self.down3 = Down(256, 512)   
        self.down4 = Down(512, 1024)  
        self.up1 = Up(1024, 512, bilinear)   
        self.up2 = Up(512, 256, bilinear)   
        self.up3 = Up(256, 128, bilinear)  
        self.up4 = Up(128, 64, bilinear)  
        self.outc = OutConv(64, n_classes)   
 
    def forward(self, x):
        x1 = self.inc(x)
#        print('x1 shape',x1.shape)
        x2 = self.down1(x1)
#        print('x2 shape',x2.shape)
        x3 = self.down2(x2)
#        print('x3 shape',x3.shape)
        x4 = self.down3(x3)
#        print('x4 shape',x4.shape)
        x5 = self.down4(x4)
#        print('x5 shape',x5.shape)
        x = self.up1(x5, x4)
#        print('xup1 shape',x.shape)
        x = self.up2(x, x3)
#        print('xup2 shape',x.shape)
        x = self.up3(x, x2)
#        print('xup3 shape',x.shape)
        x = self.up4(x, x1)
#        print('xup4 shape',x.shape)
        logits = self.outc(x)
#        print('logit shape',logits.shape)
        return logits

###training
 
def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):
    train_dataset = DATA_Loader(data_path)
    print("image number", len(train_dataset))  
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)   
#    optimizer = optim.Adam(net.parameters())
    criterion = nn.BCEWithLogitsLoss() 
    #    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')

    for epoch in range(epochs):
        net.train()
        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = net(image)
#            print('pred.shape',pred.shape)
#            print('label.shape',label.shape)
            loss = criterion(pred, label)
            print('Loss/train', loss.item())
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')   

            loss.backward()
            optimizer.step()
 


if __name__ == "__main__":
    net = UNet(n_channels=1, n_classes=1)
    print(net)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = "data/train/"           
    train_net(net, device, data_path)

