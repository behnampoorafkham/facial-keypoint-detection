## Define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
       
        # 224 by 224 pixels
        
        ## self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        # output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (1, 220, 220)
        # after one pool layer, this becomes (10, 13, 13)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # 220/2 = 110
        # output size = (W-F)/S +1 = (110-3)/1 + 1 = 108
        # the output Tensor for one image, will have the dimensions: (32, 110, 110)
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # output size = (W-F)/S +1 = (54-3)/1 + 1 = 52
        # the output Tensor for one image, will have the dimensions: (64, 54, 54)
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # output size = (W-F)/S +1 = (26-3)/1 + 1 = 24
        # the output Tensor for one image, will have the dimensions: (128, 26, 26)
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        # output size = (W-F)/S +1 = (12-3)/1 + 1 = 10
        # the output Tensor for one image, will have the dimensions: (256, 12, 12)
        self.conv5 = nn.Conv2d(256, 512, 1)
        
        # output size = (W-F)/S +1 = (6-1)/1 + 1 = 6
        # the output Tensor for one image, will have the dimensions: (512, 6, 6)
        
        # Fully-connected (linear) layers
        self.fc1 = nn.Linear(512*6*6, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 68*2)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.25)
        
        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        # 5 conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        
        # Prep for linear layer / Flatten
        x = x.view(x.size(0), -1)
        
        # linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
