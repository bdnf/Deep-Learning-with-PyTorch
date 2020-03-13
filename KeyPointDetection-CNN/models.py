## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # output size = (W-F)/S +1 = (244-5)/1 +1 = 240
        # the output Tensor for one image, will have the dimensions: (128, 240, 240)
        # after one pool layer, this becomes (128, 48, 48)
        self.conv1 = nn.Conv2d(1, 128, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(5, 5)
        # second conv layer: 32 inputs, 20 outputs, 5x5 conv
        # output size = (W-F)/S +1 = (48-5)/1 +1 = 44
        # the output Tensor for one image, will have the dimensions: (64, 44, 44)
        # after one pool layer, this becomes (64, 8, 8)
        
        self.conv2 = nn.Conv2d(128, 64, 5)
        # third conv layer: 64 inputs, 32 outputs, 3x3 conv
        # output size = (W-F)/S +1 = (8-3)/1 +1 = 6
        # the output Tensor for one image, will have the dimensions: (32, 6, 6)
        # after one pool layer, this becomes (32, 3, 3)
        self.conv3 = nn.Conv2d(64, 32, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # fully connected layer
        self.fc1 = nn.Linear(32*3*3, 256)
        # dropout with p=0.2
        #self.fc1_drop = nn.Dropout(p=0.2)
        
        #self.fc2 = nn.Linear(256, 128)
        #self.fc2_drop = nn.Dropout(p=0.2)
        
        # output layer
        self.fc2 = nn.Linear(256, 136)

   
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # two conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        #x = self.fc1_drop(x)
        #x = F.relu(self.fc2(x))
        #x = self.fc2_drop(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
