
#build model
import math
import random
from torchvision import datasets ,transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset
from torch import nn 
from torch.nn.functional import relu ,softmax 
import copy 
from torch.utils.data import Subset
import pandas as pd 
import numpy as np
from PIL import Image
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch

class Model(nn.Module):
    def __init__(self,input_size, hidden_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size , hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.drop = nn.Dropout (p=0.4)

    def masking (self,act1,mask,p):
        if (self.training ==True ):
            return ((act1*mask)/p)
        else :
            return (act1)
        

    def Forward(self, x,mask,p):
        x = F.relu(self.fc1(x))
        x = self.masking (x,mask,p)
        x = self.fc2(x)
        return x