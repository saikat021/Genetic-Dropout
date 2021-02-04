#the mask has to be specified 
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

def train_model(model, criterion, optimizer,mask,p,X_train,y_train,X_test,y_test, num_epochs=25):
  
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    losses=[]
    accuracies=[]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs= model.Forward(X_train,mask,p) #inputs
                    train_loss = criterion(outputs, y_train)# backward + optimize only if in training phase,labels
                    train_loss.backward()
                    optimizer.step()
                train_acc = 100 * torch.sum(y_train==torch.max(outputs.data, 1)[1]).double() / len(y_train)
                print('********************{}*************'.format(phase))
                print ('Epoch [%d/%d] Loss: %.4f   Acc: %.4f'%(epoch+1, num_epochs, train_loss.item(), train_acc.item()))
                losses.append(train_loss)
                accuracies.append(train_acc)
            else:
                model.eval()   # Set model to evaluate mode
                outputs= model.Forward(X_test,mask,p) #inputs
                _, predicted = torch.max(outputs, 1)
                test_loss = criterion(outputs, y_test)
                test_acc = 100 * torch.sum(y_test==predicted).double() / len(y_test)
                print('********************{}*************'.format(phase))
                print ('Epoch [%d/%d] Loss: %.4f   Acc: %.4f'%(epoch+1, num_epochs, test_loss.item(), test_acc.item()))
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

              



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
          time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    #load best model weights
    model.load_state_dict(best_model_wts)
    return model,losses,accuracies