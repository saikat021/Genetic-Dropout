import torch
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
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

class DNA: 
    #constructor for the creation of the mask as a gene object 
    def __init__(self,maskLength):
        self.maskLength=maskLength
        self.gene=torch.bernoulli(torch.empty(1,maskLength).uniform_(0,1))#creation of mask 
        self.fit=0

    def keep_prob (self):
        num_one =0
        for i in range (self.maskLength):
            if (self.gene[0,i]==1):
                num_one=num_one+1
        return float(num_one/self.maskLength)
    
    
    #finding the fitness of a particular mask
    #accuracy of all training set is the fitness in one epoch
    #putting model in train mode
    def fitness(self,model,X_train,criterion,y_train):
 
        running_loss=0
        running_corrects=0
        model.train()
        outputs=model.Forward(X_train,self.gene,self.keep_prob())
        _,preds=torch.max(outputs,1)
        loss=criterion (outputs,y_train)
        acc = 100*torch.sum(y_train==torch.max(outputs.data, 1)[1]).double() / len(y_train)
        self.fit=acc
        return acc
        
    #one parent is the passed in the argument 
    #another parent is the one from which this function is called 
    #another parent is self.gene
    def crossover (self,parent2):
        
        child =DNA(self.maskLength)
        midpoint =random .randint (0,self.maskLength-1)
        for i in range (0,self.maskLength):
            if (i>midpoint):
                child.gene [0,i]=self.gene[0,i]
            else :
                child.gene [0,i]=parent2.gene[0,i]
        
        return child
    
    
    #randomly activate some of the nodes  
    #mutate some of the genes
    def mutate(self,mutation_rate):
         
        for i in range (self.maskLength):
            if (random.randint (0,99)<=mutation_rate*100):
                self.gene[0,i]=1 