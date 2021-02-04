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

from Population import Population
from DNA import DNA
from train import train_model
from Model import Model

device="cpu"
import pandas as pd
path ='Hand_Written_script_classification data/Line-level-Features/DHT_Algorithm_144_Features.csv'
#Write the path of the dataset
df = pd.read_csv(path) #header=None
df.head()
print(df.shape)
print(df.shape)
#X=df.loc[:,df.columns != 'Classifier'].values
X=df.loc[:,df.columns != df.columns[-1]].values
print(type(X))
print(X.shape)
#y_label=df['Classifier'].values 
y_label =df.iloc[:,-1].values
#print(type(y_label))
#print(X[0])
print()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y_label)
print(le.classes_)
y=le.transform(y_label)
print(y[:10])
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #for normalization
sc = StandardScaler()
X = sc.fit_transform(X)
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0, 1))
# X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
# print(y_train)
# print(y_test)
#print(len(X_train), len(y_train), len(X_test), len(y_test))
# # X=X_train
# # y=y_train
print(set(y_train), set(y_test))
print(X.shape, y.shape)
X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.long)
#n_inputs = X_train.shape[1]
#n_hidden = 180
#n_classes = len(le.classes_)
#N = X_train.shape[0] # Number of samples
#print(N)
#hyperparameters

input_size = X_train.shape[1]    #The image size = 28 x 28 = 784
hidden_size = X_train.shape[1]    #The number of nodes at the hidden layer
num_classes = len(le.classes_)       #The number of output classes. In this case, from 0 to 9
num_epochs = 100      # The number of times entire dataset is trained
batch_size = X_train.shape[0]    # The size of input data took for one iteration
lr = 0.001  # The speed of convergence
mutation_rate =0
max_population=30
maskLength=X_train.shape[1]
model=Model(input_size, hidden_size, num_classes)#creating the object of the class
model.to(device)
criterion=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#**CONTROL BLOCK** controls the epochs and the generations of mask
#step1: an object of the population class randomly generating the first population 
#step2: calculate fitness of each entitiy of the population 
#step3: creates a mating pool of the population based on the worst two performing parent 
#step4: fittest mask of the generating along with keep_prob found 
#step5: if 0th ,10th ,20th, the epochs starts training on the worst performing mask /other wise new generation is created 

epochgens=0
population =Population(mutation_rate,max_population,maskLength,X_train,criterion,y_train)
total_acc=[]
p=0
while (epochgens<=num_epochs):
    print ('Epoch generations (',epochgens,'/{})'.format(num_epochs),end=' :')
    if p < 0.8:
        population .calcFitness(model)
        population.naturalSelection()
        fittestmask,p = population .fittest()
        accuracy=fittestmask.fitness(model,X_train,criterion,y_train)
        print ("accuracy(fittest mask)",accuracy,"keep_prob",p,end='\n')
    if (epochgens%10==0):
        model,losses,accuracies=train_model(model,criterion,optimizer,fittestmask.gene,p,X_train,y_train,X_test,y_test,30)
        total_acc=total_acc+accuracies
    population.generate()
    epochgens+=1