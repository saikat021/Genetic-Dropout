from DNA import DNA
import math
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

class Population:   
    #constructor for initialising the population list
    #list of DNA objects
    def __init__(self,m,num,maskLength,X_train,criterion,y_train):
        self.population=[]
        self.mutation_rate=m #mutation rate for mutation
        self.popmax=num #maximum number of entities in the population
        self.maskLength=maskLength
        self.train=X_train
        self.criterion=criterion
        self.y_train=y_train

        for i in range (num):
            #creating a dna object
            #an initial random population created 
            dna =DNA(self.maskLength)
            self.population.append (dna)
      
        self.matingPool=[]
    
    #going through all the entities of population 
    #finding fitness of all population entities 
    def calcFitness (self,model):
        
        for i in range(0,self.popmax):
            self.population[i].fitness (model,self.train,self.criterion,self.y_train)

    def naturalSelection(self):
        self.matingPool=[]
        maxFitness=0
        for i in range (self.popmax):
            # moving throught the entire population 
            if (self.population[i].fit>maxFitness):
                maxFitness=self.population[i].fit
       
        #max Fitness has the maximum loss score of the entire population  
        #iterating through the all inviduals of the population
        for i in range (self.popmax ):
        
            n=self.Mymap(self.population[i].fit,0,maxFitness,0,1)
        
            n=math.floor(n*100)
            
            for j in range (n):
                self.matingPool.append (self.population[i])#creating mating pool

    def Mymap(self,num,prevlow,prevhigh,nextlow,nexthigh):
        
        prevrange =float((num-prevlow)/(prevhigh-prevlow))
        return nextlow+(nexthigh-nextlow)*prevrange

    def   generate (self):
        for i in range (self.popmax ):
            index_1=math.floor(random.randint  (0,len(self.matingPool)-1))
            index_2=math.floor (random.randint (0,len(self.matingPool)-1))
            parent1=self.matingPool[index_1]
            parent2=self.matingPool[index_2]
            child=parent1.crossover(parent2)
            child.mutate(self.mutation_rate)
            self.population[i]=child
            
    #returns the fiitest individual mask of the population 
    #also returns the keeping probability of the fittest mask 
    def fittest(self):
        fittest=self.population[0]
        for i  in range (self.popmax):
            if (fittest.fit<self.population[i].fit):
                fittest=self.population[i]
        return fittest,fittest.keep_prob()