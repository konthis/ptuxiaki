import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models 

class SimpleLinearNetwork_Softmax(nn.Module):
    def __init__(self, inputDim, secondLayerDim, thirdLayerDim, outputDim):
        self.kwargs = {'inputDim': inputDim, 'secondLayerDim': secondLayerDim, 'thirdLayerDim': thirdLayerDim, 'outputDim':outputDim}
        super().__init__()
        self.fc1 = nn.Linear(inputDim, secondLayerDim)
        self.fc2 = nn.Linear(secondLayerDim, thirdLayerDim)
        self.fc3 = nn.Linear(thirdLayerDim, outputDim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

class SimpleLinearNetwork_DUQ(nn.Module):
    def __init__(self, inputDim,outFeatureDim,centroidDim,outputDim,std,initSigma):
        super().__init__()
        ###### for saving
        self.kwargs = {'inputDim': inputDim, 'outFeatureDim': outFeatureDim, 'centroidDim': centroidDim, 'outputDim':outputDim, 'std':std, 'initSigma':initSigma}
        self.fc1 = nn.Linear(inputDim, outFeatureDim)
        ## DUQ weight vector, Parameter so the optimizer and backprop get it into consideration
        ## size = centroidDim x noClasses x featureDim(prev output)
        self.W = nn.Parameter(
            torch.normal(torch.zeros(centroidDim, outputDim, outFeatureDim),std=1)
        )

        # length scale
        #self.sigma = 1
        # trainable sigma

        # maybe lower lr ###############################
        self.sigma = nn.Parameter(torch.ones_like(torch.zeros(outputDim))*initSigma) ##### change
        # momentum
        self.gamma = 0.999


        self.std = std #  standard div of m
        #centroids are calculated as e_ct = m_ct/n_ct, c=class t = minibatch
        # register buffers = parameters that dont return with parameters() call, so it wont calc the derivs for backprop
        self.register_buffer("n",torch.ones(outputDim))
        self.register_buffer('m', torch.normal(torch.zeros(centroidDim, outputDim), std = std))


        # for pirnting
        self.architecture = [inputDim, outFeatureDim, centroidDim, outputDim]

    def forwardFeatures(self,x):
        x = F.relu(self.fc1(x))
        return x


    def embeddingLayer(self, x):
        #  last weight layer, on DUQ part
        # simple matrix mul
        # x size = batchSize x outFeatureDim
        # z size = batchSize x centroidDim x noclasses
        z = torch.einsum("ij,mnj->imn", x, self.W)
        return z

    def updateCentroids(self,x,y):
        z = self.embeddingLayer(self.forwardFeatures(x))
        # y =  one hot encoded
        ################# MAYBE CHANGE
        #self.n = self.gamma * self.n + (1 - self.gamma) * y.sum(0) # y onehot, sum 0  is total of class i
        self.n = torch.max(self.gamma * self.n + (1 - self.gamma) * y.sum(0), torch.ones_like(self.n)) # IF 0 SAMPLES OF A CLASS FOUND, SET IT TO 1
        self.m = self.gamma * self.m + (1 - self.gamma) * torch.einsum("ijk,ik->jk",z,y) # einsum with onehot enc y, to activate on correct class

    def calcDistanceLayer(self, z):
        # centroids
        e = self.m / self.n
        diff = z - e # no abs as it is squared 
        distances = (-(diff**2)).mean(1).div(2 * self.sigma**2).exp()
        return distances

    def forward(self,x):
        # features
        x = self.forwardFeatures(x)
        # embed
        z = self.embeddingLayer(x)
        # distances/certainty
        ypred = self.calcDistanceLayer(z)
        return ypred,z

    def getCentroids(self):
        return self.m / self.n

    def printSigma(self):
        print(self.sigma.detach().to('cpu').numpy())
    
    def getArchitecture(self):
        return self.architecture

class customLinearNetwork(nn.Module):
    # layersSize = list of sizes [insize,out1size,out2size,...,outsize]
    def __init__(self, layersNum, layersSize):
        super().__init__()
        self.kwargs = {'layersNum': layersNum, 'layersSize': layersSize}
        self.layersNum = layersNum
        self.layers = nn.ModuleList() # create the list, so it uses the torch utilities
        for l in range(layersNum-1):
            self.layers.append(nn.Linear(layersSize[l],layersSize[l+1]))
    def forward(self,x):
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))
        # last layer
        x = self.layers[-1](x)
        return F.softmax(x,dim=1)
        
#https://github.com/y0ast/deterministic-uncertainty-quantification/blob/master/utils/cnn_duq.py
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)

        #self.fc1 = nn.Linear(2 * 2 * 128, 256)
        self.fc1 = nn.Linear(115200,1024)

    def calc_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)

        x = x.flatten(1)

        x = F.relu(self.fc1(x))

        return x

class MLPwithCNN(CNN):
    def __init__(self,num_class):
        super().__init__()
        self.fc2 = nn.Linear(1024,256)
        self.fc3 = nn.Linear(256,num_class)
    
    def forward(self, x):
        x = self.calc_features(x)
        x = F.relu(self.fc2(x))
        y = F.softmax(self.fc3(x),dim=1)
        return y
