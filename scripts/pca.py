from datasets import *
from duq_mlp_emb import *
from kan_test_train import *

import numpy as np
from torchvision.transforms import transforms
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from medmnist import PathMNIST,ChestMNIST,DermaMNIST,OCTMNIST

import joblib
import os



#torch.manual_seed(42)
##np.random.seed(42)
##
## Load OCTMNIST dataset
#data_transform = transforms.Compose([transforms.ToTensor()])
#octmnist = PathMNIST(root='../datasets/data', split='train', transform=data_transform, download=True)
#octmnist_test = PathMNIST(root='../datasets/data', split='test', transform=data_transform, download=True)
#
#X_train = octmnist.imgs
#y_train = octmnist.labels
#X_test = octmnist_test.imgs
#y_test = octmnist_test.labels
#
#X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
#X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
#y_train = y_train.squeeze()
#y_test = y_test.squeeze()
#
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)
#
#pca = PCA(n_components=62)
#X_train_pca = pca.fit_transform(X_train_scaled)
#X_test_pca = pca.transform(X_test_scaled)
#
#
#X_train_tensor = torch.FloatTensor(X_train_pca)
#y_train_tensor = torch.LongTensor(y_train)
#X_test_tensor = torch.FloatTensor(X_test_pca)
#y_test_tensor = torch.LongTensor(y_test)
#
## Create DataLoader
#traindataset = TensorDataset(X_train_tensor, y_train_tensor)
#testdataset = TensorDataset(X_test_tensor, y_test_tensor)
#
#batch_size = 128
#trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
#testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)
#
#print(pca.n_components_)
#saveDataset(traindataset,'../datasets/data/PathMNIST_62/train.pth')
#saveDataset(testdataset,'../datasets/data/PathMNIST_62/test.pth')




traindataset = loadDataset('../datasets/data/PathMNIST_62/train.pth')
testdataset  = loadDataset('../datasets/data/PathMNIST_62/test.pth')
falsedataset = loadDataset('../datasets/data/PathMNIST_62/noisedtrain.pth')
#falsedataset2 = loadDataset('../datasets/data/ChestMNIST_62/train.pth')
#falsedataset3 = loadDataset('../datasets/data/DermaMNIST_62/train.pth')

trainloader = DataLoader(traindataset, batch_size=64, shuffle=True)
testloader = DataLoader(testdataset, batch_size=64, shuffle=False)
falseloader = DataLoader(falsedataset, batch_size=64, shuffle=False)
#falseloader2 = DataLoader(falsedataset2, batch_size=64, shuffle=False)
#falseloader3 = DataLoader(falsedataset3, batch_size=64, shuffle=False)
falseloader2 = falseloader3 = False

architecture  = [353,64,32,9]
epochs      = 2
lr          = 1e-1
sigma       = 20
lrSigma     = 1e-1
lamda       = 0
std         = 1
modelsNum   = 1
lossFunction = LogitNormLoss()



#trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2,aurocs3 = MLPcreateAndTrain(trainloader,testloader,falseloader,
#                                                                                           falseloader2,falseloader3,architecture,lr,epochs)
trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2,aurocs3 = DUQcreateAndTrain(trainloader,testloader,falseloader,falseloader2,
                                                                                           falseloader3,architecture,std,sigma,lr,lrSigma,
                                                                                           lamda,epochs)
print(f"TrainACC:{trainAccs:>.3f}")
print(f"TestACC:{testAccs:>.3f}")
print(f"AUROC:{aurocs1:>.3f}")
#dataloaders = [trainloader,testloader,falseloader,falseloader2,falseloader3]
#trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2,aurocs3 = KANcreateAndTrain(dataloaders,1,architecture,lossFunction)
##trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2,aurocs3 = zip(*results) # creates lists for each return variable
#finalResult = [architecture,lr,lamda, std,
#                  np.mean(trainAccs),    np.std(trainAccs),
#                  np.mean(trainLosses),  np.std(trainLosses),
#                  np.mean(testAccs),     np.std(testAccs),
#                  np.mean(testLosses),   np.std(testLosses),
#                  np.mean(aurocs1),      np.std(aurocs1),
#              ]# double list for save to excel implement
#if falseloader2: finalResult.extend([np.mean(aurocs2),np.std(aurocs2)])
#if falseloader3: finalResult.extend([np.mean(aurocs3),np.std(aurocs3)])
#print(f"Train Acc {100*finalResult[4]:>.1f}% std {100*finalResult[5]:>.1f}, AvgLoss {finalResult[6]:>.3f} std {finalResult[7]:>.3f}")
#print(f"Test  Acc {100*finalResult[8]:>.1f}% std {100*finalResult[9]:>.1f}, AvgLoss {finalResult[10]:>.3f} std {finalResult[11]:>.3f}")
#print(f"AUROC wine  {finalResult[12]:>.3f} std {finalResult[13]:>.3f}")
#if falseloader2:
#    print(f"AUROC iris  {finalResult[14]:>.3f} std {finalResult[15]:>.3f}")
#if falseloader3:
#    print(f"AUROC canc. {finalResult[16]:>.3f} std {finalResult[17]:>.3f}")
#results = DUQcreateAndTrain(trainloader,testloader,falseloader,False,False,
#                            architecture,std,sigma,lr,lrSigma,lamda,epochs)