from dataset_1 import *
from oodEvaluation import *
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from sklearn.datasets import load_iris



## feature network
class SimpleLinearNetwork(nn.Module):
    def __init__(self,inFeaturesDim,outFeaturesDim):
        super().__init__()
        self.fc1 = nn.Linear(inFeaturesDim, outFeaturesDim)
        # output layer
        self.fc2 = nn.Linear(outFeaturesDim, 3)

    def forwardSoftmax(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x,dim=1)
    
    def forwardFeatures(self,x):
        x = F.relu(self.fc1(x))
        return x

class SimpleLinearNetwork_DUQ(SimpleLinearNetwork):
    def __init__(self, inputDim,outFeatureDim,centroidDim,outputDim):
        super().__init__(inputDim,outFeatureDim)
        ## DUQ weight vector, Parameter so the optimizer and backprop get it into consideration
        ## size = centroidDim x noClasses x featureDim(prev output)
        self.W = nn.Parameter(
            torch.normal(torch.zeros(centroidDim, outputDim, outFeatureDim),std=1)
        )

        # length scale
        #self.sigma = 1
        self.sigma = nn.Parameter(torch.rand_like(torch.zeros(outputDim)))
        # momentum
        self.gamma = 0.999

        #centroids are calculated as e_ct = m_ct/n_ct, c=class t = minibatch
        # register buffers = parameters that dont return with parameters() call, so it wont calc the derivs for backprop
        self.register_buffer("n",torch.ones(outputDim))
        self.register_buffer('m', torch.normal(torch.zeros(centroidDim, outputDim), std = 0.1))
    
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

def gradPenalty2sideCalc(x, ypred):
    gradients = torch.autograd.grad(
            outputs=ypred,
            inputs=x,
            grad_outputs=torch.ones_like(ypred),
            create_graph=True,
        )[0]
    gradPenalty = ((gradients.norm(2,dim=1)**2 - 1)**2).mean()
    return gradPenalty


def trainSoftmax(device, dataloader, model, lossfn, optimizer):
    model.train()
    trloss,correct = 0, 0 
    for batch, (X,y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model.forwardSoftmax(X)
        loss = lossfn(pred, y)
        trloss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        correct += (torch.argmax(pred,dim=1) == y).sum().item()
    print(f"Train Error = Accuracy: {100*correct/len(dataloader.dataset):>.1f}, Avg Loss: {trloss/len(dataloader):>.4f}")

def testSoftmax(device,dataloader, model, lossfn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model.forwardSoftmax(X)
            test_loss += lossfn(pred, y).item()
            correct += (torch.argmax(pred,dim=1) == y).type(torch.float).sum().item()
            #print(torch.argmax(pred,dim=1).tolist())
    test_loss /= num_batches
    correct /= size
    print(f"Test Error = Accuracy: {(100*correct):>.1f}%, Avg loss: {test_loss:>.4f} \n")

def softmaxTrainTest(device,dataloader):
    model = SimpleLinearNetwork(inFeaturesDim=3,outFeaturesDim=16).to(device)
    for e in range(1,11):
        print(f"Epoch {e}")
        trainSoftmax(device,dataloader,model,nn.CrossEntropyLoss(),torch.optim.Adam(model.parameters(), lr = 1e-2))
        testSoftmax(device,dataloader,model, nn.CrossEntropyLoss())


def duqModelOptimizerInit(device):
    duqmodel = SimpleLinearNetwork_DUQ(3,32,16,3).to(device)
    optimizer = torch.optim.SGD(duqmodel.parameters(),lr = 1e-2)
    return duqmodel, optimizer


def duqTrain(device, dataloader, model, optimizer, l):
    totalloss = 0
    correct = 0
    for x,y in dataloader:
        y = F.one_hot(y,num_classes=3).type(torch.float)
        x = x.to(device)
        y = y.to(device)

        x.requires_grad_(True) # must for grad penalty

        model.train() # just train flag
        optimizer.zero_grad() # set grads to 0 to not accum
        ypred,z = model.forward(x)
        loss = F.cross_entropy(ypred,y)
        #### 2-side grad penalty
        loss += l * gradPenalty2sideCalc(x,ypred)
        correct += (torch.argmax(ypred,dim=1) == torch.argmax(y,dim=1)).sum().item()
        loss.backward()
        totalloss += loss
        optimizer.step()
        with torch.no_grad():
            model.updateCentroids(x,y)
    print(f"Train: Accuracy: {100*correct/len(dataloader.dataset):>.1f}%, Avg Loss: {totalloss/len(dataloader):>.4f}")

def duqTest(device, dataloader, model):
    model.eval()
    test_loss, correct = 0, 0
    totalCert = 0
    with torch.no_grad():
        for x, y in dataloader:
            y = F.one_hot(y,num_classes=3).type(torch.float)
            x = x.to(device)
            y = y.to(device)
            ypred,z = model.forward(x)
            totalCert += torch.max(ypred,dim=1)[0].mean()
            test_loss += F.cross_entropy(ypred, y)
            correct += (torch.argmax(ypred,dim=1) == torch.argmax(y,dim=1)).sum().item()
    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    print(f"Test: Accuracy: {(100*correct):>.1f}%, Avg loss: {test_loss:>.4f} Avg Cert: {totalCert/len(dataloader.dataset):>.5f}")

def duqTrainTest(device, model, optimizer,trainloader,testloader, epochs):
    # grad penalty var
    l = 0.1
    for e in range(epochs):
        print(f"Epoch {e+1}")
        duqTrain(device, trainloader,model,optimizer,l)
    print(f"\nMain test")
    duqTest(device, testloader,model)


def loadIrisDataloader():
    iris = load_iris()
    irisX = iris['data'][:,1:4]
    scaler = StandardScaler()
    irisX = scaler.fit_transform(irisX)
    irisX = torch.autograd.Variable(torch.tensor(irisX,dtype=torch.float))
    irisY = torch.autograd.Variable(torch.tensor(iris['target'], dtype=torch.long))
    irisDataloader = DataLoader(TensorDataset(irisX, irisY), batch_size=1000, shuffle=True)
    return irisDataloader

def main():
    # set dev to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"On {device}")
    ambTestSet, ambTrainLoader, ambTestLoader, ambFeaturesDim = load_D1(1)
    #softmaxTrainTest(device,ambTrainLoader)

    print(f"Train size: {len(ambTrainLoader.dataset)}")
    print(f"Test size: {len(ambTestLoader.dataset)}")

    epochs = 20

    model, optimizer = duqModelOptimizerInit(device)

    duqTrainTest(device,model,optimizer, ambTrainLoader, ambTestLoader, epochs)

    irisDataloader = loadIrisDataloader()
    #print("IRIS test")
    #duqTest(device,irisDataloader,model)

    print("AUROC")
    auroc = get_auroc_ood(true_dataset=ambTestLoader.dataset, ood_dataset=irisDataloader.dataset, model=model, device="cuda:0", standard_model=False)
    print(auroc)
    #get_auroc_ood(ambTestSet,irisDataloader.dataset,model)


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#ambTestSet, ambTrainLoader, ambTestLoader, ambFeaturesDim = load_D1(1)
#softmaxTrainTest(device, ambTrainLoader)
#loadIrisDataloader()
main()