import torch

from torch.functional import F
from tqdm import tqdm
from scripts.functions import gradPenalty2sideCalc
from scripts.oodEvaluation import get_auroc_ood
from scripts.tools import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def KANDUQtrainStep(model,optimizer,lossFunction,trainLoader,numClasses,gradPenaltyL):
    model.train()
    totalLoss  = 0
    correct    = 0
    for x,y in trainLoader:
        x = x.to(device)
        y = y.to(device)
        y = F.one_hot(y,num_classes=numClasses).type(torch.float)

        if gradPenaltyL:
            x.requires_grad_(True) # must for grad penalty
        optimizer.zero_grad()
        #kanoutput = model.forwardKAN(x)
        output = model(x)
        #output = model.forwardDUQ(kanoutput)
        loss = lossFunction(output,y)
        if gradPenaltyL:
            #loss += gradPenaltyL * gradPenalty2sideCalc(kanoutput,output)
            loss += gradPenaltyL * gradPenalty2sideCalc(x,output)
        loss.backward()
        correct += (torch.argmax(output,dim=1) == torch.argmax(y,dim=1)).sum().item()
        totalLoss += loss.item()
        optimizer.step()

        with torch.no_grad():
            model.updateCentroids(output,y)
            #model.updateCentroids(kanoutput,y)
    totalLoss /= len(trainLoader)
    accuracy = correct / len(trainLoader.dataset)
    return accuracy, totalLoss

def networkTrainStep(netType, model,optimizer,lossFunction,trainLoader,numClasses,gradPenaltyL):
    model.train()
    totalLoss  = 0
    correct    = 0
    for x,y in trainLoader:
        x = x.to(device)
        y = y.to(device)
        y = F.one_hot(y,num_classes=numClasses).type(torch.float)

        if gradPenaltyL:
            x.requires_grad_(True) # must for grad penalty
        optimizer.zero_grad()
        output = model(x)
        loss = lossFunction(output,y)
        if gradPenaltyL:
            loss += gradPenaltyL * gradPenalty2sideCalc(x,output)
        loss.backward()
        correct += (torch.argmax(output,dim=1) == torch.argmax(y,dim=1)).sum().item()
        totalLoss += loss.item()
        optimizer.step()

        if netType.lower()=='duq':
            with torch.no_grad():
                model.updateCentroids(x,y)
    totalLoss /= len(trainLoader)
    accuracy = correct / len(trainLoader.dataset)
    return accuracy, totalLoss

def networkTest(model,lossFunction,testLoader):
    model.eval()
    valLoss = 0
    valAccuracy = 0
    with torch.no_grad():
        for x, y in testLoader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            valLoss += lossFunction(output, y).item()
            valAccuracy += (output.argmax(dim=1) == y).float().mean().item()
    valLoss /= len(testLoader)
    valAccuracy /= len(testLoader)
    return valAccuracy, valLoss

def networkTrain(netType,model,optimizer,scheduler,lossFunction,trainLoader,testLoader,falseLoader,numClasses,gradPenaltyL,epochs):
    trainAccs   = []
    testAccs    = []
    trainLosses = []
    testLosses  = []
    aurocs      = []

    for _ in tqdm(range(epochs),desc="Epochs"):
        if netType.lower() == 'kanduq':
            trainAcc,trainLoss = KANDUQtrainStep(model,optimizer,lossFunction,trainLoader,numClasses,gradPenaltyL)
        else:
            trainAcc,trainLoss = networkTrainStep(netType,model,optimizer,lossFunction,trainLoader,numClasses,gradPenaltyL)
        testAcc,testLoss = networkTest(model,lossFunction,testLoader)
        aurocs.append(get_auroc_ood(true_dataset=testLoader.dataset, ood_dataset=falseLoader.dataset, model=model, device=device, model_type=netType))

        scheduler.step(testLoss)
        trainAccs.append(trainAcc)
        trainLosses.append(trainLosses)
        testAccs.append(testAcc)
        testLosses.append(testLosses)
    
    print(f"TrainAcc:{trainAcc:>.4f}, TrainLoss:{trainLoss:>.3f}")
    print(f"TestAcc: {testAcc:>.4f}, TestLoss: {testLoss:>.3f}")
    print(f"AUROC: {aurocs[-1]:>.3f}")

    plot([i for i in range(epochs)],'epochs',testAccs,'Test acc','b','-')
    plot([i for i in range(epochs)],'epochs',aurocs,'auroc','b','-')

    return trainAccs,trainLosses,testAccs,testLosses,aurocs[-1]
