from fast_kan import *
from oodEvaluation import *
from cnn_duq import *
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune

from torch.optim.lr_scheduler import ReduceLROnPlateau

#import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from prettytable import PrettyTable

from functions import *
from BSRBF_KAN import *
import random
import numpy as np

import torch
import torch.utils.data
from torch.nn import functional as F

from cnn_duq import CNN_DUQ



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_non_zero_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def count_pruned_parameters(model):
    total = 0
    pruned = 0
    
    for module in model.modules():
        for name, param in module.named_buffers():
            if name.endswith('_mask'):  # For parameters that have been pruned
                total += param.numel()
                pruned += torch.sum(param == 0).item()
    
    return pruned, total, pruned/total if total > 0 else 0

def print_params(model):
    for name, param in model.layers[0].named_parameters(): 
        if param.requires_grad: print(name)
    
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def apply_pruning(model, pruning_method, amount):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
        #if isinstance(module, SplineLinear) or isinstance(module, nn.Linear):
            if pruning_method == 'random':
                prune.random_unstructured(module, name='weight', amount=amount)
                prune.random_unstructured(module, name='weight', amount=amount)
            elif pruning_method == 'l1_unstructured':
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.l1_unstructured(module, name='weight', amount=amount)


def testDE(models,valloader,oodset,criterion):
    #images = valset.data.float()
    #images = images.reshape(-1,1,28,28)
    #labels = valset.targets
    for model in models:
        model.eval()
    preds = []
    totallabels = []
    aurocs = []
    with torch.no_grad():
        for images, labels in valloader:
            batchprobs = []
            for model in models:
                output = model(images.to(device))
                batchprobs.append(output)

        
            avg_probs = torch.stack(batchprobs).mean(dim=0)  
            preds.append(torch.argmax(avg_probs, dim=1))
            totallabels.append(labels)


    for model in models:
        aurocs.append(get_auroc_ood(true_dataset=oodset, ood_dataset=valloader.dataset, model=model, device=device, model_type='mlp'))
    preds = torch.cat(preds)
    totallabels= torch.cat(totallabels)
    val_loss = criterion(avg_probs, labels.to(device)).item()
    val_accuracy = ((preds == totallabels.to(device)).float().mean().item())
    auroc = np.mean(aurocs)

    return val_accuracy,val_loss,auroc

def test(model,valloader,oodset,criterion,type):
    model.eval()
    val_loss, val_accuracy = 0, 0
        
    #y_pred = []
    with torch.no_grad():
        for images, labels in valloader:
            if type == 'kan':
                images = images.view(-1, 28*28).to(device)
            else:
                images = images.to(device)
            output = model(images.to(device))
            val_loss += criterion(output, labels.to(device)).item()
            
            #y_pred += output.argmax(dim=1).tolist()

            val_accuracy += ((output.argmax(dim=1) == labels.to(device)).float().mean().item())
       
        
    auroc = get_auroc_ood(true_dataset=oodset, ood_dataset=valloader.dataset, model=model, device=device, model_type=type)

    val_loss /= len(valloader)
    val_accuracy /= len(valloader)
    return val_accuracy,val_loss,auroc

def train(model,trainloader,valloader,oodset,optimizer,scheduler,criterion,epochs,type):
    trainTime = 0 
    
    for _ in tqdm(range(epochs),desc="Epochs"):
        # Train
        time1 = time.time()
        model.train()
        train_accuracy, train_loss = 0, 0
        for images, labels in trainloader:
            if type == 'kan':
                images = images.view(-1, 28*28).to(device)
            else:
                images = images.to(device)
            optimizer.zero_grad()
            output = model(images.to(device))
            loss = criterion(output, labels.to(device))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            #accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
            train_accuracy += (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            #pbar.set_postfix(loss=train_loss/len(trainloader), accuracy=train_accuracy/len(trainloader), lr=optimizer.param_groups[0]['lr'])
        
        train_loss /= len(trainloader)
        train_accuracy /= len(trainloader)
        scheduler.step()


        trainTime += time.time() - time1
        #val_accuracy,val_loss,auroc = 0,0,0
        val_accuracy,val_loss,auroc = test(model,valloader,oodset,criterion,type)
        #print(f"Epoch {epoch}, Train Loss: {train_loss:.2f}, Train Accuracy: {100*train_accuracy:.2f}")
        #print(f"Epoch {epoch}, Test Accuracy:{100*val_accuracy:.2f} AUROC: {auroc:.2f}")

    print('----')
    return train_accuracy,train_loss,val_accuracy,val_loss,auroc,trainTime

def trainKAN(trainloader,valloader,oodset,epochs,n_input,n_hidden, n_output,gridsize):
    model = BSRBF_KAN([n_input, 64,16,n_output],grid_size=gridsize).to(device)
    print("BSRBFKAN 3l")
    #print("FastKAN 3l")
    #model = FastKAN([n_input, 64,16,n_output],num_grids=8).to(device)

    lr = 1e-3
    wc = 1e-4
    lrDenom = 1e-1
    #denomParam = [p for name, p in model.named_parameters() if 'denominator' in name]
    #othersParam = [p for name, p in model.named_parameters() if 'denominator' not in name]
    #optimizer = optim.AdamW([{"params":othersParam},{"params":denomParam,"lr":lrDenom}], lr=lr, weight_decay=wc)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wc)

    #optimizer = torch.optim.SGD([{"params":othersParam},{"params":denomParam,"lr":lrDenom}],lr = lr)

    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    
    # Define loss
    criterion = LogitNormLoss()
    #criterion = nn.CrossEntropyLoss()
    trainAccs, trainLosses, testAccs, testLosses, aurocs,trainTime = train(model,trainloader,valloader,oodset,optimizer,scheduler,criterion,epochs,type='kan')

    print(f"Train Acc {100*trainAccs:>.2f}%")
    print(f"Test  Acc {100*testAccs:>.2f}")
    print(f"AUROC {aurocs:>.3f}")
    return trainAccs, trainLosses, testAccs, testLosses, aurocs,trainTime

def trainMLP(trainloader,valloader,oodset,epochs):
    lr = 1e-3
    wc = 1e-4
    model = MLPwithCNN(10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wc)
    #optimizer = torch.optim.SGD(model.parameters(),lr = lr)

    #optimizer = torch.optim.SGD([{"params":othersParam},{"params":denomParam,"lr":lrDenom}],lr = lr)

    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    
    # Define loss
    criterion = nn.CrossEntropyLoss()

    trainAccs, trainLosses, testAccs, testLosses, auroc,trainTime = train(model,trainloader,valloader,oodset,optimizer,scheduler,criterion,epochs,type='mlp')

    print(f"Train Acc {100*trainAccs:>.2f}%")
    print(f"Test  Acc {100*testAccs:>.2f}%")
    print(f"AUROC {auroc:>.3f}")

    return trainAccs, trainLosses, testAccs, testLosses, auroc,trainTime

def trainDUQ(trainloader,valloader,oodset,epochs):
    num_classes = 10
    embedding_size = 256
    learnable_length_scale = False
    length_scale = 0.1
    gamma = 0.999
    l = 0.1

    model = CNN_DUQ(
        num_classes,
        embedding_size,
        learnable_length_scale,
        length_scale,
        gamma,
    )
    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(optimizer, 'min',patience=5)

    def gradPenalty2sideCalc(x, ypred):
        gradients = torch.autograd.grad(
                outputs=ypred,
                inputs=x,
                grad_outputs=torch.ones_like(ypred),
                create_graph=True
            )[0]
        #gradPenalty = ((gradients.norm(2,dim=1)**2 - 1)**2).mean()

        gradPenalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradPenalty

    def duqTrainStep(dataloader):
        totalloss = 0
        correct = 0
        for x,y in dataloader:
            y = F.one_hot(y,num_classes=num_classes).type(torch.float)
            x = x.to(device)
            y = y.to(device)

            x.requires_grad_(True) # must for grad penalty

            model.train() # just train flag
            optimizer.zero_grad() # set grads to 0 to not accum
            ypred,_ = model.forward(x)
            loss = F.cross_entropy(ypred,y)
            #### 2-side grad penalty
            loss += l * gradPenalty2sideCalc(x,ypred)
            correct += (torch.argmax(ypred,dim=1) == torch.argmax(y,dim=1)).sum().item()
            loss.backward()
            totalloss += loss.item()
            optimizer.step()
            with torch.no_grad():
                model.update_embeddings(x,y)
        accuracy = correct / len(dataloader.dataset)
        avgLoss = totalloss/len(dataloader)
        return accuracy, avgLoss

    def duqTrain(trainloader, testloader, epochs):
        trainLosses = []
        testLosses  = []
        trainTime = 0
        for _ in tqdm(range(epochs),desc="Epochs"):

            time1 = time.time()
            trainAcc, trainAvgLoss= duqTrainStep(trainloader)
            trainTime += time.time() - time1
            trainLosses.append(trainAvgLoss) 
            testAcc, testAvgLoss = duqTest(testloader)
            scheduler.step(testAvgLoss)
        return trainAcc, trainAvgLoss, trainLosses, testAcc, testAvgLoss, testLosses,trainTime

    def duqTest(dataloader):
        model.eval()
        testLoss, correct = 0, 0
        totalCert = 0
        with torch.no_grad():
            for x, y in dataloader:
                y = F.one_hot(y,num_classes=num_classes).type(torch.float)
                x = x.to(device)
                y = y.to(device)
                ypred,_ = model.forward(x)
                totalCert += torch.max(ypred,dim=1)[0].mean()
                testLoss += F.cross_entropy(ypred, y).item()
                correct += (torch.argmax(ypred,dim=1) == torch.argmax(y,dim=1)).sum().item()
        accuracy = correct / len(dataloader.dataset)
        avgLoss = testLoss/len(dataloader)
        return accuracy, avgLoss 

    trainAcc, trainAvgLoss, trainLosses, testAcc, testAvgLoss, testLosses,trainTime = duqTrain(trainloader,valloader,epochs)

    auroc = get_auroc_ood(true_dataset=valloader.dataset, ood_dataset=oodset, model=model, device=device, model_type='duq')
    print(f"Train Acc {100*trainAcc:>.2f}%")
    print(f"Test  Acc {100*testAcc:>.2f}")
    print(f"AUROC {auroc:>.3f}")

    return trainAcc, trainLosses, testAcc, testLosses, auroc, trainTime

def trainDE(trainloader,valloader,oodset,epochs,n_models):
    lr = 1e-3
    wc = 1e-4
    models = [MLPwithCNN(10).to(device) for i in range(n_models)]
    optimizers = [optim.AdamW(model.parameters(), lr=lr, weight_decay=wc) for model in models]
    schedulers = [optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8) for optimizer in optimizers]
    
    # Define loss
    criterion = nn.CrossEntropyLoss()

    trainAccs = []
    trainLosses = []
    totalTrainTime = 0
    for i in range(n_models):
        trainAcc, trainLoss, testAcc, testLoss, auroc,trainTime = train(models[i],trainloader,valloader,oodset,
                                                                            optimizers[i],schedulers[i],criterion,epochs,type='mlp')
        trainAccs.append(trainAcc)
        trainLosses.append(trainLoss)
        totalTrainTime += trainTime

    testAcc,testLoss,auroc = testDE(models,valloader,oodset,criterion)
    trainAcc = np.mean(trainAccs)
    trainLoss = np.mean(trainLosses)
    print(f"Train Acc {100*trainAcc:>.2f}%")
    print(f"Test  Acc {100*testAcc:>.2f}%")
    print(f"AUROC {auroc:>.3f}")

    return trainAcc, trainLoss, testAcc, testLoss, auroc,totalTrainTime

def runKAN(batch_size = 64, n_input = 28*28, epochs = 25, n_output = 10, n_hidden = 64, \
        grid_size = 5, num_grids = 8, spline_order = 3, ds_name = 'mnist', n_examples = -1,prune_amount=0.3):

    print('KAN')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])

    trainset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
    valset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )

    oodset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    #falseloader = DataLoader(oodset, batch_size=batch_size, shuffle=False)

    #model = FastKAN([n_input, n_hidden, n_output], num_grids = num_grids).to(device)
    

    modelsNum = 5
    

    results = [trainKAN(trainloader,valloader,oodset,epochs,n_input,n_hidden, n_output,num_grids) for i in range(modelsNum)]
    trainAccs, trainLosses, testAccs, testLosses, aurocs,trainTime = zip(*results) # creates lists for each return variable
    finalResult = [
                      np.mean(trainAccs),    np.std(trainAccs),
                      np.mean(trainLosses),  np.std(trainLosses),
                      np.mean(testAccs),     np.std(testAccs),
                      np.mean(testLosses),   np.std(testLosses),
                      np.mean(aurocs),      np.std(aurocs),
                      np.mean(trainTime),      np.std(trainTime),
                  ]# double list for save to excel implement
    print(f"Train Acc {100*finalResult[0]:>.2f}% std {100*finalResult[1]:>.2f}, AvgLoss {finalResult[2]:>.3f} std {finalResult[3]:>.3f}")
    print(f"Test  Acc {100*finalResult[4]:>.2f}% std {100*finalResult[5]:>.2f}, AvgLoss {finalResult[6]:>.3f} std {finalResult[7]:>.3f}")
    print(f"AUROC {finalResult[8]:>.3f} std {finalResult[9]:>.3f}")
    print(f"Train time:{finalResult[10]:>.2f} std {finalResult[11]:>.2f}")

def runDUQ():
    print('DUQ')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])

    trainset, valset = [], []

    trainset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
    valset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )

    oodset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

    trainloader = DataLoader(trainset, batch_size=64, shuffle=False)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)
    #falseloader = DataLoader(oodset, batch_size=batch_size, shuffle=False)

    

    modelsNum = 5
    epochs = 20
    
    results = [trainDUQ(trainloader,valloader,oodset,epochs) for i in range(modelsNum)]
    trainAccs, trainLosses, testAccs, testLosses, aurocs,trainTime = zip(*results) # creates lists for each return variable
    finalResult = [
                      np.mean(trainAccs),    np.std(trainAccs),
                      np.mean(trainLosses),  np.std(trainLosses),
                      np.mean(testAccs),     np.std(testAccs),
                      np.mean(testLosses),   np.std(testLosses),
                      np.mean(aurocs),      np.std(aurocs),
                      np.mean(trainTime),      np.std(trainTime),
                  ]# double list for save to excel implement
    print(f"Train Acc {100*finalResult[0]:>.2f}% std {100*finalResult[1]:>.2f}, AvgLoss {finalResult[2]:>.3f} std {finalResult[3]:>.3f}")
    print(f"Test  Acc {100*finalResult[4]:>.2f}% std {100*finalResult[5]:>.2f}, AvgLoss {finalResult[6]:>.3f} std {finalResult[7]:>.3f}")
    print(f"AUROC {finalResult[8]:>.3f} std {finalResult[9]:>.3f}")
    print(f"Train Time {finalResult[10]:>.3f} std {finalResult[11]:>.3f}")

def runMLP(): 
    print('MLP')
    batch_size = 64
    epochs = 10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])

    trainset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
    valset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )

    oodset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    #falseloader = DataLoader(oodset, batch_size=batch_size, shuffle=False)

    modelsNum = 5
    
    results = [trainMLP(trainloader,valloader,oodset,epochs) for _ in range(modelsNum)]
    trainAccs, trainLosses, testAccs, testLosses, aurocs,trainTime = zip(*results) # creates lists for each return variable
    finalResult = [
                      np.mean(trainAccs),    np.std(trainAccs),
                      np.mean(trainLosses),  np.std(trainLosses),
                      np.mean(testAccs),     np.std(testAccs),
                      np.mean(testLosses),   np.std(testLosses),
                      np.mean(aurocs),      np.std(aurocs),
                      np.mean(trainTime),      np.std(trainTime),
                  ]# double list for save to excel implement
    print(f"Train Acc {100*finalResult[0]:>.2f}% std {100*finalResult[1]:>.2f}, AvgLoss {finalResult[2]:>.3f} std {finalResult[3]:>.3f}")
    print(f"Test  Acc {100*finalResult[4]:>.2f}% std {100*finalResult[5]:>.2f}, AvgLoss {finalResult[6]:>.3f} std {finalResult[7]:>.3f}")
    print(f"AUROC {finalResult[8]:>.3f} std {finalResult[9]:>.3f}")
    print(f"Train time:{finalResult[10]:>.2f} std {finalResult[11]:>.2f}")
        
def runDE():
    print('5-ensemble')
    n_ens_models = 5
    batch_size = 64
    epochs = 10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])

    trainset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
    valset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )

    oodset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    #falseloader = DataLoader(oodset, batch_size=batch_size, shuffle=False)

    modelsNum = 5
    

    results = [trainDE(trainloader,valloader,oodset,epochs,n_ens_models) for _ in range(modelsNum)]
    trainAccs, trainLosses, testAccs, testLosses, aurocs,trainTime = zip(*results) # creates lists for each return variable
    

    finalResult = [
                      np.mean(trainAccs),    np.std(trainAccs),
                      np.mean(trainLosses),  np.std(trainLosses),
                      np.mean(testAccs),     np.std(testAccs),
                      np.mean(testLosses),   np.std(testLosses),
                      np.mean(aurocs),      np.std(aurocs),
                      np.mean(trainTime),      np.std(trainTime),
                  ]# double list for save to excel implement
    print(f"Train Acc {100*finalResult[0]:>.2f}% std {100*finalResult[1]:>.2f}, AvgLoss {finalResult[2]:>.3f} std {finalResult[3]:>.3f}")
    print(f"Test  Acc {100*finalResult[4]:>.2f}% std {100*finalResult[5]:>.2f}, AvgLoss {finalResult[6]:>.3f} std {finalResult[7]:>.3f}")
    print(f"AUROC {finalResult[8]:>.3f} std {finalResult[9]:>.3f}")
    print(f"Train time:{finalResult[10]:>.2f} std {finalResult[11]:>.2f}")

#prune_amounts= [0.1,0.2]
#for pr in prune_amounts:
#    runFastKAN(prune_amount=pr)
#runDUQ()
#runDE()
#runMLP()
runKAN()