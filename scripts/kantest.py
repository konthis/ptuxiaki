from datasets import *
from models import *
from sklearn.datasets import load_iris,load_diabetes,load_breast_cancer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from RBF_KAN import *
from BSRBF_KAN import *
from Fast_KAN import *
from tools import *
from oodEvaluation import *
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
import torch.nn as nn

from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ambTestSet, ambTrainLoader, ambTestLoader, ambFeaturesDim = load_D1(5)
falseloader = createSklearnDataloader(load_diabetes(),[1,2,3])
falseloader2 = createSklearnDataloader(load_iris(),[1,2,3])
falseloader3 = createSklearnDataloader(load_breast_cancer(),[2,3,4])

trainloader = ambTrainLoader
testloader  = ambTestLoader

def KANTrainStep(model,optimizer, lossFunction, l):
    model.train()
    total_loss = 0
    correct = 0
    for x, y in trainloader:

        x = x.to(device)
        y = y.to(device)
        y = F.one_hot(y,num_classes=3).type(torch.float)
        x.requires_grad_(True) # must for grad penalty

        optimizer.zero_grad()
        output = model(x)
        loss = lossFunction(output,y)
        #loss += l * gradPenalty2sideLogits(x,output)
        loss.backward()
        correct += (torch.argmax(output,dim=1) == torch.argmax(y,dim=1)).sum().item()
        total_loss += loss.item()
        optimizer.step()
    total_loss /= len(trainloader)
    accuracy = correct / len(trainloader.dataset)
    return accuracy, total_loss

def KANTest(model, lossFunction):
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for x, y in testloader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            val_loss += lossFunction(output, y).item()
            val_accuracy += (output.argmax(dim=1) == y).float().mean().item()
    val_loss /= len(testloader)
    val_accuracy /= len(testloader)
    return val_accuracy, val_loss

def KANTrain(model,optimizer, scheduler, lossFunction, epochs,l):

    trainaccs = []
    testaccs = []
    trainLosses =[]
    testLosses = []
    auroc  = []
    lrs = []
    for _ in tqdm(range(epochs),desc="Epochs"):
        trainAcc, trainLoss = KANTrainStep(model,optimizer,lossFunction,l)
        testAcc, testLoss = KANTest(model,lossFunction)
        auroc.append(get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader.dataset, model=model, device=device, model_type='kan'))
        trainaccs.append(trainAcc)
        testaccs.append(testAcc)
        trainLosses.append(trainLoss)
        testLosses.append(testLoss)
        scheduler.step(testLoss)
        lrs.append(scheduler.get_last_lr())
    #print(lrs)

    return trainaccs,trainLosses,testaccs,testLosses,auroc,lrs

def KANcreateAndTrain(kanType,architecture,lossFunction, gridsize=8,lr=1e-3,lrDenom=1e-3,initDenominator=1.,epochs=25,lamda=0,gamma=0.5,plot=False):
    model = None
    base_activation = F.silu 
    #base_activation = ActivationFunctions(gamma).RBF_SiLU
    if kanType == 1:
        model = FastKAN(architecture,num_grids=gridsize, base_activation=base_activation,initDenominator=initDenominator).to(device)
        print(model)
    elif kanType == 2:
        model = BSRBF_KAN(architecture,gridsize,base_activation=base_activation,initDenominator=initDenominator).to(device)
        print(model)


    denomParam = [p for name, p in model.named_parameters() if 'denominator' in name]
    othersParam = [p for name, p in model.named_parameters() if 'denominator' not in name]
    optimizer = torch.optim.SGD([{"params":othersParam},{"params":denomParam,"lr":lrDenom}],lr = lr)


    #optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min',patience=5)


    trainAccs, trainLosses, testAccs, testLosses,aurocProgress,lrProgress = KANTrain(model,optimizer,scheduler,lossFunction,epochs,lamda)
    auroc1 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader.dataset, model=model, device=device, model_type='kan')
    auroc2 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader2.dataset, model=model, device=device, model_type='kan')
    auroc3 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader3.dataset, model=model, device=device, model_type='kan')

    #for name, param in model.named_parameters():
    #    if 'rbf.denominator' in name:
    #        print(param)
    if plot:
        plt.plot(trainLosses,label='train loss')
        plt.plot(testLosses,label='test loss')
        plt.show()
        plt.plot(trainAccs,label='train ac')
        plt.plot(testAccs,label='test ac')
        plt.show()
        plt.plot(lrProgress,label='lrprog')
        plt.show()
        plt.plot(aurocProgress,label='auroc2')
        plt.show()
    return trainAccs[-1],trainLosses[-1],testAccs[-1],testLosses[-1],auroc1,auroc2,auroc3

def testKAN():
    save = str(input("Save?(y/n):"))
    plot = False
    if save.lower() == 'y':
        save = True
        runName = str(input('Run name: '))
    elif save.lower() == 'n':
        save = False
        plot = str(input("plot?(y/n):"))
        plot = True if plot.lower() == 'y' else False
    else: exit()        

    #lossF = int(input("1. Cross entropy\n2. Lognorm\n:"))
    #lossFunction = nn.CrossEntropyLoss() if lossF == 1 else LogitNormLoss()
    lossFunction = LogitNormLoss()
    lossFunction = nn.CrossEntropyLoss()

    # 1 fast kan, 2 brsbf
    kanType = 2
    lrs = [1e-1,1e-2,1e-3,1e-4]
    lrs = [1e-3]
    lrsDenom = [1e-1,1e-2,1e-3,1e-4,1e-5]
    lrsDenom = [1e-4]
    lamdas = [0,0.0001,0.001,0.1]
    lamdas = [0]
    epochs = 80 
    archs = [[3,16,8,3],[3,32,16,3],[3,64,32,3]]
    archs = [[3,4,2,3]]
    gridsizes = [2,4,8,12]
    #gridsizes = [4]
    gammas = [0.1,0.25,0.5,1,2,4]#for rbfs
    gammas = [4]
    initDenoms = [0.1,0.25,0.5,1,1.5,2]
    initDenoms = [1.3] ###### check initializer at implementation 
    modelsNum = 10


    totalResults = []
    (kanTypeName := 'FastKAN') if kanType == 1 else (kanTypeName := 'BSRBF')
    print(kanTypeName)
    for arch in archs:
        print(f"Arch: {arch}")
        for gridsize in gridsizes:
            print(f"gridsize: {gridsize}")
            for lr in lrs:
                print(f"Lr: {lr}")
                for lrDenom in lrsDenom:
                    print(f"Denom lr: {lrDenom}")
                    for lamda in lamdas:
                        print(f"Lamda: {lamda}")
                        for gamma in gammas:
                            print(f"Gamma: {gamma}")
                            for initDenom in initDenoms:
                                print(f"init Denom: {initDenom}")
                                results = [KANcreateAndTrain(kanType, arch,lossFunction, gridsize=gridsize, lr=lr,
                                            lrDenom=lrDenom, initDenominator=initDenom, epochs=epochs, lamda=lamda,
                                            gamma=gamma, plot=plot) for _ in range(modelsNum)]
                                trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2,aurocs3 = zip(*results) # creates lists for each return variable
                                finalResult = [arch,gridsize,lr,lrDenom,initDenom,gamma, lamda,
                                                 np.mean(trainAccs),    np.std(trainAccs),
                                                 np.mean(trainLosses),  np.std(trainLosses),
                                                 np.mean(testAccs),     np.std(testAccs),
                                                 np.mean(testLosses),   np.std(testLosses),
                                                 np.mean(aurocs1),      np.std(aurocs1),
                                                 np.mean(aurocs2),      np.std(aurocs2),
                                                 np.mean(aurocs3),      np.std(aurocs3)
                                             ]# double list for save to excel implement
                                totalResults.append(finalResult)
                                print(f"Train Acc {100*finalResult[7]:>.1f}% std {100*finalResult[8]:>.1f}, AvgLoss {finalResult[9]:>.3f} std {finalResult[10]:>.3f}")
                                print(f"Test  Acc {100*finalResult[11]:>.1f}% std {100*finalResult[12]:>.1f}, AvgLoss {finalResult[13]:>.3f} std {finalResult[14]:>.3f}")
                                print(f"AUROC diab  {finalResult[15]:>.3f} std {finalResult[16]:>.3f}")
                                print(f"AUROC iris  {finalResult[17]:>.3f} std {finalResult[18]:>.3f}")
                                print(f"AUROC canc. {finalResult[19]:>.3f} std {finalResult[20]:>.3f}")


    if save:
        colNames = ['arch','gridsize','lr','lr denom','initDenom','gamma','lambda',
                    'trainAccs','train acc std', 'trainLosses','train loss std',
                    'testAccs', 'test acc std','testLosses','test loss std',
                    'auroc diab','auroc diab std','auroc iris','auroc iris std', 'auroc cancer','auroc cancer std']
        Path(f"results/KAN/{kanTypeName}/{runName}").mkdir(parents=True, exist_ok=True)
        saveToEXCEL(totalResults,colNames,f"results/KAN/{kanTypeName}/{runName}/results")


def prompttestKAN():
    runName     = str(input("Run name: "))
    kanTypeI    = int(input("1. FastKAN\n2. BSRBFKAN\n:"))
    kanType     = 'FastKAN' if kanTypeI == 1 else 'BSRBFKAN' 
    gridsize    = int(input("Gridsize: "))
    epochs      = int(input("Epochs: "))
    lr          = float(input("Learning rate: "))
    architectureI = input("Architecture (ex 3,16,32,3): ")
    architecture  = list(map(int, architectureI.split(',')))
    lamda       = float(input("Lamda: "))
    modelsNum   = int(input("Number of trained models: "))
    saveModel   = str(input("Save model(y,n): "))
    saveRes     = str(input("Save results(y,n): "))

    if saveModel.lower() == 'y': Path(f"../models/KAN/{kanType}/{runName}").mkdir(parents=True, exist_ok=True)
    if saveRes.lower()  ==  'y': Path(f"results/KAN/{kanType}/{runName}").mkdir(parents=True, exist_ok=True)

    # save run parameters
    with open(f"results/KAN/{kanType}/{runName}_run_parameters.txt", "w") as f:
        f.write(f"run name: {runName}\n")
        f.write(f"Epochs : {epochs}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Architecture: [{architectureI}]\n")
        f.write(f"Lamda: {lamda}\n")
        f.write(f"Models trained: {modelsNum}\n")


    results = [KANcreateAndTrain(kanTypeI, architecture, gridsize=gridsize, lr=lr, epochs=epochs, lamda=lamda, plot=False) for _ in range(modelsNum) ]
    trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2 = zip(*results) # creates lists for each return variable


    if saveRes.lower()  ==  'y': 

        colNames = ['arch','lambda','trainAccs','train acc std', 'trainLosses','train loss std',
                     'testAccs', 'test acc std','testLosses','test loss std',
                     'auroc diab','auroc diab std','auroc iris','auroc iris std']
        finalResult = [[architecture,lamda,
                         np.mean(trainAccs),    np.std(trainAccs),
                         np.mean(trainLosses),  np.std(trainLosses),
                         np.mean(testAccs),     np.std(testAccs),
                         np.mean(testLosses),   np.std(testLosses),
                         np.mean(aurocs1),      np.std(aurocs1),
                         np.mean(aurocs2),      np.std(aurocs2)
                     ]]# double list for save to excel implement

        saveToEXCEL(finalResult,colNames,f"results/KAN/{kanType}/{runName}")


testKAN()
#prompttestKAN()

#topResultPerformersFromEXCEL('results/KAN/FastKAN/rawtrain_0lamda_lr/results',
#                             'results/KAN/FastKAN/rawtrain_0lamda_lr/topResults',
#                             [(9,0.51)])

#topResultPerformersFromEXCEL('results/KAN/FastKAN/multirun_grid8/results',
#                             'results/KAN/FastKAN/multirun_grid8/highlamda_results',
#                             [(2,2)])