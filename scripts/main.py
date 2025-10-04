from datasets import *
from models import *
from duq_mlp_emb import *
from kan_test_train import *
from BSRBF_KAN import *
from Fast_KAN import *
from tools import *
from oodEvaluation import *
from pathlib import Path
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def chooseModelTrain():
    save = str(input("Save?(y/n):"))
    imgTrain = str(input("Image train?(PCA)(y/n):"))
    print("1. Train DUQ")
    print("2. Train MLP")
    print("3. Train n-ensemble")
    print("4. Train KAN")
    print("0. Exit")
    netType = int(input(':'))
    runName = None
    if save.lower() == 'y':
        runName = str(input("Run name: "))

    if netType == 1:
        netType == 'DUQ'
        architectureI = input("Architecture (ex 3,32,16,3): ")
        architecture  = list(map(int, architectureI.split(',')))
        epochs      = int(input("Epochs: "))
        lr          = float(input("Learning rate: "))
        sigma       = float(input("Sigma: "))
        lrSigma     = float(input("Sigma lr: "))
        lamda       = float(input("Lamda: "))
        std         = float(input("Std: "))
        modelsNum   = int(input("How many models per config: "))
        #architecture  = [3,32,16,2]
        #epochs      = 120
        #lr          = 1e-1
        #sigma       = 0.5
        #lrSigma     = 1e-3
        #lamda       = 0.2
        #std         = 1e-3
        #modelsNum   = 5

        if imgTrain.lower() == 'y':
            trainloader,testloader,falseloader,falseloader2,falseloader3 = loadImageDataloaders()
        else:
            trainloader,testloader,falseloader,falseloader2,falseloader3,falseloader4 = loadAllDataloaders(binary=True if architecture[-1]==2 else False)

        results = [DUQcreateAndTrain(trainloader,testloader,falseloader,falseloader2,falseloader3, falseloader4,
                                    architecture,std,sigma,lr,lrSigma,lamda,epochs) for _ in range(modelsNum)]
        trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2,aurocs3,aurocs4 = zip(*results) # creates lists for each return variable
        finalResult = [architecture,lr,lamda, std,
                          np.mean(trainAccs),    np.std(trainAccs),
                          np.mean(trainLosses),  np.std(trainLosses),
                          np.mean(testAccs),     np.std(testAccs),
                          np.mean(testLosses),   np.std(testLosses),
                          np.mean(aurocs1),      np.std(aurocs1),
                        ]# double list for save to excel implement
        if falseloader2: finalResult.extend([np.mean(aurocs2),np.std(aurocs2)])
        if falseloader3: finalResult.extend([np.mean(aurocs3),np.std(aurocs3)])
        if falseloader4: finalResult.extend([np.mean(aurocs4),np.std(aurocs4)])
        print(f"Train Acc {100*finalResult[4]:>.2f}% std {100*finalResult[5]:>.2f}, AvgLoss {finalResult[6]:>.3f} std {finalResult[7]:>.3f}")
        print(f"Test  Acc {100*finalResult[8]:>.2f}% std {100*finalResult[9]:>.2f}, AvgLoss {finalResult[10]:>.3f} std {finalResult[11]:>.3f}")
        print(f"AUROC 1 {finalResult[12]:>.3f} std {finalResult[13]:>.3f}")
        if falseloader2:
            print(f"AUROC 2 {finalResult[14]:>.3f} std {finalResult[15]:>.3f}")
        if falseloader3:
            print(f"AUROC 3 {finalResult[16]:>.3f} std {finalResult[17]:>.3f}")
        if falseloader4:
            print(f"AUROC 4 {finalResult[18]:>.3f} std {finalResult[19]:>.3f}")
        if save:
            colNames = ['arch','lr','lamda','std',
                        'trainAccs','train acc std', 'trainLosses','train loss std',
                        'testAccs', 'test acc std','testLosses','test loss std',
                        'auroc 1','auroc 1 std']
            if falseloader2:
                colNames.extend(['auroc 2','auroc 2 std'])
            if falseloader3:
                colNames.extend(['auroc 3','auroc 3 std'])
            if falseloader4:
                colNames.extend(['auroc 4','auroc 4 std'])
            Path(f"results/DUQ/{runName}").mkdir(parents=True, exist_ok=True)
            saveToEXCEL([finalResult],colNames,f"results/DUQ/{runName}/results")

    elif netType == 2:
        netType = 'MLP'

        architectureI = input("Architecture (ex 3,32,16,3): ")
        architecture  = list(map(int, architectureI.split(',')))
        epochs      = int(input("Epochs: "))
        lr          = float(input("Learning rate: "))
        modelsNum   = int(input("How many models per config: "))

        #architecture  = [3,32,16,2]
        #epochs      = 120
        #lr          = 1e-1
        #modelsNum   = 5

        if imgTrain.lower() == 'y':
            trainloader,testloader,falseloader,falseloader2,falseloader3 = loadImageDataloaders()
        else:
            #trainloader,testloader,falseloader,falseloader2,falseloader3 = loadAllDataloaders(binary=True if architecture[-1]==2 else False)
            trainloader,testloader,falseloader,falseloader2,falseloader3,falseloader4 = loadAllDataloaders(binary=True if architecture[-1]==2 else False)

        results = [MLPcreateAndTrain(trainloader,testloader,falseloader,falseloader2,falseloader3,falseloader4,
                                    architecture,lr,epochs) for _ in range(modelsNum)]
        trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2,aurocs3,aurocs4 = zip(*results) # creates lists for each return variable
        finalResult = [architecture,lr,
                          np.mean(trainAccs),    np.std(trainAccs),
                          np.mean(trainLosses),  np.std(trainLosses),
                          np.mean(testAccs),     np.std(testAccs),
                          np.mean(testLosses),   np.std(testLosses),
                          np.mean(aurocs1),      np.std(aurocs1),
                          np.mean(aurocs2),      np.std(aurocs2),
                          np.mean(aurocs3),      np.std(aurocs3),
                          np.mean(aurocs4),      np.std(aurocs4)
                      ]# double list for save to excel implement
        print(f"Train Acc {100*finalResult[2]:>.2f}% std {100*finalResult[3]:>.2f}, AvgLoss {finalResult[4]:>.3f} std {finalResult[5]:>.3f}")
        print(f"Test  Acc {100*finalResult[6]:>.2f}% std {100*finalResult[7]:>.2f}, AvgLoss {finalResult[8]:>.3f} std {finalResult[9]:>.3f}")
        print(f"AUROC 1 {finalResult[10]:>.3f} std {finalResult[11]:>.3f}")
        print(f"AUROC 2 {finalResult[12]:>.3f} std {finalResult[13]:>.3f}")
        print(f"AUROC 3 {finalResult[14]:>.3f} std {finalResult[15]:>.3f}")
        print(f"AUROC 4 {finalResult[16]:>.3f} std {finalResult[17]:>.3f}")

        if save:
            colNames = ['arch','lr',
                        'trainAccs','train acc std', 'trainLosses','train loss std',
                        'testAccs', 'test acc std','testLosses','test loss std',
                        'auroc 1','auroc 1 std','auroc 2','auroc 2 std', 'auroc 3','auroc 3 std','auroc 4','auroc 4 std']
            Path(f"results/MLP/{runName}").mkdir(parents=True, exist_ok=True)
            saveToEXCEL([finalResult],colNames,f"results/MLP/{runName}/results")

    elif netType == 3:
        netType = 'DE'

        architectureI = input("Architecture (ex 3,32,16,3): ")
        architecture  = list(map(int, architectureI.split(',')))
        epochs      = int(input("Epochs: "))
        lr          = float(input("Learning rate: "))
        modelsNumEnsemble = int(input("Ensemble modelsize: "))
        modelsNum   = int(input("How many models per config: "))

        #architecture  = [3,32,16,2]
        #epochs      = 120
        #lr          = 1e-1
        #modelsNumEnsemble = 5
        #modelsNum   = 5

        if imgTrain.lower() == 'y':
            trainloader,testloader,falseloader,falseloader2,falseloader3 = loadImageDataloaders()
        else:
            trainloader,testloader,falseloader,falseloader2,falseloader3,falseloader4 = loadAllDataloaders(binary=True if architecture[-1]==2 else False)

        results = [DEcreateAndTrain(trainloader,testloader,falseloader,falseloader2,falseloader3,falseloader4,
                                    modelsNumEnsemble,architecture,lr,epochs) for _ in range(modelsNum)]
        trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2,aurocs3,aurocs4 = zip(*results) # creates lists for each return variable
        finalResult = [architecture,lr,
                          np.mean(trainAccs),    np.std(trainAccs),
                          np.mean(trainLosses),  np.std(trainLosses),
                          np.mean(testAccs),     np.std(testAccs),
                          np.mean(testLosses),   np.std(testLosses),
                          np.mean(aurocs1),      np.std(aurocs1),
                          np.mean(aurocs2),      np.std(aurocs2),
                          np.mean(aurocs3),      np.std(aurocs3),
                          np.mean(aurocs4),      np.std(aurocs4),
                      ]# double list for save to excel implement
        print(f"Train Acc {100*finalResult[2]:>.2f}% std {100*finalResult[3]:>.2f}, AvgLoss {finalResult[4]:>.3f} std {finalResult[5]:>.3f}")
        print(f"Test  Acc {100*finalResult[6]:>.2f}% std {100*finalResult[7]:>.2f}, AvgLoss {finalResult[8]:>.3f} std {finalResult[9]:>.3f}")
        print(f"AUROC 1 {finalResult[10]:>.3f} std {finalResult[11]:>.3f}")
        print(f"AUROC 2 {finalResult[12]:>.3f} std {finalResult[13]:>.3f}")
        print(f"AUROC 3 {finalResult[14]:>.3f} std {finalResult[15]:>.3f}")
        print(f"AUROC 4 {finalResult[16]:>.3f} std {finalResult[17]:>.3f}")

        if save:
            colNames = ['arch','lr',
                        'trainAccs','train acc std', 'trainLosses','train loss std',
                        'testAccs', 'test acc std','testLosses','test loss std',
                        'auroc 1','auroc 1 std','auroc 2','auroc 2 std', 'auroc 3','auroc 3 std', 'auroc 4','auroc 4 std']
            Path(f"results/DE/{runName}").mkdir(parents=True, exist_ok=True)
            saveToEXCEL([finalResult],colNames,f"results/DE/{runName}/results")

    elif netType == 4:
        netType == 'KAN'
        architectureI = input("Architecture (ex 3,32,16,3): ")
        architecture  = list(map(int, architectureI.split(',')))
        gridsize    = int(input("Gridsize: "))
        actF        = int(input("1.SiLU\n2.RBF SiLU\n:"))
        kanTypeI    = int(input("1. FastKAN\n2. BSRBFKAN\n:"))
        kanType     = 'FastKAN' if kanTypeI == 1 else 'BSRBF' 
        #architecture  = [3,64,32,2]
        #gridsize     = 8
        #actF        = 2
        gamma       = None
        #gamma = 1
        #epochs = 65
        #lr = 1e-3
        #initDenom = 1
        #lrDenom = 1e-4
        #modelsNum = 5
        if actF == 1:
            actF = 'silu'
        else:
            actF = 'rbf_silu'
            gamma   = float(input("Gamma: "))

        lossF       = int(input("1.Crossentropy loss\n2.LogNormloss\n:"))
        lossFunction = LogitNormLoss()
        if lossF == 1:
            lossFunction = nn.CrossEntropyLoss()
        epochs      = int(input("Epochs: "))
        lr          = float(input("Learning rate: "))
        initDenom   = float(input("Initial Denominator: "))
        lrDenom     = float(input("Learning rate of denominator: "))
        #lamda       = float(input("Lamda: "))
        lamda       = '-'
        modelsNum   = int(input("Number of trained models: "))

        if imgTrain.lower() == 'y':
            dataloaders = loadImageDataloaders()
        else:
            dataloaders = loadAllDataloaders(binary=True if architecture[-1]==2 else False)

        results = [KANcreateAndTrain(dataloaders,kanTypeI, architecture,lossFunction, gridsize=gridsize, lr=lr,
                    lrDenom=lrDenom, initDenominator=initDenom, epochs=epochs, lamda=lamda,
                    gamma=gamma,base_activ=actF, plot=False) for _ in range(modelsNum)]
        trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2,aurocs3,aurocs4 = zip(*results) # creates lists for each return variable
        finalResult = [architecture,gridsize,lr,lrDenom,initDenom,gamma, lamda,
                         np.mean(trainAccs),    np.std(trainAccs),
                         np.mean(trainLosses),  np.std(trainLosses),
                         np.mean(testAccs),     np.std(testAccs),
                         np.mean(testLosses),   np.std(testLosses),
                         np.mean(aurocs1),      np.std(aurocs1),
                         np.mean(aurocs2),      np.std(aurocs2),
                         np.mean(aurocs3),      np.std(aurocs3),
                         np.mean(aurocs4),      np.std(aurocs4)
                     ]# double list for save to excel implement
        print(f"Train Acc {100*finalResult[7]:>.2f}% std {100*finalResult[8]:>.2f}, AvgLoss {finalResult[9]:>.3f} std {finalResult[10]:>.3f}")
        print(f"Test  Acc {100*finalResult[11]:>.2f}% std {100*finalResult[12]:>.2f}, AvgLoss {finalResult[13]:>.3f} std {finalResult[14]:>.3f}")
        print(f"AUROC 1  {finalResult[15]:>.3f} std {finalResult[16]:>.3f}")
        print(f"AUROC 2  {finalResult[17]:>.3f} std {finalResult[18]:>.3f}")
        print(f"AUROC 3 {finalResult[19]:>.3f} std {finalResult[20]:>.3f}")
        print(f"AUROC 4 {finalResult[21]:>.3f} std {finalResult[22]:>.3f}")

        if save:
            colNames = ['arch','gridsize','lr','lr denom','initDenom','gamma','lambda',
                        'trainAccs','train acc std', 'trainLosses','train loss std',
                        'testAccs', 'test acc std','testLosses','test loss std',
                        'auroc 1','auroc 1 std','auroc 2','auroc 2 std', 'auroc 3','auroc 3 std','auroc 4','auroc 4 std']
            Path(f"results/KAN/{kanType}/{runName}").mkdir(parents=True, exist_ok=True)
            saveToEXCEL([finalResult],colNames,f"results/KAN/{kanType}/{runName}/results")

    else:
        exit()

chooseModelTrain()