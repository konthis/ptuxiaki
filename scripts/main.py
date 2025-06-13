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

        trainloader,testloader,falseloader,falseloader2,falseloader3 = loadAllDataloaders(binary=True if architecture[-1]==2 else False)

        results = [DUQcreateAndTrain(trainloader,testloader,falseloader,falseloader2,falseloader3,
                                    architecture,std,sigma,lr,lrSigma,lamda,epochs) for _ in range(modelsNum)]
        trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2,aurocs3 = zip(*results) # creates lists for each return variable
        finalResult = [architecture,lr,lamda, std,
                          np.mean(trainAccs),    np.std(trainAccs),
                          np.mean(trainLosses),  np.std(trainLosses),
                          np.mean(testAccs),     np.std(testAccs),
                          np.mean(testLosses),   np.std(testLosses),
                          np.mean(aurocs1),      np.std(aurocs1),
                          np.mean(aurocs2),      np.std(aurocs2),
                          np.mean(aurocs3),      np.std(aurocs3)
                      ]# double list for save to excel implement
        print(f"Train Acc {100*finalResult[4]:>.1f}% std {100*finalResult[5]:>.1f}, AvgLoss {finalResult[6]:>.3f} std {finalResult[7]:>.3f}")
        print(f"Test  Acc {100*finalResult[8]:>.1f}% std {100*finalResult[9]:>.1f}, AvgLoss {finalResult[10]:>.3f} std {finalResult[11]:>.3f}")
        print(f"AUROC wine  {finalResult[12]:>.3f} std {finalResult[13]:>.3f}")
        print(f"AUROC iris  {finalResult[14]:>.3f} std {finalResult[15]:>.3f}")
        print(f"AUROC canc. {finalResult[16]:>.3f} std {finalResult[17]:>.3f}")
        if save:
            colNames = ['arch','lr','lamda','std',
                        'trainAccs','train acc std', 'trainLosses','train loss std',
                        'testAccs', 'test acc std','testLosses','test loss std',
                        'auroc wine','auroc wine std','auroc iris','auroc iris std', 'auroc cancer','auroc cancer std']
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
        trainloader,testloader,falseloader,falseloader2,falseloader3 = loadAllDataloaders(binary=True if architecture[-1]==2 else False)

        results = [MLPcreateAndTrain(trainloader,testloader,falseloader,falseloader2,falseloader3,
                                    architecture,lr,epochs) for _ in range(modelsNum)]
        trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2,aurocs3 = zip(*results) # creates lists for each return variable
        finalResult = [architecture,lr,
                          np.mean(trainAccs),    np.std(trainAccs),
                          np.mean(trainLosses),  np.std(trainLosses),
                          np.mean(testAccs),     np.std(testAccs),
                          np.mean(testLosses),   np.std(testLosses),
                          np.mean(aurocs1),      np.std(aurocs1),
                          np.mean(aurocs2),      np.std(aurocs2),
                          np.mean(aurocs3),      np.std(aurocs3)
                      ]# double list for save to excel implement
        print(f"Train Acc {100*finalResult[2]:>.1f}% std {100*finalResult[3]:>.1f}, AvgLoss {finalResult[4]:>.3f} std {finalResult[5]:>.3f}")
        print(f"Test  Acc {100*finalResult[6]:>.1f}% std {100*finalResult[7]:>.1f}, AvgLoss {finalResult[8]:>.3f} std {finalResult[9]:>.3f}")
        print(f"AUROC wine  {finalResult[10]:>.3f} std {finalResult[11]:>.3f}")
        print(f"AUROC iris  {finalResult[12]:>.3f} std {finalResult[13]:>.3f}")
        print(f"AUROC canc. {finalResult[14]:>.3f} std {finalResult[15]:>.3f}")

        if save:
            colNames = ['arch','lr',
                        'trainAccs','train acc std', 'trainLosses','train loss std',
                        'testAccs', 'test acc std','testLosses','test loss std',
                        'auroc wine','auroc wine std','auroc iris','auroc iris std', 'auroc cancer','auroc cancer std']
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

        trainloader,testloader,falseloader,falseloader2,falseloader3 = loadAllDataloaders(binary=True if architecture[-1]==2 else False)

        results = [DEcreateAndTrain(trainloader,testloader,falseloader,falseloader2,falseloader3,
                                    modelsNumEnsemble,architecture,lr,epochs) for _ in range(modelsNum)]
        trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2,aurocs3 = zip(*results) # creates lists for each return variable
        finalResult = [architecture,lr,
                          np.mean(trainAccs),    np.std(trainAccs),
                          np.mean(trainLosses),  np.std(trainLosses),
                          np.mean(testAccs),     np.std(testAccs),
                          np.mean(testLosses),   np.std(testLosses),
                          np.mean(aurocs1),      np.std(aurocs1),
                          np.mean(aurocs2),      np.std(aurocs2),
                          np.mean(aurocs3),      np.std(aurocs3)
                      ]# double list for save to excel implement
        print(f"Train Acc {100*finalResult[2]:>.1f}% std {100*finalResult[3]:>.1f}, AvgLoss {finalResult[4]:>.3f} std {finalResult[5]:>.3f}")
        print(f"Test  Acc {100*finalResult[6]:>.1f}% std {100*finalResult[7]:>.1f}, AvgLoss {finalResult[8]:>.3f} std {finalResult[9]:>.3f}")
        print(f"AUROC wine {finalResult[10]:>.3f} std {finalResult[11]:>.3f}")
        print(f"AUROC iris  {finalResult[12]:>.3f} std {finalResult[13]:>.3f}")
        print(f"AUROC canc. {finalResult[14]:>.3f} std {finalResult[15]:>.3f}")

        if save:
            colNames = ['arch','lr',
                        'trainAccs','train acc std', 'trainLosses','train loss std',
                        'testAccs', 'test acc std','testLosses','test loss std',
                        'auroc wine','auroc wine std','auroc iris','auroc iris std', 'auroc cancer','auroc cancer std']
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

        dataloaders = loadAllDataloaders(binary=True if architecture[-1]==2 else False)
        results = [KANcreateAndTrain(dataloaders,kanTypeI, architecture,lossFunction, gridsize=gridsize, lr=lr,
                    lrDenom=lrDenom, initDenominator=initDenom, epochs=epochs, lamda=lamda,
                    gamma=gamma,base_activ=actF, plot=False) for _ in range(modelsNum)]
        trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2,aurocs3 = zip(*results) # creates lists for each return variable
        finalResult = [architecture,gridsize,lr,lrDenom,initDenom,gamma, lamda,
                         np.mean(trainAccs),    np.std(trainAccs),
                         np.mean(trainLosses),  np.std(trainLosses),
                         np.mean(testAccs),     np.std(testAccs),
                         np.mean(testLosses),   np.std(testLosses),
                         np.mean(aurocs1),      np.std(aurocs1),
                         np.mean(aurocs2),      np.std(aurocs2),
                         np.mean(aurocs3),      np.std(aurocs3)
                     ]# double list for save to excel implement
        print(f"Train Acc {100*finalResult[7]:>.1f}% std {100*finalResult[8]:>.1f}, AvgLoss {finalResult[9]:>.3f} std {finalResult[10]:>.3f}")
        print(f"Test  Acc {100*finalResult[11]:>.1f}% std {100*finalResult[12]:>.1f}, AvgLoss {finalResult[13]:>.3f} std {finalResult[14]:>.3f}")
        print(f"AUROC wine  {finalResult[15]:>.3f} std {finalResult[16]:>.3f}")
        print(f"AUROC iris  {finalResult[17]:>.3f} std {finalResult[18]:>.3f}")
        print(f"AUROC canc. {finalResult[19]:>.3f} std {finalResult[20]:>.3f}")

        if save:
            colNames = ['arch','gridsize','lr','lr denom','initDenom','gamma','lambda',
                        'trainAccs','train acc std', 'trainLosses','train loss std',
                        'testAccs', 'test acc std','testLosses','test loss std',
                        'auroc wine','auroc wine std','auroc iris','auroc iris std', 'auroc cancer','auroc cancer std']
            Path(f"results/KAN/{kanType}/{runName}").mkdir(parents=True, exist_ok=True)
            saveToEXCEL([finalResult],colNames,f"results/KAN/{kanType}/{runName}/results")

    else:
        exit()

chooseModelTrain()