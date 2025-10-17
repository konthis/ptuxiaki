# python3 -m scripts.proposed_method.test
# exec like that

from torch.optim.lr_scheduler import ReduceLROnPlateau
from scripts.proposed_method.models import *
#from scripts.Fast_KAN import *
from scripts.datasets import *
from scripts.proposed_method.train import *
from scripts.functions import *


from pathlib import Path
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def DUQKANTest():
    save = int(input('Save?(0/1):'))
    if save:
        runName = str(input("Run name: "))
    root = 'datasets'
    #testDataset, trainLoader, testLoader, dimensions = load_D1(2,root)
    #falseLoader = load_noisedD1(2,False,root)
    trainLoader,testLoader,falseLoader1,falseLoader2,falseLoader3,falseLoader4 = loadAllDataloaders(root,False)
    falseloaders = [falseLoader1,falseLoader2,falseLoader3,falseLoader4]

    lossFunction = LogitNormLoss() #conf
    #lossFunction = nn.CrossEntropyLoss()
    netType = 'kanduq'
    #netType = 'duq'
    numClasses = 3
    gradPenaltyL = [0.0]
    #gradPenaltyL = [0,.001,.01,.1,.3,.5]
    epochs = 100
    numTestModels = 5

    lrs = [0.001] #conf
    #lrs = [0.0001,0.001,0.01,0.1]
    wd = 1e-4

    numGrids = [4,8,10,12]
    numGrids = [4] #conf

    lengthScales = [0.1,0.5,1,2,5]
    lengthScales = [1] #conf

    #2l
    #archs = [[3,4,3],[3,8,3],[3,16,3]]
    #3l
    archs = [[3,16,3],[3,32,3],[3,64,3],[3,8,4,3],[3,16,8,3],[3,32,16,3],[3,16,16,16,3],
             [3,16,16,32,3],[3,16,32,16,3],[3,32,16,16,3],[3,16,32,64,3]]
    archs = [[3,16,16,16,3]] # conf
    #archs = [[3,128,3]]

    gammas = [0.01,0.1,0.25,0.5,0.75]
    gammas = [0.15,0.2]
    #gammas = [0.25] #conf

    totalResults = []

    for ar in archs:
        print(f"Architecture: {ar}")
        for lr in lrs:
            print(f"lr: {lr}")
            for gp in gradPenaltyL:
                print(f"gp: {gp}")
                for grids in numGrids:
                    print(f"Num grids: {grids}")
                    for ls in lengthScales:
                        print(f"Length scale: {ls}")
                        for gamma in gammas:
                            print(f"Activation function gamma: {gamma}")
                            baseActivation = ActivationFunctions(gamma).RBF_SiLU
                            #baseActivation = ActivationFunctions(gamma).GaussianRBF
                            #baseActivation = ActivationFunctions(gamma).RBF_Swish
                            trainAccs = []
                            trainLosses = []
                            testAccs = []
                            testLosses = []
                            aurocs = []
                            for _ in range(numTestModels):
                                featureExtractor = FastKAN(ar[:-1],num_grids=grids,base_activation=baseActivation).to(device)
                                model = KANDUQ(featureExtractor,numClasses,ar[-2],ar[-2],length_scale=ls,gamma=0.999).to(device)
                                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
                                scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
                                trainAcc,trainLoss,testAcc,testLoss,auroc = networkTrain(netType,model,optimizer,scheduler,lossFunction,trainLoader,testLoader,falseloaders,numClasses,gp,epochs)
                                trainAccs.append(trainAcc)
                                trainLosses.append(trainLoss)
                                testAccs.append(testAcc)
                                testLosses.append(testLoss)
                                aurocs.append(auroc)
                            print(f"TrainAcc:{np.mean(trainAccs):>.3f} std {np.std(trainAccs):>.3f}, TrainLoss:{np.mean(trainLosses):>.3f} std {np.std(trainLosses):>.3f}")
                            print(f"TestAcc:{np.mean(testAccs):>.3f} std {np.std(testAccs):>.3f}, TrainLoss:{np.mean(testLosses):>.3f} std {np.std(testLosses):>.3f}")
                            aurocsR = np.mean(aurocs,axis=0)
                            aurocsStd = np.std(aurocs,axis=0)
                            results = [ar,lr,gp,grids,ls,gamma,np.mean(trainAccs),np.std(trainAccs),
                                       np.mean(trainLosses),np.std(trainLosses),
                                       np.mean(testAccs),np.std(testAccs),
                                       np.mean(testLosses),np.std(testLosses)]
                            for i in range(len(aurocsR)):
                                print(f"AUROC {i+1}: {aurocsR[i]:>.3f} std {aurocsStd[i]:>.3f}")
                                results.append(aurocsR[i])
                                results.append(aurocsStd[i])
                            totalResults.append(results)
                            
    if save:
        colNames = ['arch','lr','gp','grids','Length Scale','gamma',
                    'trainAccs','train acc std', 'trainLosses','train loss std',
                    'testAccs', 'test acc std','testLosses','test loss std',
                    'auroc 1','auroc 1 std']
        if falseloaders[1]: colNames.extend(['auroc 2','auroc 2 std'])
        if falseloaders[2]: colNames.extend(['auroc 3','auroc 3 std'])
        if falseloaders[3]: colNames.extend(['auroc 4','auroc 4 std'])
        Path(f"scripts/proposed_method/results").mkdir(parents=True, exist_ok=True)
        saveToEXCEL(totalResults,colNames,f"scripts/proposed_method/results/{runName}")


DUQKANTest()