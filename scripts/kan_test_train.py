from models import *
from datasets import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from BSRBF_KAN import *
from Fast_KAN import *
from tools import *
from oodEvaluation import *
from tqdm import tqdm
from pathlib import Path
import torch.nn.utils.prune as prune
from prettytable import PrettyTable

from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def KANTrainStep(trainloader,num_classes,model,optimizer, lossFunction, l):
    model.train()
    total_loss = 0
    correct = 0
    for x, y in trainloader:

        x = x.to(device)
        y = y.to(device)
        y = F.one_hot(y,num_classes=num_classes).type(torch.float)
        #x.requires_grad_(True) # must for grad penalty

        optimizer.zero_grad()
        output = model(x)
        loss = lossFunction(output,y)
        #loss += l * gradPenalty2sideCalc(x,output)
        loss.backward()
        correct += (torch.argmax(output,dim=1) == torch.argmax(y,dim=1)).sum().item()
        total_loss += loss.item()
        optimizer.step()
    total_loss /= len(trainloader)
    accuracy = correct / len(trainloader.dataset)
    return accuracy, total_loss

def KANTest(testloader,model, lossFunction):
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

def KANTrain(trainloader,testloader,falseloader,num_classes,model,optimizer, scheduler, lossFunction, epochs,l):

    trainaccs = []
    testaccs = []
    trainLosses =[]
    testLosses = []
    auroc  = []
    lrs = []
    for _ in tqdm(range(epochs),desc="Epochs"):
        trainAcc, trainLoss = KANTrainStep(trainloader,num_classes,model,optimizer,lossFunction,l)
        testAcc, testLoss = KANTest(testloader,model,lossFunction)
        auroc.append(get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader.dataset, model=model, device=device, model_type='kan'))
        trainaccs.append(trainAcc)
        testaccs.append(testAcc)
        trainLosses.append(trainLoss)
        testLosses.append(testLoss)
        scheduler.step(testLoss)
        lrs.append(scheduler.get_last_lr())
    #print(lrs)

    return trainaccs,trainLosses,testaccs,testLosses,auroc,lrs

def KANcreateAndTrain(dataloaders,kanType,architecture,lossFunction, gridsize=8,lr=1e-3,lrDenom=1e-3,initDenominator=1.,
                      epochs=25,lamda=0,gamma=0.5,base_activ='silu',plot=False):
    trainloader,testloader,falseloader,falseloader2,falseloader3,falseloader4 = dataloaders
    model = None
    base_activation = ActivationFunctions(gamma).RBF_SiLU
    if base_activ.lower() == 'silu':
        base_activation = F.silu
    if kanType == 1:
        model = FastKAN(architecture,num_grids=gridsize, base_activation=base_activation,denominator=initDenominator).to(device)
    elif kanType == 2:
        model = BSRBF_KAN(architecture,gridsize,base_activation=base_activation,denominator=initDenominator).to(device)

    #count_parameters(model)

    denomParam = [p for name, p in model.named_parameters() if 'denominator' in name]
    othersParam = [p for name, p in model.named_parameters() if 'denominator' not in name]
    optimizer = torch.optim.SGD([{"params":othersParam},{"params":denomParam,"lr":lrDenom}],lr = lr)


    #optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min',patience=5)


    trainAccs, trainLosses, testAccs, testLosses,aurocProgress,lrProgress = KANTrain(trainloader,testloader,falseloader,architecture[-1],
    model,optimizer,scheduler,lossFunction,epochs,lamda)
    auroc2 = auroc3 = '-'
    auroc1 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader.dataset, model=model, device=device, model_type='kan')
    if falseloader2:
        auroc2 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader2.dataset, model=model, device=device, model_type='kan')
    if falseloader3:
        auroc3 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader3.dataset, model=model, device=device, model_type='kan')
    if falseloader4:
        auroc4 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader4.dataset, model=model, device=device, model_type='kan')

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
    
    return trainAccs[-1],trainLosses[-1],testAccs[-1],testLosses[-1],auroc1,auroc2,auroc3,auroc4


    ############

# train multiple models and save results
def testKAN():
    #dataloaders = [trainloader,testloader,falseloader,falseloader2,falseloader3]
    save = str(input("Save?(y/n):"))
    plot = False
    if save.lower() == 'y':
        save = True
        runName = str(input('Run name: '))
    elif save.lower() == 'n':
        save = False
        plot = str(input("plot?(y/n):"))
        plot = True if plot.lower() == 'y' else False

    #lossF = int(input("1. Cross entropy\n2. Lognorm\n:"))
    #lossFunction = nn.CrossEntropyLoss() if lossF == 1 else LogitNormLoss()
    lossFunction = LogitNormLoss()

    # 1 fast kan, 2 brsbf
    kanType = 1
    lrs = [1e-1,1e-2,1e-3,1e-4]
    lrs = [1e-3]
    lrsDenom = [1e-1,1e-2,1e-3,1e-4,1e-5]
    lrsDenom = [1e-1]
    lamdas = [0,0.0001,0.001,0.1]
    lamdas = ['-']
    epochs = 100
    #archs = [[3,64,32,3]]
    archs = [[353,16,8,9],[353,16,16,9],[353,32,16,9],[353,64,32,9]]
    archs = [[353,32,9]]
    archs = [[3,16,3],[3,32,3],[3,64,3]]
    archs = [[3,128,3]]
    gridsizes = [4,8,12,16]
    gridsizes = [4]
    gammas = [0.001,0.1,0.25,0.5,1,2,4,8]#for rbfs
    #gammas = [4,6,8,10]#for rbfs
    gammas = [4]
    #gammas = ['-']
    initDenoms = [0.1,0.25,0.5,1,1.5,2]
    #initDenoms = [1.3] ###### check initializer at implementation 
    initDenoms = [1]
    base_activation = 'rbf-silu'
    modelsNum = 5

    dataloaders = loadAllDataloaders(binary=True if archs[0][-1]==2 else False)
    #dataloaders = loadImageDataloaders()
    #dataloaders = loadAllDataloaders(binary=False)

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
                                results = [KANcreateAndTrain(dataloaders,kanType, arch,lossFunction, gridsize=gridsize, lr=lr,
                                            lrDenom=lrDenom, initDenominator=initDenom, base_activ=base_activation,epochs=epochs, lamda=lamda,
                                            gamma=gamma, plot=plot) for _ in range(modelsNum)]
                                trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2,aurocs3 = zip(*results) # creates lists for each return variable
                                finalResult = [arch,gridsize,lr,lrDenom,initDenom,gamma, lamda,
                                                 np.mean(trainAccs),    np.std(trainAccs),
                                                 np.mean(trainLosses),  np.std(trainLosses),
                                                 np.mean(testAccs),     np.std(testAccs),
                                                 np.mean(testLosses),   np.std(testLosses),
                                                 np.mean(aurocs1),      np.std(aurocs1),
                                                        ]# double list for save to excel implement
                                if dataloaders[3]: finalResult.extend([np.mean(aurocs2),np.std(aurocs2)])
                                if dataloaders[4]: finalResult.extend([np.mean(aurocs3),np.std(aurocs3)])
                                totalResults.append(finalResult)
                                print(f"Train Acc {100*finalResult[7]:>.1f}% std {100*finalResult[8]:>.1f}, AvgLoss {finalResult[9]:>.3f} std {finalResult[10]:>.3f}")
                                print(f"Test  Acc {100*finalResult[11]:>.1f}% std {100*finalResult[12]:>.1f}, AvgLoss {finalResult[13]:>.3f} std {finalResult[14]:>.3f}")
                                print(f"AUROC 1 {finalResult[15]:>.3f} std {finalResult[16]:>.3f}")
                                if dataloaders[3]:
                                    print(f"AUROC 2  {finalResult[17]:>.3f} std {finalResult[18]:>.3f}")
                                if dataloaders[4]:
                                    print(f"AUROC 3 {finalResult[19]:>.3f} std {finalResult[20]:>.3f}")


    if save:
        colNames = ['arch','gridsize','lr','lr denom','initDenom','gamma','lambda',
                    'trainAccs','train acc std', 'trainLosses','train loss std',
                    'testAccs', 'test acc std','testLosses','test loss std',
                    'auroc 1','auroc 1 std']
        if dataloaders[3]:
            colNames.extend(['auroc 2','auroc 2 std'])
        if dataloaders[4]:
            colNames.extend(['auroc 3','auroc 3 std'])
        Path(f"results/KAN/{kanTypeName}/{runName}").mkdir(parents=True, exist_ok=True)
        saveToEXCEL(totalResults,colNames,f"results/KAN/{kanTypeName}/{runName}/results")


if __name__ == "__main__":
    testKAN()
    
    #topResultPerformersFromEXCEL('results/KAN/BSRBF/gammas_denoms_16_8/results',
    #                             'results/KAN/BSRBF/gammas_denoms_16_8/best_results',
    #                             [(11,0.5),(15,0.65),(17,0.65),(19,.65)])