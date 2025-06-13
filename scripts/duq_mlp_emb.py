from datasets import *
from models import *
from functions import *
from BSRBF_KAN import *
from tools import *
from oodEvaluation import *
from pathlib import Path
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def softmaxTrainStep(dataloader, model, lossfn, optimizer):
    model.train()
    trloss,correct = 0, 0 
    for X,y in dataloader:
        X = X.to(device)
        y = y.to(device)
        pred = model.forward(X)
        loss = lossfn(pred, y)
        trloss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        correct += (torch.argmax(pred,dim=1) == y).sum().item()
    accuracy = correct / len(dataloader.dataset)
    avgLoss = trloss/len(dataloader)
    return accuracy, avgLoss

def softmaxTest(dataloader, model):
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model.forward(X)
            test_loss += F.cross_entropy(pred, y).item()
            correct += (torch.argmax(pred,dim=1) == y).type(torch.float).sum().item()
    accuracy = correct / len(dataloader.dataset)
    avgLoss = test_loss/len(dataloader)
    return accuracy, avgLoss

def softmaxTrain(trainloader, testloader,falseloader, model, optimizer, scheduler,epochs):
    lrs = []
    auroc = []
    for e in tqdm(range(epochs),desc="Epochs"):
        trainAcc, trainAvgLoss = softmaxTrainStep(trainloader,model,nn.CrossEntropyLoss(),optimizer)
        testAcc, testAvgLoss= softmaxTest(testloader,model)

        auroc.append(get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader.dataset, model=model, device=device, model_type='mlp'))
        scheduler.step(testAvgLoss) ########
        lrs.append(scheduler.get_last_lr())
    return trainAcc, trainAvgLoss, testAcc, testAvgLoss, auroc,lrs

def softmaxForward(model, dataloader):
    model.eval()
    pred = []
    with torch.no_grad():
        for X,y in dataloader:
            X = X.to(device)
            pred.append(model.forward(X).cpu().numpy())
    return np.vstack(pred)

def deepEnsembleForward(models, dataloader):
    predictions = []
    for model in models:
        predictions.append(softmaxForward(model,dataloader))
    return np.mean(predictions,axis=0) # merging results

def duqTrainStep(dataloader,num_classes, model, optimizer, l):
    totalloss = 0
    correct = 0
    for x,y in dataloader:
        y = F.one_hot(y,num_classes=num_classes).type(torch.float)
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
        totalloss += loss.item()
        optimizer.step()
        with torch.no_grad():
            model.updateCentroids(x,y)
    accuracy = correct / len(dataloader.dataset)
    avgLoss = totalloss/len(dataloader)
    return accuracy, avgLoss

def duqTrain(trainloader, testloader, falseloader,num_classes, model, optimizer,scheduler, l, epochs):
    trainLosses = []
    testLosses  = []
    auroc = []
    lrs = []
    for _ in tqdm(range(epochs),desc="Epochs"):

        trainAcc, trainAvgLoss,= duqTrainStep(trainloader,num_classes, model, optimizer, l)
        testAcc, testAvgLoss = duqTest(testloader,num_classes, model)
        trainLosses.append(trainAvgLoss) 
        testLosses.append(testAvgLoss) 
        auroc.append(get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader.dataset, model=model, device=device, model_type='duq'))

        scheduler.step(testAvgLoss)
        lrs.append(scheduler.get_last_lr())
    #print(scheduler.get_last_lr())

    #plot([i for i in range(epochs)],'epochs',ylosses,'loss','r')
    return trainAcc, trainAvgLoss, trainLosses, testAcc, testAvgLoss, testLosses,auroc, lrs

def duqTest(dataloader,num_classes, model):
    model.eval()
    testLoss, correct = 0, 0
    totalCert = 0
    with torch.no_grad():
        for x, y in dataloader:
            y = F.one_hot(y,num_classes=num_classes).type(torch.float)
            x = x.to(device)
            y = y.to(device)
            ypred,z = model.forward(x)
            totalCert += torch.max(ypred,dim=1)[0].mean()
            testLoss += F.cross_entropy(ypred, y).item()
            correct += (torch.argmax(ypred,dim=1) == torch.argmax(y,dim=1)).sum().item()
    accuracy = correct / len(dataloader.dataset)
    avgLoss = testLoss/len(dataloader)
    return accuracy, avgLoss 

def DUQcreateAndTrain(trainloader,testloader,falseloader,falseloader2,falseloader3,arch,std,initSigma,lr,lrSigma,l,epochs):
    model = SimpleLinearNetwork_DUQ(arch[0],arch[1],arch[2],arch[3],std,initSigma=initSigma).to(device)
    num_classes = arch[-1]
    sigmaParam = [p for name, p in model.named_parameters() if 'sigma' in name]
    othersParam = [p for name, p in model.named_parameters() if 'sigma' not in name]
    optimizer = torch.optim.SGD([{"params":othersParam},{"params":sigmaParam,"lr":lrSigma}],lr = lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min',patience=5)
    trainAcc, trainAvgLoss, trainLosses, testAcc, testAvgLoss, testLosses,auroc, lrs = duqTrain(trainloader,testloader,falseloader,num_classes,
                                                                                                model, optimizer,scheduler,l,epochs)

    auroc1 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader.dataset, model=model, device=device, model_type='duq')
    auroc2 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader2.dataset, model=model, device=device, model_type='duq')
    auroc3 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader3.dataset, model=model, device=device, model_type='duq')

    return trainAcc,trainLosses[-1],testAcc,testLosses[-1],auroc1,auroc2,auroc3

def MLPcreateAndTrain(trainloader,testloader,falseloader,falseloader2,falseloader3,arch,lr,epochs):
    model = customLinearNetwork(len(arch),arch).to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr = lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min',patience=5)
    trainAcc, trainAvgLoss, testAcc, testAvgLoss, aurocLog, lrLog = softmaxTrain(trainloader,testloader,falseloader, model,optimizer,scheduler,epochs)

    auroc1 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader.dataset, model=model, device=device, model_type='mlp')
    auroc2 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader2.dataset, model=model, device=device, model_type='mlp')
    auroc3 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader3.dataset, model=model, device=device, model_type='mlp')

    return trainAcc,trainAvgLoss,testAcc,testAvgLoss,auroc1,auroc2,auroc3

def DEcreateAndTrain(trainloader,testloader,falseloader,falseloader2,falseloader3,modelsize,arch,lr,epochs):
    models = [customLinearNetwork(len(arch),arch).to(device) for _ in range(modelsize)]
    optimizers = [torch.optim.SGD(model.parameters(),lr = lr) for model in models]
    schedulers = [ReduceLROnPlateau(optimizer, 'min',patience=5) for optimizer in optimizers]
    aurocs1 = []
    aurocs2 = []
    aurocs3 = []
    trainAccs = []
    trainLosses = []
    testAccs = []
    testLosses = []
#
    for i in range(len(models)):
        trainAcc, trainAvgLoss, testAcc, testAvgLoss, aurocLog, lrLog = softmaxTrain(trainloader,testloader,falseloader, models[i],
                                                                                     optimizers[i],schedulers[i], epochs)
#
        auroc1 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader.dataset, model=models[i], device=device, model_type='mlp')
        auroc2 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader2.dataset, model=models[i], device=device, model_type='mlp')
        auroc3 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader3.dataset, model=models[i], device=device, model_type='mlp')
        aurocs1.append(auroc1)
        aurocs2.append(auroc2)
        aurocs3.append(auroc3)
        trainAccs.append(trainAcc)
        trainLosses.append(trainAvgLoss)
        testAccs.append(testAcc)
        testLosses.append(testAvgLoss)
    print()
    
    return np.mean(trainAccs),np.mean(trainLosses),np.mean(testAccs),np.mean(testLosses),np.mean(aurocs1),np.mean(aurocs2),np.mean(aurocs3)

def main():

    print("1. Train DUQ")
    print("2. Train MLP")
    print("3. Train n-ensemble")
    print("0. Exit")
    ans = str(input())
    save = str(input("Save?(y/n):"))
    if save.lower() == 'y':
        save = True
        runName = str(input('Run name: '))
    else: save = False

    # for answers like 123
    for a in ans:
        match a:
            case '1':
                epochs = 1
                lrs = [1e-1]
                #lrs = [1e-4,1e-3,1e-2]
                lrsSigma = [1e-1,1e-2,1e-3,1e-4,1e-5]
                lrsSigma = [1e-3]
                initSigmas = [0.1,0.3,0.5,1,2]
                #initSigmas = [0.5]
                architectures = [[3,32,16,2]]
                #architectures = [[3,32,16,2], [3,16,32,2]]
                lam = [0, 0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, .25, .3, .5, 1]
                stds = [1e-2]
                #stds = [1e-4, 1e-3, 1e-2, 1e-1]
                modelsPerTest = 5 # no of models per set of hyperparams, for stat signif.

                trainloader,testloader,falseloader,falseloader2,falseloader3 = loadAllDataloaders(binary=True if architectures[0][-1]==2 else False)

                totalResults = []
                for arch in architectures:
                    print(f"Arch: {arch}")
                    for lr in lrs:
                        print(f"Lr: {lr}")
                        for lrSigma in lrsSigma:
                            print(f"Denom lr: {lrSigma}")
                            for initSigma in initSigmas:
                                print(f"Init sigma: {initSigma}") 
                                for lamda in lam:
                                    print(f"Lamda: {lamda}")
                                    for std in stds:
                                        print(f"std: {std}")
                                        results = [DUQcreateAndTrain(trainloader,testloader,falseloader,falseloader2,falseloader3,
                                                                    arch,std,initSigma,lr,lrSigma,lamda,epochs) for _ in range(modelsPerTest)]
                                        trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2,aurocs3 = zip(*results) # creates lists for each return variable
                                        finalResult = [arch,lr,lamda, std,
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
                                        totalResults.append(finalResult)
                if save:
                    colNames = ['arch','lr','lamda','std',
                                'trainAccs','train acc std', 'trainLosses','train loss std',
                                'testAccs', 'test acc std','testLosses','test loss std',
                                'auroc wine','auroc wine std','auroc iris','auroc iris std', 'auroc cancer','auroc cancer std']
                    Path(f"results/DUQ/{runName}").mkdir(parents=True, exist_ok=True)
                    saveToEXCEL(totalResults,colNames,f"results/DUQ/{runName}/results")

            case '2':
                epochs = 1
                lrs = [1e-1,1e-2]
                architectures = [[3,32,16,2]]
                #architectures = [[3,32,16,2], [3, 16,32,2]]
                modelsPerTest = 1 # no of models per set of hyperparams, for stat signif.

                trainloader,testloader,falseloader,falseloader2,falseloader3 = loadAllDataloaders(binary=True if architectures[0][-1]==2 else False)
                totalResults = []
                for arch in architectures:
                    print(f"Architecture: {arch}")
                    for lr in lrs:
                        print(f"Lr: {lr}")
                        results = [MLPcreateAndTrain(trainloader,testloader,falseloader,falseloader2,falseloader3,
                                                    arch,lr,epochs) for _ in range(modelsPerTest)]
                        trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2,aurocs3 = zip(*results) # creates lists for each return variable
                        finalResult = [arch,lr,
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
                        totalResults.append(finalResult)

                if save:
                    colNames = ['arch','lr',
                                'trainAccs','train acc std', 'trainLosses','train loss std',
                                'testAccs', 'test acc std','testLosses','test loss std',
                                'auroc wine','auroc wine std','auroc iris','auroc iris std', 'auroc cancer','auroc cancer std']
                    Path(f"results/MLP/{runName}").mkdir(parents=True, exist_ok=True)
                    saveToEXCEL(totalResults,colNames,f"results/MLP/{runName}/results")

            case '3':
                epochs = 1
                lrs = [1e-1,1e-2]
                #architectures = [[3,32,16,2],[3,16,32,2]]
                architectures = [[3,32,16,2]]
                modelsNumEnsemble = 5
                modelsPerTest = 5 # no of models per set of hyperparams, for stat signif.

                trainloader,testloader,falseloader,falseloader2,falseloader3 = loadAllDataloaders(binary=True if architectures[0][-1]==2 else False)
                totalResults = []
                for arch in architectures:
                    print(f"Architecture: {arch}")
                    for lr in lrs:
                        print(f"Lr: {lr}")
                        results = [DEcreateAndTrain(trainloader,testloader,falseloader,falseloader2,falseloader3,
                                                    modelsNumEnsemble,arch,lr,epochs) for _ in range(modelsPerTest)]
                        trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2,aurocs3 = zip(*results) # creates lists for each return variable
                        finalResult = [arch,lr,
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
                        totalResults.append(finalResult)

                if save:
                    colNames = ['arch','lr',
                                'trainAccs','train acc std', 'trainLosses','train loss std',
                                'testAccs', 'test acc std','testLosses','test loss std',
                                'auroc wine','auroc wine std','auroc iris','auroc iris std', 'auroc cancer','auroc cancer std']
                    Path(f"results/DE/{runName}").mkdir(parents=True, exist_ok=True)
                    saveToEXCEL(totalResults,colNames,f"results/DE/{runName}/results")

            case '0':
                exit()
    
if __name__ == "__main__":
    main()

