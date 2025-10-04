from datasets import *
from models import *
from functions import *
from BSRBF_KAN import *
from tools import *
from oodEvaluation import *
from cnn_duq import *
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
        y = y.squeeze()
        pred = model.forward(X)
        loss = lossfn(pred, y)
        trloss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        correct += (torch.argmax(pred,dim=1) == y).sum().item()
    accuracy = correct / len(dataloader.dataset)
    avgLoss = trloss/len(dataloader)
    #print(f"Accuracy:{accuracy}, avgloss: {avgLoss}")
    return accuracy, avgLoss

def softmaxTest(dataloader, model):
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            #y = y.squeeze() ###3 comment out in ambros
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

        #auroc.append(get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader.dataset, model=model, device=device, model_type='mlp'))
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

def deTest(models,valloader,oodset1,oodset2,oodset3,oodset4,criterion):
    for model in models:
        model.eval()
    preds = []
    totallabels = []
    aurocs1 = []
    aurocs2 = []
    aurocs3 = []
    aurocs4 = []
    with torch.no_grad():
        for x, y in valloader:
            batchprobs = []
            for model in models:
                output = model(x.to(device))
                batchprobs.append(output)

        
            avg_probs = torch.stack(batchprobs).mean(dim=0)  
            preds.append(torch.argmax(avg_probs, dim=1))
            totallabels.append(y)

    
    for model in models:
        #aurocs1.append(get_auroc_ood(true_dataset=oodset1, ood_dataset=valloader.dataset, model=model, device=device, model_type='mlp'))
        aurocs1.append(get_auroc_ood(ood_dataset=oodset1, true_dataset=valloader.dataset, model=model, device=device, model_type='mlp'))
        if oodset2:
            #aurocs2.append(get_auroc_ood(true_dataset=oodset2, ood_dataset=valloader.dataset, model=model, device=device, model_type='mlp'))
            aurocs2.append(get_auroc_ood(ood_dataset=oodset2, true_dataset=valloader.dataset, model=model, device=device, model_type='mlp'))
        if oodset3:
            #aurocs3.append(get_auroc_ood(true_dataset=oodset3, ood_dataset=valloader.dataset, model=model, device=device, model_type='mlp'))
            aurocs3.append(get_auroc_ood(ood_dataset=oodset3, true_dataset=valloader.dataset, model=model, device=device, model_type='mlp'))
        if oodset4:
            #aurocs4.append(get_auroc_ood(true_dataset=oodset4, ood_dataset=valloader.dataset, model=model, device=device, model_type='mlp'))
            aurocs4.append(get_auroc_ood(ood_dataset=oodset4, true_dataset=valloader.dataset, model=model, device=device, model_type='mlp'))
    preds = torch.cat(preds)
    totallabels= torch.cat(totallabels)
    val_loss = criterion(avg_probs, y.to(device)).item()
    val_accuracy = ((preds == totallabels.to(device)).float().mean().item())
    auroc1 = np.mean(aurocs1)
    auroc2 = np.mean(aurocs2)
    auroc3 = np.mean(aurocs3)
    auroc4 = np.mean(aurocs4)

    return val_accuracy,val_loss,auroc1,auroc2,auroc3,auroc4


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
        #print(f"TrainACC:{trainAcc:>.3f}")
        #print(f"TestACC:{testAcc:>.3f}")
        #print(f"AUROC:{auroc[-1]:>.3f}")

        scheduler.step(testAvgLoss)
        lrs.append(scheduler.get_last_lr())
    #print(scheduler.get_last_lr())

    #plot([i for i in range(epochs)],'epochs',ylosses,'loss','r')
        #print(f"Train Acc:{trainAcc}")
        #print(f"Test Acc:{testAcc}")
        #print(f"Auroc:{auroc[-1]}")
    #plt.plot(trainLosses)
    #plt.plot(testLosses)
    #plt.show()
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

def DUQcreateAndTrain(trainloader,testloader,falseloader,falseloader2,falseloader3,falseloader4,arch,std,initSigma,lr,lrSigma,l,epochs):
    model = SimpleLinearNetwork_DUQ(arch[0],arch[1],arch[2],arch[3],std,initSigma=initSigma).to(device)
    #model = CNN_DUQ(10,256,True,0.5,0.999).to(device)
    num_classes = arch[-1]
    sigmaParam = [p for name, p in model.named_parameters() if 'sigma' in name]
    othersParam = [p for name, p in model.named_parameters() if 'sigma' not in name]
    optimizer = torch.optim.SGD([{"params":othersParam},{"params":sigmaParam,"lr":lrSigma}],lr = lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min',patience=5)
    trainAcc, trainAvgLoss, trainLosses, testAcc, testAvgLoss, testLosses,auroc, lrs = duqTrain(trainloader,testloader,falseloader,num_classes,
                                                                                                model, optimizer,scheduler,l,epochs)

    auroc1 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader.dataset, model=model, device=device, model_type='duq')
    auroc2 = auroc3 = '-'
    if falseloader2:
        auroc2 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader2.dataset, model=model, device=device, model_type='duq')
    if falseloader3:
        auroc3 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader3.dataset, model=model, device=device, model_type='duq')
    if falseloader4:
        auroc4 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader4.dataset, model=model, device=device, model_type='duq')

    return trainAcc,trainLosses[-1],testAcc,testLosses[-1],auroc1,auroc2,auroc3,auroc4

def MLPcreateAndTrain(trainloader,testloader,falseloader,falseloader2,falseloader3,falseloader4,arch,lr,epochs):
    model = customLinearNetwork(len(arch),arch).to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr = lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min',patience=5)
    trainAcc, trainAvgLoss, testAcc, testAvgLoss, aurocLog, lrLog = softmaxTrain(trainloader,testloader,falseloader, model,optimizer,scheduler,epochs)

    auroc1 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader.dataset, model=model, device=device, model_type='mlp')
    auroc2 = auroc3 = '-'
    if falseloader2:
        auroc2 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader2.dataset, model=model, device=device, model_type='mlp')
    if falseloader3:
        auroc3 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader3.dataset, model=model, device=device, model_type='mlp')
    if falseloader4:
        auroc4 = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader4.dataset, model=model, device=device, model_type='mlp')

    return trainAcc,trainAvgLoss,testAcc,testAvgLoss,auroc1,auroc2,auroc3,auroc4

def DEcreateAndTrain(trainloader,testloader,falseloader,falseloader2,falseloader3,falseloader4,modelsize,arch,lr,epochs):
    models = [customLinearNetwork(len(arch),arch).to(device) for _ in range(modelsize)]
    optimizers = [torch.optim.SGD(model.parameters(),lr = lr) for model in models]
    schedulers = [ReduceLROnPlateau(optimizer, 'min',patience=5) for optimizer in optimizers]
    trainAccs = []
    trainLosses = []
#
    for i in range(len(models)):
        trainAcc, trainAvgLoss, testAcc, testAvgLoss, aurocLog, lrLog = softmaxTrain(trainloader,testloader,falseloader, models[i],
                                                                                     optimizers[i],schedulers[i], epochs)
        trainAccs.append(trainAcc)
        trainLosses.append(trainAvgLoss)
        print()
    testAcc,testLoss,auroc1,auroc2,auroc3,auroc4 = deTest(models,testloader,falseloader.dataset,
                                                          falseloader2.dataset,falseloader3.dataset,falseloader4.dataset,nn.CrossEntropyLoss())
    
    if falseloader2 and falseloader3 and falseloader4:
        return np.mean(trainAccs),np.mean(trainLosses),testAcc,testLoss,auroc1,auroc2,auroc3,auroc4
    else:
        return np.mean(trainAccs),np.mean(trainLosses),testAcc,testLoss,auroc1,auroc2,None,None

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
                epochs = 20
                lrs = [1e-1]
                lrs = [1e-1,1e-2,1e-3]
                lrsSigma = [1e-3]
                #lrsSigma = [1e-1,1e-2,1e-3]
                initSigmas = [2,5,8,10,15]
                initSigmas = [2]
                #architectures = [[3,32,16,2]]
                #architectures = [[3,32,16,2], [3,16,32,2]]
                architectures = [[353,64,32,9]]
                #architectures = [[353,32,16,9],[353,64,32,9],[353,128,64,9]]
                lam = [0, 0.01, 0.1, .25, .5, 1]
                lam = [0]
                stds = [1e-2]
                #stds = [1e-3]
                #stds = [1e-4, 1e-3, 1e-2, 1e-1]
                modelsPerTest = 3 # no of models per set of hyperparams, for stat signif.

                #trainloader,testloader,falseloader,falseloader2,falseloader3 = loadAllDataloaders(binary=True if architectures[0][-1]==2 else False)
                trainloader,testloader,falseloader,falseloader2,falseloader3,falseloader4 = loadImageDataloaders()

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
                                        trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2,aurocs3,aurocs4 = zip(*results) # creates lists for each return variable
                                        finalResult = [arch,lr,lamda, std,lrSigma, initSigma,
                                                          np.mean(trainAccs),    np.std(trainAccs),
                                                          np.mean(trainLosses),  np.std(trainLosses),
                                                          np.mean(testAccs),     np.std(testAccs),
                                                          np.mean(testLosses),   np.std(testLosses),
                                                          np.mean(aurocs1),      np.std(aurocs1),
                                                        ]# double list for save to excel implement
                                        if falseloader2: finalResult.extend([np.mean(aurocs2),np.std(aurocs2)])
                                        if falseloader3: finalResult.extend([np.mean(aurocs3),np.std(aurocs3)])
                                        if falseloader4: finalResult.extend([np.mean(aurocs4),np.std(aurocs4)])
                                        print(f"Train Acc {100*finalResult[6]:>.1f}% std {100*finalResult[7]:>.1f}, AvgLoss {finalResult[8]:>.3f} std {finalResult[9]:>.3f}")
                                        print(f"Test  Acc {100*finalResult[10]:>.1f}% std {100*finalResult[11]:>.1f}, AvgLoss {finalResult[12]:>.3f} std {finalResult[13]:>.3f}")
                                        print(f"AUROC 1  {finalResult[14]:>.3f} std {finalResult[15]:>.3f}")
                                        if falseloader2:
                                            print(f"AUROC 2  {finalResult[16]:>.3f} std {finalResult[17]:>.3f}")
                                        if falseloader3:
                                            print(f"AUROC 3 {finalResult[18]:>.3f} std {finalResult[19]:>.3f}")
                                        if falseloader4:
                                            print(f"AUROC 4 {finalResult[18]:>.3f} std {finalResult[19]:>.3f}")
                                        totalResults.append(finalResult)
                if save:
                    colNames = ['arch','lr','lamda','std','lrsigma','initSigma',
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
                    saveToEXCEL(totalResults,colNames,f"results/DUQ/{runName}/results")

            case '2':
                epochs = 20
                lrs = [1e-1,1e-2]
                lrs = [1e-1]
                #architectures = [[3,32,16,2]]
                architectures = [[353,64,32,9]]
                
                #architectures = [[353,32,16,9],[353,64,32,9],[353,128,64,9]]
                #architectures = [[3,32,16,2], [3, 16,32,2]]
                modelsPerTest = 3 # no of models per set of hyperparams, for stat signif.

                #trainloader,testloader,falseloader,falseloader2,falseloader3 = loadAllDataloaders(binary=True if architectures[0][-1]==2 else False)
                trainloader,testloader,falseloader,falseloader2,falseloader3,falseloader4 = loadImageDataloaders()
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
                                                        ]# double list for save to excel implement
                        if falseloader2: finalResult.extend([np.mean(aurocs2),np.std(aurocs2)])
                        if falseloader3: finalResult.extend([np.mean(aurocs3),np.std(aurocs3)])
                        print(f"Train Acc {100*finalResult[2]:>.1f}% std {100*finalResult[3]:>.1f}, AvgLoss {finalResult[4]:>.3f} std {finalResult[5]:>.3f}")
                        print(f"Test  Acc {100*finalResult[6]:>.1f}% std {100*finalResult[7]:>.1f}, AvgLoss {finalResult[8]:>.3f} std {finalResult[9]:>.3f}")
                        print(f"AUROC 1  {finalResult[10]:>.3f} std {finalResult[11]:>.3f}")
                        if falseloader2:
                            print(f"AUROC 2  {finalResult[14]:>.3f} std {finalResult[15]:>.3f}")
                        if falseloader3:
                            print(f"AUROC 3 {finalResult[16]:>.3f} std {finalResult[17]:>.3f}")
                        totalResults.append(finalResult)
                if save:
                    colNames = ['arch','lr',
                                'trainAccs','train acc std', 'trainLosses','train loss std',
                                'testAccs', 'test acc std','testLosses','test loss std',
                                'auroc 1','auroc 1 std']
                    if falseloader2:
                        colNames.extend(['auroc 2','auroc 2 std'])
                    if falseloader3:
                        colNames.extend(['auroc 3','auroc 3 std'])
                    Path(f"results/MLP/{runName}").mkdir(parents=True, exist_ok=True)
                    saveToEXCEL(totalResults,colNames,f"results/MLP/{runName}/results")

            case '3':
                epochs = 120
                lrs = [1e-1]
                #architectures = [[3,32,16,2],[3,16,32,2]]
                architectures = [[3,32,16,2]]
                #architectures = [[62,64,32,4]]
                #architectures = [[353,64,32,9]]
                modelsNumEnsemble = 5
                modelsPerTest = 5 # no of models per set of hyperparams, for stat signif.

                trainloader,testloader,falseloader,falseloader2,falseloader3,falseloader4= loadAllDataloaders(binary=True if architectures[0][-1]==2 else False)
                #trainloader,testloader,falseloader,falseloader2,falseloader3 = loadImageDataloaders()
                totalResults = []
                for arch in architectures:
                    print(f"Architecture: {arch}")
                    for lr in lrs:
                        print(f"Lr: {lr}")
                        results = [DEcreateAndTrain(trainloader,testloader,falseloader,falseloader2,falseloader3,falseloader4,
                                                    modelsNumEnsemble,arch,lr,epochs) for _ in range(modelsPerTest)]
                        trainAccs, trainLosses, testAccs, testLosses, aurocs1, aurocs2,aurocs3,aurocs4 = zip(*results) # creates lists for each return variable
                        finalResult = [arch,lr,
                                          np.mean(trainAccs),    np.std(trainAccs),
                                          np.mean(trainLosses),  np.std(trainLosses),
                                          np.mean(testAccs),     np.std(testAccs),
                                          np.mean(testLosses),   np.std(testLosses),
                                          np.mean(aurocs1),      np.std(aurocs1),
                                                        ]# double list for save to excel implement
                        if falseloader2: finalResult.extend([np.mean(aurocs2),np.std(aurocs2)])
                        if falseloader3: finalResult.extend([np.mean(aurocs3),np.std(aurocs3)])
                        if falseloader4: finalResult.extend([np.mean(aurocs4),np.std(aurocs4)])
                        print(f"Train Acc {100*finalResult[2]:>.1f}% std {100*finalResult[3]:>.1f}, AvgLoss {finalResult[4]:>.3f} std {finalResult[5]:>.3f}")
                        print(f"Test  Acc {100*finalResult[6]:>.1f}% std {100*finalResult[7]:>.1f}, AvgLoss {finalResult[8]:>.3f} std {finalResult[9]:>.3f}")
                        print(f"AUROC 1 {finalResult[10]:>.3f} std {finalResult[11]:>.3f}")
                        if falseloader2:
                            print(f"AUROC 2 {finalResult[12]:>.3f} std {finalResult[13]:>.3f}")
                        if falseloader3:
                            print(f"AUROC 3 {finalResult[14]:>.3f} std {finalResult[15]:>.3f}")
                        if falseloader4:
                            print(f"AUROC 4 {finalResult[16]:>.3f} std {finalResult[17]:>.3f}")
                        totalResults.append(finalResult)
                if save:
                    colNames = ['arch','lr',
                                'trainAccs','train acc std', 'trainLosses','train loss std',
                                'testAccs', 'test acc std','testLosses','test loss std',
                                'auroc 1','auroc 1 std']
                    if falseloader2:
                        colNames.extend(['auroc 2','auroc 2 std'])
                    if falseloader3:
                        colNames.extend(['auroc 3','auroc 3 std'])
                    if falseloader4:
                        colNames.extend(['auroc 4','auroc 4 std'])
                    Path(f"results/DE/{runName}").mkdir(parents=True, exist_ok=True)
                    saveToEXCEL(totalResults,colNames,f"results/DE/{runName}/results")

            case '0':
                exit()
    
if __name__ == "__main__":
    main()

