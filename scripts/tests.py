from datasets import *
from models import *
from tools import *
from oodEvaluation import *
from sklearn.datasets import load_iris,load_diabetes


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def softmaxTrain(dataloader, model, lossfn, optimizer):
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
    #print(f"Train Error = Accuracy: {100*correct/len(dataloader.dataset):>.1f}, Avg Loss: {trloss/len(dataloader):>.4f}")
    accuracy = correct / len(dataloader.dataset)
    avgLoss = trloss/len(dataloader)
    return accuracy, avgLoss

def softmaxTest(dataloader, model, lossfn):
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model.forward(X)
            test_loss += lossfn(pred, y).item()
            correct += (torch.argmax(pred,dim=1) == y).type(torch.float).sum().item()
            #print(torch.argmax(pred,dim=1).tolist())
    #print(f"Test Error = Accuracy: {(100*correct):>.1f}%, Avg loss: {test_loss:>.4f} \n")
    accuracy = correct / len(dataloader.dataset)
    avgLoss = test_loss/len(dataloader)
    return accuracy, avgLoss

def softmaxTrainTest(model, optimizer, trainloader, testloader, epochs):
    for e in range(epochs):
        softmaxTrain(trainloader,model,nn.CrossEntropyLoss(),optimizer)
        acc, avgl = softmaxTest(testloader,model,nn.CrossEntropyLoss())
    return acc, avgl

def duqModelOptimizerInit():
    duqmodel = SimpleLinearNetwork_DUQ(3,32,16,3).to(device)
    optimizer = torch.optim.SGD(duqmodel.parameters(),lr = 1e-3)
    return duqmodel, optimizer

def duqTrain(dataloader, model, optimizer, l):
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
        #model.printSigma()
        totalloss += loss.item()
        optimizer.step()
        with torch.no_grad():
            model.updateCentroids(x,y)
    accuracy = correct / len(dataloader.dataset)
    avgLoss = totalloss/len(dataloader)
    #print(f"Train: Accuracy: {100*correct/len(dataloader.dataset):>.1f}%, Avg Loss: {totalloss/len(dataloader):>.4f}")
    return accuracy, avgLoss

def duqTest(dataloader, model):
    model.eval()
    testLoss, correct = 0, 0
    totalCert = 0
    with torch.no_grad():
        for x, y in dataloader:
            y = F.one_hot(y,num_classes=3).type(torch.float)
            x = x.to(device)
            y = y.to(device)
            ypred,z = model.forward(x)
            totalCert += torch.max(ypred,dim=1)[0].mean()
            testLoss += F.cross_entropy(ypred, y).item()
            correct += (torch.argmax(ypred,dim=1) == torch.argmax(y,dim=1)).sum().item()
    accuracy = correct / len(dataloader.dataset)
    avgLoss = testLoss/len(dataloader)
    #print(f"Test: Accuracy: {(100*correct):>.1f}%, Avg loss: {testLoss:>.4f} Avg Cert: {totalCert/len(dataloader.dataset):>.5f}")
    return accuracy, avgLoss 

def duqTrainTest(model, optimizer,trainloader,testloader, epochs):
    # grad penalty var
    l = 1
    for e in range(epochs):
        acc, avgl = duqTrain(trainloader,model,optimizer,l)
        #print(f"Epoch {e+1}, Acc {100*acc:>.1f}, AvgLoss {avgl:>.3f}")
    #print(f"\nMain test")
    testAcc, testAvgLoss = duqTest(testloader,model)
    return testAcc, testAvgLoss

def duqTrainTestAUROC(trainloader, testloader, falseloader):
    epochs = 50
    model, optimizer = duqModelOptimizerInit()
    testAcc, testAvgLoss = duqTrainTest(model, optimizer, trainloader, testloader, epochs)
    aurocDUQ = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader.dataset, model=model, device=device, standard_model=False, isSoftmax = False)
    print(f"DUQ:     Test Acc {100*(testAcc):>.1f}%, AvgLoss {testAvgLoss:>.3f}, AUROC {aurocDUQ:>.3f}")

def softmaxVSduq(trainloader,testloader, falseLoader):
    epochs = 30
    lrs = [1e-2,1e-3,1e-4,1e-5]
    architectures = [[3,2**i,2**j,3] for i in range(2,6) for j in range(2,6)] # 
    modelsDUQ = [SimpleLinearNetwork_DUQ(ar[0],ar[1],ar[2],ar[3]).to(device) for ar in architectures]
    modelsSoftmax = [SimpleLinearNetwork_Softmax(ar[0], ar[1], ar[2],ar[3]).to(device) for ar in architectures]


    for i in range(len(modelsDUQ)):
        softmaxAccuracies = []
        softmaxAvgLosses = []
        smAUROCs = []

        duqAccuracies = []
        duqAvgLosses = []
        duqAUROCs = []
        print("==========================")
        print(f"Architectures:{modelsDUQ[i].getArchitecture()}")
        for lr in lrs:
            print(f"Learning rate:{lr}")
            optimizerDUQ = torch.optim.SGD(modelsDUQ[i].parameters(),lr = lr)
            optimizerSM = torch.optim.SGD(modelsSoftmax[i].parameters(),lr = lr)
            accDUQ, avglDUQ = duqTrainTest(modelsDUQ[i], optimizerDUQ,trainloader,testloader,epochs)
            accSM, avglSM = softmaxTrainTest(modelsSoftmax[i], optimizerSM,trainloader,testloader,epochs)

            softmaxAccuracies.append(accSM)
            softmaxAvgLosses.append(avglSM)

            duqAccuracies.append(accDUQ)
            duqAvgLosses.append(avglDUQ)

            print(f"DUQ:     Test Acc {100*accDUQ:>.1f}%, AvgLoss {avglDUQ:>.3f}")
            print(f"Softmax: Test Acc {100*accSM:>.1f}%, AvgLoss {avglSM:>.3f}")
            aurocDUQ = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseLoader.dataset, model=modelsDUQ[i], device=device, standard_model=False, isSoftmax = False)
            aurocSM = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseLoader.dataset, model=modelsSoftmax[i], device=device, standard_model=False, isSoftmax = True)
            duqAUROCs.append(aurocDUQ)
            smAUROCs.append(aurocSM)
            print(f"duq AUROC: {aurocDUQ:>.4f}")
            print(f"softmax AUROC: {aurocSM:>.4f}")
            print("-")
        print(f"Architecture Averages:")
        print(f"DUQ:     Test Acc {100*(np.mean(duqAccuracies)):>.1f}%, AvgLoss {(np.mean(duqAvgLosses)):>.3f}")
        print(f"Softmax: Test Acc {100*(np.mean(softmaxAccuracies)):>.1f}%, AvgLoss {np.mean(softmaxAvgLosses):>.3f}")
        print(f"duq AUROC:     {np.mean(duqAUROCs):>.4f}")
        print(f"softmax AUROC: {np.mean(smAUROCs):>.4f}")
    
def deepEnsemblesTrain(dataloader, models, optimizers):
    results = []
    for i, model in enumerate(models):
        model.train()
        correct, trloss = 0, 0
        for x,y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model.forward(x)
            loss = F.cross_entropy(pred,y)
            trloss += loss.item()
            loss.backward()
            optimizers[i].step()
            optimizers[i].zero_grad()
            correct += (torch.argmax(pred,dim=1) == y).sum().item()
        accuracy = correct / len(dataloader.dataset)
        avgLoss = trloss/len(dataloader)
        results.append((accuracy,avgLoss))
    # results = list of tuples (acc, avgLoss)
    return results

def deepEnsemblesTest(dataloader, models):
    modelsPredictions = []
    targets = [] # gather all targets, from all batches
    for model in models:
        model.eval()
        pred = []
        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)
                batchpred = model.forward(X)
                pred += batchpred.cpu()
        modelsPredictions.append(pred)
    modelsPredictions = np.array(modelsPredictions)
    modelsPredictions = np.sum(modelsPredictions,axis=0)/len(models) # get average of logits
    modelsPredictions = np.argmax(modelsPredictions, axis=1)
    targets = getDataloaderTargets(dataloader)
    acc = np.sum(targets == modelsPredictions)/len(targets)
    return acc

def deepEnsemblesTrainTest(ensemblesNum, trainloader,testloader):
    epochs = 30
    # converges to 42.2 acc, with w/e architecture 
    ensembleArch = [[3,32,16,3],[3,16,16,3],[3,16,32,3]]
    ensempleModels1 = [customLinearNetwork(4,ensembleArch[0]).to(device) for _ in range(ensemblesNum)]
    ensempleModels2 = [customLinearNetwork(4,ensembleArch[1]).to(device) for _ in range(ensemblesNum)]
    ensempleModels3 = [customLinearNetwork(4,ensembleArch[2]).to(device) for _ in range(ensemblesNum)]
    ensempleOptim1 = [torch.optim.SGD(model.parameters(), lr=0.01) for model in ensempleModels1]
    ensempleOptim2 = [torch.optim.SGD(model.parameters(), lr=0.01) for model in ensempleModels2]
    ensempleOptim3 = [torch.optim.SGD(model.parameters(), lr=0.01) for model in ensempleModels3]
    for e in range(epochs):
        resEnsemple1 = deepEnsemblesTrain(trainloader, ensempleModels1, ensempleOptim1)
        resEnsemple2 = deepEnsemblesTrain(trainloader, ensempleModels2, ensempleOptim2)
        resEnsemple3 = deepEnsemblesTrain(trainloader, ensempleModels3, ensempleOptim3)
        print(f"Epoch:{e+1}")
        #for i in range(len(resEnsemple1)):
        #    print(f"Model {i+1} Acc:{resEnsemple1[i][0]:>.3f} AvgL:{resEnsemple1[i][1]:>.3f}")
        accEnsemple1 = deepEnsemblesTest(testloader, ensempleModels1)
        accEnsemple2 = deepEnsemblesTest(testloader, ensempleModels2)
        accEnsemple3 = deepEnsemblesTest(testloader, ensempleModels3)
        print(f"{ensemblesNum}-Ensemble Architecture{ensembleArch[0]} Acc: {100*accEnsemple1:>.1f}")
        print(f"{ensemblesNum}-Ensemble Architecture{ensembleArch[1]} Acc: {100*accEnsemple2:>.1f}")
        print(f"{ensemblesNum}-Ensemble Architecture{ensembleArch[2]} Acc: {100*accEnsemple3:>.1f}")

def lambdaTest(trainloader, testloader, falseLoader):
    lam = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1., 2., 3., 5. , 10.]
    model,optimizer = duqModelOptimizerInit()
    for l in lam:
        print(f"Lambda = {l:>.2f}")
        accDUQ, avglDUQ = duqTrainTest(model, optimizer,trainloader,testloader,epochs=50)
        print(f"DUQ: Test Acc {100*accDUQ:>.1f}%, AvgLoss {avglDUQ:>.3f}")
        auroc = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseLoader.dataset, model=model, device=device, standard_model=False)
        print(f"AUROC: {auroc:>.4f}")
        print("---")

    
def main():
    # set dev to gpu

    ambTestSet, ambTrainLoader, ambTestLoader, ambFeaturesDim = load_D1(1)
    print(f"Train size: {len(ambTrainLoader.dataset)}")
    print(f"Test size: {len(ambTestLoader.dataset)}")

    falseloader = createSklearnDataloader(load_diabetes(),[1,3,4])
    #duqTrainTestAUROC(ambTrainLoader,ambTestLoader,falseloader)
    #softmaxVSduq(ambTrainLoader,ambTestLoader,falseloader)
    #lambdaTest(ambTrainLoader,ambTestLoader,falseloader)
    #deepEnsemblesTrainTest(5,ambTrainLoader,ambTestLoader)


    #lambdaTest(ambTrainLoader,ambTestLoader, diabDataloader)

main()