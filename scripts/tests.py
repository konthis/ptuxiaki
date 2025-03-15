from datasets import *
from DUQ_ambrosia1 import *
from oodEvaluation import *
from sklearn.datasets import load_iris,load_diabetes


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def trainSoftmax( dataloader, model, lossfn, optimizer):
    model.train()
    trloss,correct = 0, 0 
    for batch, (X,y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model.forwardSoftmax(X)
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

def testSoftmax(dataloader, model, lossfn):
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model.forwardSoftmax(X)
            test_loss += lossfn(pred, y).item()
            correct += (torch.argmax(pred,dim=1) == y).type(torch.float).sum().item()
            #print(torch.argmax(pred,dim=1).tolist())
    #print(f"Test Error = Accuracy: {(100*correct):>.1f}%, Avg loss: {test_loss:>.4f} \n")
    accuracy = correct / len(dataloader.dataset)
    avgLoss = test_loss/len(dataloader)
    return accuracy, avgLoss

def softmaxTrainTest(model, optimizer, trainloader, testloader, epochs):
    for e in range(epochs):
        trainSoftmax(trainloader,model,nn.CrossEntropyLoss(),optimizer)
        acc, avgl = testSoftmax(testloader,model,nn.CrossEntropyLoss())
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
    l = 0.1
    for e in range(epochs):
        acc, avgl = duqTrain(trainloader,model,optimizer,l)
        #print(f"Epoch {e+1}, Acc {100*acc:>.1f}, AvgLoss {avgl:>.3f}")
    #print(f"\nMain test")
    testAcc, testAvgLoss = duqTest(testloader,model)
    return testAcc, testAvgLoss

def modelsTest(trainloader,testloader, falseLoader):
    epochs = 30
    lrs = [1e-2,1e-3,1e-4,1e-5]
    architectures = [[3,2**i,2**j,3] for i in range(2,6) for j in range(2,6)] # 
    modelsDUQ = [SimpleLinearNetwork_DUQ(ar[0],ar[1],ar[2],ar[3]).to(device) for ar in architectures]
    modelsSoftmax = [SimpleLinearNetwork(ar[0], ar[1]).to(device) for ar in architectures]

    for i in range(len(modelsDUQ)):
        print("==========================")
        print(f"Architectures:{modelsDUQ[i].getArchitecture()}")
        for lr in lrs:
            print(f"Learning rate:{lr}")
            optimizerDUQ = torch.optim.SGD(modelsDUQ[i].parameters(),lr = lr)
            optimizerSM = torch.optim.SGD(modelsSoftmax[i].parameters(),lr = lr)
            accDUQ, avglDUQ = duqTrainTest(modelsDUQ[i], optimizerDUQ,trainloader,testloader,epochs)
            accSM, avglSM = softmaxTrainTest(modelsSoftmax[i], optimizerSM,trainloader,testloader,epochs)
            print(f"DUQ:     Test Acc {100*accDUQ:>.1f}%, AvgLoss {avglDUQ:>.3f}")
            print(f"Softmax: Test Acc {100*accSM:>.1f}%, AvgLoss {avglSM:>.3f}")
            auroc = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseLoader.dataset, model=modelsDUQ[i], device=device, standard_model=False)
            print(f"AUROC: {auroc:>.4f}")
    
def lambdaTest(trainloader, testloader, falseLoader):
    lam = [.1*i for i in range(1,11)]
    model,optimizer = duqModelOptimizerInit()
    for l in lam:
        print("=====================")
        print(f"Lambda = {l:>.2f}")
        accDUQ, avglDUQ = duqTrainTest(model, optimizer,trainloader,testloader,30)
        print(f"DUQ: Test Acc {100*accDUQ:>.1f}%, AvgLoss {avglDUQ:>.3f}")
        auroc = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseLoader.dataset, model=model, device=device, standard_model=False)
        print(f"AUROC: {auroc:>.4f}")

def main():
    # set dev to gpu

    ambTestSet, ambTrainLoader, ambTestLoader, ambFeaturesDim = load_D1(1)

    #softmaxTrainTest(device,ambTrainLoader)

    print(f"Train size: {len(ambTrainLoader.dataset)}")
    print(f"Test size: {len(ambTestLoader.dataset)}")


    diabDataloader = createSklearnDataloader(load_diabetes(),[1,3,5])

    #modelsTest(ambTrainLoader,ambTestLoader, irisDataloader)

    lambdaTest(ambTrainLoader,ambTestLoader, diabDataloader)

main()