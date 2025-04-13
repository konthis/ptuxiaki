from datasets import *
from models import *
from tools import *
from oodEvaluation import *
from sklearn.datasets import load_iris,load_diabetes
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

def softmaxTrain(trainloader, testloader,falseloader, model, optimizer, epochs):
    #scheduler = ReduceLROnPlateau(optimizer, 'min',patience=5)
    lrs = []
    auroc = []
    for e in tqdm(range(epochs),desc="Epochs"):
        trainAcc, trainAvgLoss = softmaxTrainStep(trainloader,model,nn.CrossEntropyLoss(),optimizer)
        testAcc, testAvgLoss= softmaxTest(testloader,model)

        auroc.append(get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader.dataset, model=model, device=device, standard_model=False, isSoftmax = True))
        #scheduler.step(testAvgLoss) ########
        #lrs.append(scheduler.get_last_lr())
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

def duqTrainStep(dataloader, model, optimizer, l):
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
    return accuracy, avgLoss 

def duqTrain(trainloader, testloader, falseloader, model, optimizer, l, epochs):
    scheduler = ReduceLROnPlateau(optimizer, 'min',patience=5)
    trainLosses = []
    testLosses  = []
    auroc = []
    lrs = []
    for _ in tqdm(range(epochs),desc="Epochs"):

        trainAcc, trainAvgLoss,= duqTrainStep(trainloader, model, optimizer, l)
        testAcc, testAvgLoss = duqTest(testloader, model)
        trainLosses.append(trainAvgLoss) 
        testLosses.append(testAvgLoss) 
        auroc.append(get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader.dataset, model=model, device=device, standard_model=False, isSoftmax = False))

        scheduler.step(testAvgLoss)
        lrs.append(scheduler.get_last_lr())
    #print(scheduler.get_last_lr())

    #plot([i for i in range(epochs)],'epochs',ylosses,'loss','r')
    return trainAcc, trainAvgLoss, trainLosses, testAcc, testAvgLoss, testLosses,auroc, lrs


def testDUQArchitectures(trainloader, testloader, falseloader, lrs, architectures, lam, epochs, modelsPerTest, std):
    results = []
    
    print("DUQ MODEL TEST")
    for a in architectures:
        print(f"ARCHITECTURE: {a}")
        for lr in lrs:
            print(f"Learning Rate: {lr}")
            for l in lam:
                print(f"Lambda: {l}")
                for s in std:
                    print(f"std: {s}")
                    models = []
                    trainAccs = []
                    trainAvgLosses = []
                    trainLosses = []
                    testAccs = []
                    testAvgLosses = []
                    testLosses = []
                    aurocResults = []
                    aurocProgressResults = []
                    lrsLog = []
                    for _ in range(modelsPerTest):
                        model = SimpleLinearNetwork_DUQ(a[0],a[1],a[2],a[3],s,initSigma=1).to(device)
                        models.append(model)
                        sigmaParam = [p for name, p in model.named_parameters() if 'sigma' in name]
                        othersParam = [p for name, p in model.named_parameters() if 'sigma' not in name]
                        optimizer = torch.optim.SGD([{"params":othersParam},{"params":sigmaParam,"lr":1e-3}],lr = lr)
                        #optimizer = torch.optim.SGD(model.parameters(),lr = lr)
                        trainAcc, trainAvgLoss, trainLoss, testAcc, testAvgLoss, testLoss, aurocProgress, lrLog = duqTrain(trainloader,testloader, falseloader,model,optimizer,l, epochs)

                        trainLosses.append(trainLoss)
                        testLosses.append(testLoss)


                        #testAcc, testAvgLoss = duqTest(testloader, model)
                        auroc = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader.dataset, model=model, device=device, standard_model=False, isSoftmax = False)

                        trainAccs.append(trainAcc)
                        trainAvgLosses.append(trainAvgLoss)
                        testAccs.append(testAcc)
                        testAvgLosses.append(testAvgLoss)
                        aurocProgressResults.append(aurocProgress)
                        aurocResults.append(auroc)
                        lrsLog.append(lrLog)

                    #e = [i for i in range(epochs)]
                    #plotx = [e,e]
                    #lossploty = [np.mean(np.array(trainLosses), axis=0),np.mean(np.array(testLosses), axis=0)]
                    #lrlr = np.mean(np.array(lrsLog),axis=0)
                    #plot(plotx,'epochs',lossploty,'losses',['b','r'],funcNames=['train loss','test loss'])
                    #plot(plotx,'epochs',[lrlr[:,0],lrlr[:,1]],'lr',['r','g'],['weight learning rate', 'sigma learning rate'],ylogscale=True)
                    #plot(e,'epochs',np.mean(np.array(aurocProgressResults),axis=0),'auroc','g','auroc')

                    trainAccs = np.array(trainAccs)
                    trainAvgLosses= np.array(trainAvgLosses)
                    testAccs = np.array(testAccs)
                    testAvgLosses = np.array(testAvgLosses)
                    aurocResults = np.array(aurocResults)
                    results.append([a, epochs,lr, l,s,  np.mean(trainAccs,axis=0), np.std(trainAccs,axis=0),
                                                        np.mean(trainAvgLosses,axis=0), np.std(trainAvgLosses,axis=0),
                                                        np.mean(testAccs,axis=0), np.std(testAccs,axis=0),
                                                        np.mean(testAvgLosses,axis=0), np.std(testAvgLosses,axis=0),
                                                        np.mean(aurocResults, axis=0), np.std(aurocResults,axis=0)])
                    print(f"Train Acc {100*results[-1][5]:>.1f}% std {100*results[-1][6]:>.1f}, AvgLoss {results[-1][7]:>.3f} std {results[-1][8]:>.3f}")
                    print(f"Test  Acc {100*results[-1][9]:>.1f}% std {100*results[-1][10]:>.1f}, AvgLoss {results[-1][11]:>.3f} std {results[-1][12]:>.3f}")
                    print(f"AUROC {results[-1][13]:>.3f} std {results[-1][14]:>.3f}")
                    print("-")
                    #### Save model
                    for i, m in enumerate(models):
                        torch.save([m.kwargs,m.state_dict()], f"../models/duq/duqBestModel{i}.pth")
                            
        print("\n")
    return results

def testSoftmaxArchitectures(trainloader, testloader,falseloader, lrs, architectures, epochs, modelsPerTest):
    results = []

    print("SOFTMAX MODEL TEST")
    for a in architectures:
        print(f"ARCHITECTURE {a}")
        for lr in lrs:
            print(f"Learning Rate: {lr}")
            models = []
            lrsLog = []
            trainAccs = []
            trainAvgLosses = []
            testAccs = []
            testAvgLosses = []
            aurocResults = []
            aurocProgressResults = []
            for _ in range(modelsPerTest):
                model = customLinearNetwork(len(a),a).to(device)
                models.append(model)
                optimizer = torch.optim.SGD(model.parameters(),lr = lr)
                trainAcc, trainAvgLoss, testAcc, testAvgLoss, aurocLog, lrLog = softmaxTrain(trainloader,testloader,falseloader, model,optimizer, epochs)
                trainAccs.append(trainAcc)
                lrsLog.append(lrLog)
                trainAvgLosses.append(trainAvgLoss)
                testAccs.append(testAcc)
                testAvgLosses.append(testAvgLoss)
                aurocProgressResults.append(aurocLog)
                auroc = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader.dataset, model=model, device=device, standard_model=False, isSoftmax = True)
                aurocResults.append(auroc)

            trainAccs = np.array(trainAccs)
            trainAvgLosses= np.array(trainAvgLosses)
            testAccs = np.array(testAccs)
            testAvgLosses = np.array(testAvgLosses)
            aurocResults = np.array(aurocResults)
            results.append([a, epochs, lr, np.mean(trainAccs,axis=0), np.std(trainAccs,axis=0),
                                    np.mean(trainAvgLosses,axis=0), np.std(trainAvgLosses,axis=0),
                                    np.mean(testAccs,axis=0), np.std(testAccs,axis=0),
                                    np.mean(testAvgLosses,axis=0), np.std(testAvgLosses,axis=0),
                                    np.mean(aurocResults, axis=0), np.std(aurocResults,axis=0)])
            #print(f"Train Acc {(100*np.mean(trainAccs,axis=0)):>.1f}% std {100*np.std(trainAccs,axis=0):>.1f}, AvgLoss {np.mean(trainAvgLosses,axis=0):>.3f} std {np.std(trainAvgLosses,axis=0):>.3f}")
            #print(f"Test  Acc {(100*np.mean(testAccs,axis=0)):>.1f}% std {100*np.std(testAccs,axis=0):>.1f}, AvgLoss {np.mean(testAvgLosses,axis=0):>.3f} std {np.std(trainAvgLosses,axis=0):>.3f}")
            print(f"Train Acc {100*results[-1][3]:>.1f}% std {100*results[-1][4]:>.1f}, AvgLoss {results[-1][5]:>.3f} std {results[-1][6]:>.3f}")
            print(f"Test  Acc {100*results[-1][7]:>.1f}% std {100*results[-1][8]:>.1f}, AvgLoss {results[-1][9]:>.3f} std {results[-1][10]:>.3f}")
            print(f"AUROC {results[-1][11]:>.3f} std {results[-1][12]:>.3f}")
            print("-")

            ##### Save model
            for i, m in enumerate(models):
                torch.save([m.kwargs,m.state_dict()], f"../models/softmax/softmaxModel{i}.pth")
        print("\n")
    
    return results
    
def testDeepEnsemblesArchitectures(trainloader, testloader,falseloader,lrs, architectures, epochs, modelsPerTest):
    ytrainloader = getDataloaderTargets(trainloader)
    ytestloader = getDataloaderTargets(testloader)

    results = []

    print("5-ENSEMBLE MODEL TEST")
    for a in architectures:
        print(f"ARCHITECTURE: {a}")
        for lr in lrs:
            print(f"Learning Rate: {lr}")
            trainAccs = []
            testAccs = []
            for j in range(modelsPerTest):
                models = [customLinearNetwork(len(a),a).to(device) for _ in range(5)]
                optimizers = [torch.optim.SGD(model.parameters(),lr = lr) for model in models]
                probsTrain = []
                probsTest = []
                aurocResults = []
                for i in range(5):
                    softmaxTrain(trainloader, testloader,falseloader,models[i],optimizers[i],epochs)
                    probsTrain.append(softmaxForward(models[i],trainloader))
                    probsTest.append(softmaxForward(models[i],testloader))


                for i, m in enumerate(models):
                    torch.save([m.kwargs,m.state_dict()], f"../models/deepensembles/deModel{j}{i}.pth")

                probsTrain = np.array(probsTrain)
                probsTest = np.array(probsTest)
                # get average logits for the 5 models of ensebmle
                probsTrain = np.mean(probsTrain,axis=0)
                probsTest = np.mean(probsTest,axis=0)
                auroc = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader.dataset, model=models, device=device, standard_model=False, isSoftmax = False)
                aurocResults.append(auroc)

                trainAccs.append(np.mean(np.argmax(probsTrain,axis=1) == ytrainloader))
                testAccs.append(np.mean(np.argmax(probsTest,axis=1) == ytestloader))

            trainAccs = np.array(trainAccs)
            testAccs = np.array(testAccs)
            results.append([a, epochs,lr,  np.mean(trainAccs,axis=0), np.std(trainAccs,axis=0),
                                        np.mean(testAccs,axis=0), np.std(testAccs,axis=0),
                                        np.mean(aurocResults, axis=0), np.std(aurocResults,axis=0)])
            #print(f"Train Acc {(100*np.mean(trainAccs,axis=0)):>.1f}% std {100*np.std(trainAccs,axis=0):>.1f}, AvgLoss {np.mean(trainAvgLosses,axis=0):>.3f} std {np.std(trainAvgLosses,axis=0):>.3f}")
            #print(f"Test  Acc {(100*np.mean(testAccs,axis=0)):>.1f}% std {100*np.std(testAccs,axis=0):>.1f}, AvgLoss {np.mean(testAvgLosses,axis=0):>.3f} std {np.std(trainAvgLosses,axis=0):>.3f}")
            #print(f"AUROC {np.mean(aurocResults, axis=0):>.3f} std {np.std(aurocResults,axis=0):>.3f}")
            print(f"Train Acc {100*results[-1][3]:>.1f}% std {100*results[-1][4]:>.1f}")
            print(f"Test  Acc {100*results[-1][5]:>.1f}% std {100*results[-1][6]:>.1f}")
            print(f"AUROC {results[-1][7]:>.3f} std {results[-1][8]:>.3f}")
            print("-")

        print("\n")
    return results

def testDUQHyperparams(trainloader, testloader, falseloader, lr=1e-3, lrSigma=1e-3, initSigma = 1 ,architecture=[3,32,16,3], lam=0.25, epochs=100, modelsPerTest=5, std=1e-2):
    if not isinstance(lrSigma,list):
        lrSigma = [lrSigma]
    if not isinstance(initSigma,list):
        initSigma = [initSigma]

    results = []
    
    print("DUQ MODEL TEST")
    for lrsig in lrSigma:
        print(f"Sigma Learning Rate: {lrsig}")
        for inits in initSigma:
            print(f"Initial sigma: {inits}")
            trainAccs = []
            trainAvgLosses = []
            testAccs = []
            testAvgLosses = []
            aurocResults = []
            for _ in range(modelsPerTest):
                model = SimpleLinearNetwork_DUQ(architecture[0],architecture[1],architecture[2],architecture[3],std,inits).to(device)
                sigmaParam = [p for name, p in model.named_parameters() if 'sigma' in name]
                othersParam = [p for name, p in model.named_parameters() if 'sigma' not in name]
                optimizer = torch.optim.SGD([{"params":othersParam},{"params":sigmaParam,"lr":lrsig}],lr = lr)


                #optimizer = torch.optim.SGD(model.parameters(),lr = lr)
                #trainAcc, trainAvgLoss = duqTrain(trainloader,testloader,falseloader,model,optimizer,lam, epochs)
                trainAcc, trainAvgLoss, trainLoss, testAcc, testAvgLoss, testLoss, aurocProgress, lrLog = duqTrain(trainloader,testloader, falseloader,model,optimizer,lam, epochs)
                #testAcc, testAvgLoss = duqTest(testloader, model)


                auroc = get_auroc_ood(true_dataset=testloader.dataset, ood_dataset=falseloader.dataset, model=model, device=device, standard_model=False, isSoftmax = False)

                trainAccs.append(trainAcc)
                trainAvgLosses.append(trainAvgLoss)
                testAccs.append(testAcc)
                testAvgLosses.append(testAvgLoss)
                aurocResults.append(auroc)

            trainAccs = np.array(trainAccs)
            trainAvgLosses= np.array(trainAvgLosses)
            testAccs = np.array(testAccs)
            testAvgLosses = np.array(testAvgLosses)
            aurocResults = np.array(aurocResults)
            results.append([architecture, lrsig, inits, np.mean(trainAccs,axis=0), np.std(trainAccs,axis=0),
                                                        np.mean(trainAvgLosses,axis=0), np.std(trainAvgLosses,axis=0),
                                                        np.mean(testAccs,axis=0), np.std(testAccs,axis=0),
                                                        np.mean(testAvgLosses,axis=0), np.std(testAvgLosses,axis=0),
                                                        np.mean(aurocResults, axis=0), np.std(aurocResults,axis=0)])
            print(f"Train Acc {100*results[-1][3]:>.1f}% std {100*results[-1][4]:>.1f}, AvgLoss {results[-1][5]:>.3f} std {results[-1][6]:>.3f}")
            print(f"Test  Acc {100*results[-1][7]:>.1f}% std {100*results[-1][8]:>.1f}, AvgLoss {results[-1][9]:>.3f} std {results[-1][10]:>.3f}")
            print(f"AUROC {results[-1][11]:>.3f} std {results[-1][12]:>.3f}")
            print("-")

        print("\n")
    return results


def main():
    # set dev to gpu

    ambTestSet, ambTrainLoader, ambTestLoader, ambFeaturesDim = load_D1(5)
    falseloader = createSklearnDataloader(load_diabetes(),[1,2,3])
    print(f"Train size: {len(ambTrainLoader.dataset)}")
    print(f"Test size: {len(ambTestLoader.dataset)}")
    print("----------------")

    print("1. Train DUQ")
    print("2. Train single Softmax")
    print("3. Train 5-ensemble")
    print("0. Exit")
    ans = str(input())
    # for answers like 123
    for a in ans:
        match a:
            case '1':
                epochs = 120
                #epochs = 200
                #epochs = 500
                print(f"epochs {epochs}")
                modelsPerTest = 5 # no of models per set of hyperparams, for stat signif.
                lrs = [1e-1]
                #lrs = [1e-4,1e-3,1e-2]
                #lrSigma = [1e-1,1e-2,1e-3,1e-4,1e-5]
                #lrSigma = [1e-2]
                #initSigma = [0.1]
                lrSigma = [1e-2,1e-3,1e-4,1e-5]
                initSigma = [0.1,0.3,0.5,1,2]
                architectures = [[3,32,16,3]]
                #architectures = [[3,32,16,3], [3,16,32,3]]
                #lam = [0, 0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, .25, .3, .5, 1]
                lam = [0.25]
                std = [1e-2]
                #std = [1e-4, 1e-3, 1e-2, 1e-1]
                duqResults = testDUQArchitectures(ambTrainLoader, ambTestLoader, falseloader,lrs,architectures,lam,epochs,modelsPerTest, std)
                columnsA = ["arch", "epochs","lr","lambda", "init std", "train mean acc", "train std acc", "train mean avgloss", "train std avgloss",
                            "test mean acc", "test std acc","test mean avgloss","test std avgloss", "auroc mean", 'auroc std']
                #duqResults = testDUQHyperparams(ambTrainLoader, ambTestLoader, falseloader,lrs[0],lrSigma,initSigma,
                #                                architectures[0],lam[0],epochs,modelsPerTest, std[0])
                columnsH = ["arch", "lrsigma","initsigma", "train mean acc", "train std acc", "train mean avgloss", "train std avgloss",
                            "test mean acc", "test std acc","test mean avgloss","test std avgloss", "auroc mean", 'auroc std']
                #saveToEXCEL(duqResults,columnsH,f"results/duqResults_s_lrs2_e{epochs}")
                saveToEXCEL(duqResults,columnsA,f"results/duqResults_best_e{epochs}")
            case '2':
                epochs = 120
                #epochs = 200
                modelsPerTest = 5 # no of models per set of hyperparams, for stat signif.
                lrs = [1e-1]
                architectures = [[3,16,32,3]]
                #architectures = [[3,32,16,3], [3, 16,32,3]]
                softresults = testSoftmaxArchitectures(ambTrainLoader, ambTestLoader,falseloader,lrs,architectures,epochs,modelsPerTest)
                columns = ["arch", "epochs","lr", "train mean acc", "train std acc", "train mean avgloss", "train std avgloss",
                            "test mean acc", "test std acc","test mean avgloss","test std avgloss", "auroc mean", 'auroc std']
                saveToEXCEL(softresults,columns,f"results/softmaxResults_a{len(architectures)}e{epochs}diab")
            case '3':
                epochs = 120
                modelsPerTest = 5 # no of models per set of hyperparams, for stat signif.
                lrs = [1e-1]
                #architectures = [[3,32,16,3],[3,16,32,3]]
                architectures = [[3,32,16,3]]
                ensembleresults = testDeepEnsemblesArchitectures(ambTrainLoader, ambTestLoader,falseloader,lrs,architectures,epochs,modelsPerTest)
                columns = ["arch", "epochs","lr", "train mean acc", "train std acc",
                            "test mean acc", "test std acc", "auroc mean", 'auroc std']
                saveToEXCEL(ensembleresults,columns,f"results/deepEnsembleResults_a{len(architectures)}e{epochs}diab")
            case '0':
                exit()
    
main()

