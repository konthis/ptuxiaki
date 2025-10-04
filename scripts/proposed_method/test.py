# python3 -m scripts.proposed_method.test
# exec like that

from torch.optim.lr_scheduler import ReduceLROnPlateau
from scripts.proposed_method.models import *
from scripts.datasets import *
from scripts.proposed_method.train import *
from scripts.functions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = 'datasets'
#testDataset, trainLoader, testLoader, dimensions = load_D1(2,root)
#falseLoader = load_noisedD1(2,False,root)
trainLoader,testLoader,falseLoader1,falseLoader2,falseLoader3,falseLoader4 = loadAllDataloaders(root,False)

#model = FastKAN([3,16,16]).to(device)
#model = EfficientKAN([3,16,3]).to(device)
#model = DUQ([3,32,16,3],1e-2,0.5).to(device)
#model = SoftmaxNet([3,32,16,3],nn.ReLU()).to(device)


lossFunction = LogitNormLoss()
#lossFunction = nn.CrossEntropyLoss()
netType = 'kanduq'
#netType = 'duq'
numClasses = 3
gradPenaltyL = 0.0
epochs = 50
numTestModels = 5

archs = [[3,8,16,3]]
for ar in archs:
    print(f"Arch {ar}")
    for _ in range(numTestModels):
        model = KANDUQ(ar,1,1).to(device)
        #model = DUQ(ar,0.1,1).to(device)

        #sigmaParam = [p for name, p in model.named_parameters() if 'sigma' in name]
        #othersParam = [p for name, p in model.named_parameters() if 'sigma' not in name]
        #optimizer = torch.optim.SGD([{"params":othersParam},{"params":sigmaParam,"lr":1e-1}],lr = 1e-1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
        #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

        networkTrain(netType,model,optimizer,scheduler,lossFunction,trainLoader,testLoader,falseLoader1,numClasses,gradPenaltyL,epochs)
        #print(sigmaParam)


