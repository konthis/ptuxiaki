import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
import numpy as np

from sklearn.decomposition import PCA
import os
from torchvision.io import decode_image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from PIL import Image, ImageFile
from torchvision.datasets import FashionMNIST
from torchvision.datasets import MNIST
from medmnist import PathMNIST,ChestMNIST,DermaMNIST,OCTMNIST



from sklearn.datasets import load_iris,load_breast_cancer,load_wine


# 0: Non-infectious SIRS
# 1: Sepsis
# 2: Septic Shock

BATCH_SIZE = 16

def load_D1(seed,path='../datasets', only_biomarkers=True, binary=False):

    df = pd.read_excel(f"{path}/ambrosia.xlsx")
    df = pd.get_dummies(df, columns=['Sex'], prefix='Sex')
    df.drop(columns=['Blood culture microorganism 1', 'Blood culture microorganism 2'], inplace=True)

    df.dropna(inplace=True)
    df = pd.get_dummies(df, columns=['Blood culture result'], prefix='Blood_culture_result')

    label_encoder = LabelEncoder()
    df['AMBROSSIA'] = label_encoder.fit_transform(df['AMBROSSIA'])


    if only_biomarkers:
        df.drop(columns=['Age', 'SOFA', 'Blood_culture_result_Positive', 'Blood_culture_result_Negative', 'Sex_Female', 'Sex_Male', 'CSG  '], inplace=True)

    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)

    X, y = df.drop(columns=['AMBROSSIA']), df['AMBROSSIA']

    # Combine the two classes (1: Sepsis, 2: Septic Shock) into a single class (1)
    if binary:
        y = y.replace(2, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    scalers = []
    for column in ['PCR (mg/dL)', 'PCT  (ng/mL)', 'IL6 (pg/mL)', 'Age', 'SOFA']:
        if column in X_train.columns:
            scaler = StandardScaler()
            X_train[column] = scaler.fit_transform(X_train[[column]])
            X_test[column] = scaler.transform(X_test[[column]])
            scalers.append(scaler)

    train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32),
                                  torch.tensor(y_train.values, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32),
                                 torch.tensor(y_test.values, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    dimensions = X_train.shape[1]
    return test_dataset, train_loader, test_loader, dimensions

def createSklearnDataloader(dataset, featureIdxs):
    datasetX = dataset['data'][:,featureIdxs]
    scaler = StandardScaler()
    datasetX = scaler.fit_transform(datasetX)
    datasetX = torch.autograd.Variable(torch.tensor(datasetX,dtype=torch.float))
    datasetY = torch.autograd.Variable(torch.tensor(dataset['target'], dtype=torch.long))
    dataloader = DataLoader(TensorDataset(datasetX,datasetY), batch_size=1000, shuffle=True)
    return dataloader

def loadFashionMNIST():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    fmnist_test = FashionMNIST(root='../datasets/data', train=False, download=True, transform=transform)
    fmnist_train = FashionMNIST(root='../datasets/data', train=True, download=True, transform=transform)
    trainloader = DataLoader(fmnist_train, batch_size=64, shuffle=True)
    testloader  = DataLoader(fmnist_test, batch_size=64, shuffle=True)
    return trainloader, testloader

def loadMNIST():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_test = MNIST(root='../datasets/data', train=False, download=True, transform=transform)
    mnist_train = MNIST(root='../datasets/data', train=True, download=True, transform=transform)
    trainloader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    testloader  = DataLoader(mnist_test, batch_size=64, shuffle=True)
    return trainloader, testloader

def loadPathMNIST():
    #transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((28,28)),
    #                                transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),transforms.Lambda(torch.flatten)])
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
                ])

    mnist_train= PathMNIST(root='../datasets/data', split="train", download=True, transform=transform)
    mnist_test = PathMNIST(root='../datasets/data', split="test", download=True, transform=transform)
    trainloader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    testloader  = DataLoader(mnist_test, batch_size=64, shuffle=True)
    return trainloader, testloader

def loadChestMNIST():
    #transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((28,28)),transforms.ToTensor(), 
    #                                transforms.Normalize((0.5,), (0.5,)),transforms.Lambda(torch.flatten)])
    transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((28, 28)),  
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                transforms.Lambda(lambda x: torch.flatten(x))])
    mnist_train = ChestMNIST(root='../datasets/data', split="train", download=True, transform=transform)
    mnist_test  = ChestMNIST(root='../datasets/data', split="test", download=True, transform=transform)
    trainloader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    testloader  = DataLoader(mnist_test, batch_size=64, shuffle=True)
    return trainloader, testloader

def loadDermaMNIST():
    #transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((28,28)),transforms.ToTensor(), 
    #                                transforms.Normalize((0.5,), (0.5,)),transforms.Lambda(torch.flatten)])
    transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                transforms.Lambda(lambda x: torch.flatten(x))])
    mnist_train= DermaMNIST(root='../datasets/data', split="train", download=True, transform=transform)
    mnist_test = DermaMNIST(root='../datasets/data', split="test", download=True, transform=transform)
    trainloader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    testloader  = DataLoader(mnist_test, batch_size=64, shuffle=True)
    return trainloader, testloader

def loadOCTMNIST():
    data_transform = transforms.Compose([transforms.ToTensor()])
    octmnist = OCTMNIST(root='../datasets/data', split='train', transform=data_transform, download=True)
    octmnist_test = OCTMNIST(root='../datasets/data', split='test', transform=data_transform, download=True)

    X_train = octmnist.imgs
    y_train = octmnist.labels
    X_test = octmnist_test.imgs
    y_test = octmnist_test.labels

    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    #return trainloader, testloader

def loadPathMNISTandNoised(root='../datasets/data'):
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                ])
    noisetransform = transforms.Compose([
                transforms.ToTensor(),
                GaussianNoiseTransform(mean=0,std=0.5,patch_only=True),
                transforms.Normalize((0.5,), (0.5,))
                ])

    traindataset = PathMNIST(root=root, split="train", transform=transform)
    testdataset = PathMNIST(root=root, split="test", transform=transform)
    noisedtraindataset = PathMNIST(root=root, split="train", transform=noisetransform)
    trainloader = DataLoader(traindataset, batch_size=64, shuffle=True)
    testloader  = DataLoader(testdataset, batch_size=64, shuffle=True)
    noiseddataloader = DataLoader(noisedtraindataset, batch_size=64, shuffle=False)
    return trainloader, testloader, noiseddataloader

class GaussianNoisedDataset(Dataset):
    def __init__(self, dataset, mean=3.0, std=5):
        self.dataset = dataset
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        noisy_data = data + torch.randn_like(data) * self.std + self.mean
        noisy_data = torch.clamp(noisy_data, data.min(), data.max()) # keep range
        return noisy_data, target

class CustomImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label



def printDataloader(dataloader):
    #transform = transforms.Compose([
    #transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,))
    #])
    #noisetransform = transforms.Compose([
    #transforms.ToTensor(),
    #GaussianNoiseTransform(patch_only=True),
    #transforms.Normalize((0.5,), (0.5,))
    #])

    #mnist_train= PathMNIST(root='../datasets/data', split="train",transform=transform)
    #noisemnist = PathMNIST(root='../datasets/data', split="train",transform=noisetransform)
    #dataloader= DataLoader(noisemnist, batch_size=4, shuffle=True)

    #trainloader,testloader,noisedloader = loadPathMNISTandNoised()

    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    def imshow(img, title=None):
        img = img / 2 + 0.5     # Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        if title is not None:
            plt.title(title)

    # Create figure
    plt.figure(figsize=(10, 4))

    img_num = 4
    # Plot each image in the batch
    for i in range(img_num):
        plt.subplot(1, img_num, i+1)  # 1 row, N columns, current subplot
        imshow(images[i], f'Label: {labels[i]}')

    plt.tight_layout()
    plt.show()

def add_gaussian_noise(image, mean=0, std=0.1, patch_only=True, patch_size=(9, 9)):
    # Convert PyTorch tensor to NumPy if needed
    if isinstance(image, torch.Tensor):
        device = image.device
        image_np = image.cpu().numpy()
    else:
        image_np = image.copy()
    
    # Ensure image is in (C, H, W) format
    if image_np.ndim == 2:
        image_np = np.expand_dims(image_np, axis=0)  # (H, W) -> (1, H, W)
    c, h, w = image_np.shape
    
    if patch_only:
        x = np.random.randint(0, w - patch_size[1])
        y = np.random.randint(0, h - patch_size[0])
        
        noise = np.random.normal(mean, std, (c, patch_size[0], patch_size[1]))
        
        image_np[:, y:y+patch_size[0], x:x+patch_size[1]] += noise
    else:
        noise = np.random.normal(mean, std, image_np.shape)
        image_np += noise
    image_np = np.clip(image_np, -1, 1)
    
    if isinstance(image, torch.Tensor):
        return torch.from_numpy(image_np).to(device)
    else:
        return image_np.astype(np.float32)

def createNoisedPathMNIST(root='../datasets/data'):
    #_,testloader = loadPathMNIST()
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                ])
    testdataset = PathMNIST(root=root, split="test",transform=transform)
    images = np.stack([x[0].numpy() for x in testdataset])
    labels = np.array([x[1] for x in testdataset])
    noisyimages = [add_gaussian_noise(img,mean=0,std=1.0) for img in images]
    noisydataset = CustomImageDataset(np.array(noisyimages), labels)

    saveDataset(noisydataset,f'{root}/PathMNIST/noisedtest.pth')
    
    return noisydataset

def createPathMNIST():
    #_,testloader = loadPathMNIST()
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                ])
    #traindataset = PathMNIST(root='../datasets/data', split="train",transform=transform)
    testdataset = PathMNIST(root='../datasets/data', split="train",transform=transform)
    images = np.stack([x[0].numpy() for x in testdataset])
    labels = np.array([x[1] for x in testdataset])
    trainset = CustomImageDataset(np.array(images), labels)

    saveDataset(trainset,f'../datasets/data/PathMNIST/train.pth')
    #return noisydataset

class GaussianNoiseTransform:
    def __init__(self, mean=0.1, std=0.5, patch_only=False, patch_size=(10, 10)):
        self.mean = mean
        self.std = std
        self.patch_only = patch_only
        self.patch_size = patch_size
    
    def __call__(self, img):
        img_np = img.numpy().squeeze()
        noisy_img = add_gaussian_noise(img_np, self.mean, self.std, self.patch_only, self.patch_size)
        return torch.from_numpy(noisy_img)

def pcaDataloader(n_components=0.95):

    traindataset = loadDataset('../datasets/data/PathMNIST/train.pth')
    testdataset  = loadDataset('../datasets/data/PathMNIST/test.pth')
    noiseddataset = loadDataset('../datasets/data/PathMNIST/noisedtest.pth')

    
    X_train = traindataset.images
    y_train = traindataset.labels
    X_test = testdataset.images
    y_test = testdataset.labels
    X_noise= noiseddataset.images
    y_noise = noiseddataset.labels
    #print(np.max(X_test[5000]-X_noise[5000]))

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_noise= X_noise.reshape(X_noise.shape[0], -1)
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    y_noise= y_noise.squeeze()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_noise_scaled = scaler.transform(X_noise)


    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    X_noise_pca = pca.transform(X_noise_scaled)

    X_train_tensor = torch.FloatTensor(X_train_pca)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_pca)
    y_test_tensor = torch.LongTensor(y_test)
    X_noise_tensor = torch.FloatTensor(X_noise_pca)
    y_noise_tensor = torch.LongTensor(y_noise)

    # Create DataLoader
    traindataset = TensorDataset(X_train_tensor, y_train_tensor)
    testdataset = TensorDataset(X_test_tensor, y_test_tensor)
    noiseddataset = TensorDataset(X_noise_tensor, y_noise_tensor)

    #batch_size = 64
    #trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    #testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)
    #noisedloader = DataLoader(noiseddataset, batch_size=batch_size, shuffle=False)

    saveDataset(traindataset,f'../datasets/data/PathMNIST_{pca.n_components_}/train.pth')
    saveDataset(testdataset,f'../datasets/data/PathMNIST_{pca.n_components_}/test.pth')
    saveDataset(noiseddataset,f'../datasets/data/PathMNIST_{pca.n_components_}/noisedtest.pth')
    print("Done")

def load_noisedD1(seed,root,binary):
    _,trainloader,_,_ = load_D1(seed,root,binary=binary)
    noiseddataset = GaussianNoisedDataset(trainloader.dataset)
    return DataLoader(noiseddataset, batch_size=BATCH_SIZE, shuffle=True)

def loadAllDataloaders(root,binary):
    #datasetNames = int(input("Choose ood datasets(ex 123):\n1.Wine\n2.iris\n3.Breast Cancer\n4.Noised Ambrosia\n:"))

    _, ambTrainLoader, ambTestLoader, _= load_D1(2,root,binary=binary)
    falseloader  = createSklearnDataloader(load_wine(),[0,1,2])
    falseloader2 = createSklearnDataloader(load_iris(),[0,1,2]) ######## for some reason 
    falseloader3 = createSklearnDataloader(load_breast_cancer(),[0,1,2])
    falseloader4 = load_noisedD1(2,root,binary)

    return ambTrainLoader,ambTestLoader,falseloader,falseloader2,falseloader3,falseloader4

def loadImageDataloaders():
    traindataset = loadDataset('../datasets/data/PathMNIST_353/train.pth')
    testdataset  = loadDataset('../datasets/data/PathMNIST_353/test.pth')
    falsedataset = loadDataset('../datasets/data/PathMNIST_353/noisedtest.pth')

    trainloader = DataLoader(traindataset, batch_size=64, shuffle=True)
    testloader = DataLoader(testdataset, batch_size=64, shuffle=False)
    falseloader = DataLoader(falsedataset, batch_size=64, shuffle=False)
    #falseloader2 = DataLoader(falsedataset2, batch_size=64, shuffle=False)
    #falseloader3 = DataLoader(falsedataset3, batch_size=64, shuffle=False)
    falseloader2=falseloader3 = False

    #t1 = iter(testloader)
    #t2 = iter(falseloader)
    #images, labels = next(t1)
    #images2, labels2 = next(t2)
    #print(torch.mean(images-images2))
    #print(labels-labels2)

    return trainloader,testloader,falseloader,falseloader2,falseloader3

def saveDataset(dataset,path):
    torch.save(dataset,path)

def loadDataset(path):
    return torch.load(path)

if __name__ == "__main__":
    createNoisedPathMNIST()
    #createPathMNIST()
    noiseddataset= loadDataset('../datasets/data/PathMNIST/noisedtest.pth')
    printDataloader(DataLoader(noiseddataset, batch_size=64, shuffle=False))
    pcaDataloader(0.95)