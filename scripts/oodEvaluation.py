import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F
from matplotlib import pyplot as plt


#train_device = "cuda:0"
train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_ood_datasets(true_dataset, ood_dataset):
    # Preprocess OoD dataset same as true dataset
    datasets = [true_dataset, ood_dataset]

    anomaly_targets = torch.cat(
        (torch.zeros(len(true_dataset)), torch.ones(len(ood_dataset)))
    )

    concat_datasets = torch.utils.data.ConcatDataset(datasets)

    dataloader = torch.utils.data.DataLoader(
        concat_datasets, batch_size=500, shuffle=False, num_workers=4, pin_memory=False
    )

    return dataloader, anomaly_targets


def loop_over_dataloader(model, dataloader, model_type):#standard_model, isSoftmax):
    if isinstance(model,list):
        for m in model:
            m.eval()
    else:
        model.eval()
    global train_device
    with torch.no_grad():
        scores = []
        accuracies = []
        for data, target in dataloader:
            #data = data.cuda()
            #target = target.cuda()
            data = data.to(train_device)
            target = target.to(train_device)

            if model_type.lower() == 'duq':
                output,_ = model(data)
                kernel_distance, pred = output.max(1)
                uncertainty = - kernel_distance ############ ask
            
            elif model_type.lower() == 'softmax':
                output = model(data)
                _, pred = output.max(1)
                uncertainty = torch.sum(output * torch.log(output+ 1e-10), dim=1)

            elif model_type.lower() == 'kan':
                output = model.forwardSoftmax(data) 
                _, pred = output.max(1)
                uncertainty = torch.sum(output * torch.log(output+ 1e-10), dim=1)
            
            else: ## embedings
                output = []
                for m in model:
                    output.append(m.forward(data))
                output = torch.stack(output, dim=0)
                output = torch.mean(output, dim=0)
                _, pred = output.max(1)
                uncertainty = torch.sum(output * torch.log(output+ 1e-10), dim=1)

            accuracy = pred.eq(target)
            accuracies.append(accuracy.cpu().numpy())

            scores.append(uncertainty.cpu().numpy())

    scores = np.concatenate(scores)
    accuracies = np.concatenate(accuracies)

    return scores, accuracies



def get_auroc_ood(true_dataset, ood_dataset, model, device, model_type):#standard_model=False, isSoftmax = False):
    global train_device
    train_device = device
    dataloader, anomaly_targets = prepare_ood_datasets(true_dataset, ood_dataset)

    scores, accuracies = loop_over_dataloader(model, dataloader, model_type)# standard_model, isSoftmax)

    #accuracy = np.mean(accuracies[: len(true_dataset)])
    roc_auc = roc_auc_score(anomaly_targets, scores)

    return roc_auc
