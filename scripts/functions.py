import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivationFunctions(nn.Module):
    def __init__(self,gamma):
        super(ActivationFunctions,self).__init__()
        self.gamma = gamma
    def GaussianRBF(self, x):
        return torch.exp(-self.gamma*torch.square(x))

    def RBF_SiLU(self, x):
        return x*torch.exp(-self.gamma*torch.square(x))

    def RBF_Swish(self, x):
        return F.silu(x)*x*torch.exp(-self.gamma*torch.square(x))

def gradPenalty2sideCalc(x, ypred):
    gradients = torch.autograd.grad(
            outputs=ypred,
            inputs=x,
            grad_outputs=torch.ones_like(ypred),
            create_graph=True
        )[0]
    #gradPenalty = ((gradients.norm(2,dim=1)**2 - 1)**2).mean()
    gradPenalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradPenalty


def gradPenalty2sideLogits(x,ypred,tau=0.04,eps=1e-7):
    norm = torch.norm(ypred, p=2, dim=1, keepdim=True).clamp(min=eps)
    normalized_logits = ypred / (tau * norm)
    gradients = torch.autograd.grad(
            outputs=normalized_logits,
            inputs=x,
            grad_outputs=torch.ones_like(ypred),
            create_graph=True
        )[0]
    
    gradPenalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    #print(gradPenalty)
    return gradPenalty


class LogitNormLoss(nn.Module):
    def __init__(self, tau: float = 0.04, eps: float = 1e-7):
        super().__init__()
        self.tau = tau
        self.eps = eps
 
    def forward(self, logits: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
        # Compute the L2 norm per sample
        norm = torch.norm(logits, p=2, dim=1, keepdim=True).clamp(min=self.eps)
        
        # Normalize logits
        normalized_logits = logits / (self.tau * norm)
 
        # Compute cross-entropy on normalized logits
        return F.cross_entropy(normalized_logits, targets)
