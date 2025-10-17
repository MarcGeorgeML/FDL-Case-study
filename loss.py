import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=[0.5,0.5], gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1-targets)*(1-probs)
        alpha_t = targets*self.alpha[1] + (1-targets)*self.alpha[0]
        loss = alpha_t * (1 - p_t)**self.gamma * bce
        return loss.mean() if self.reduction=='mean' else loss.sum()


