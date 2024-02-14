import torch.nn as nn

class DeepLineupLoss(nn.Module):
    def forward(self, preds, target):
        loss = 0
        for i in range(preds.size(0)):
            raise NotImplementedError()
            # pred_i = preds[i] 
            # target_i = target[i]
            # loss += torch.sum((pred_i - target_i)**2) 
        loss = loss / preds.size(0)
        return loss