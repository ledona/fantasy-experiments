import torch.nn as nn


class DeepLineupLoss(nn.Module):
    """
    Loss is the difference in score between the predicted lineup and top lineup
    of a slate. If the predicted lineup is invalid then the loss is increased by 1
    for every additional invalid player
    """
    def __init__(self, target_cols: list[str], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.target_cols = target_cols

    def _calc_loss(self, pred, target):
        raise NotImplementedError()
        pred = preds[i]
        target = targets[i]
        if sum(preds[i]) != len(targets[i]):
            # too many players selected
            loss += sum(target[i])

    def forward(self, preds, targets):
        loss = 0
        for i in range(preds.size(0)):
            loss += self._calc_loss(preds[i], targets[i])
        loss = loss / preds.size(0)
        return loss
