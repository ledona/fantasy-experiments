import torch.nn as nn


class DeepLineupLoss(nn.Module):
    """
    Loss is the difference between the top lineup score and the predicted lineup score.
    Invalid lineups will have a negative score defined by the following rules. The rules
    are defined to prioritize for optimizing away from various types of lineup construction
    errors.

    1. If the number of players in the lineup is not equal to the required number
       (regardless of position) then the score is negative the number of missing or excess
       players
    2. If the number of players is correct but the positions are not then the score is
       negative .5 * the number of positions/slots not fillable by the lineup
    3. If the lineup is overbudget then the score is
       -(amount over budget) / (median cost of all players)
    4. -(number of failed viability tests)


    the difference in score between the predicted lineup and top lineup
    of a slate. If the predicted lineup is invalid then the loss is increased by 1
    for every additional invalid player
    """

    def __init__(
        self, input_cols: list[str], target_cols: list[str], constraints, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.target_cols = target_cols
        self.input_cols = input_cols
        self.lineup_constraints = constraints.lineup_constraints

        # if isinstance(lineup_constraints, dict):
        #     lineup_slot_count = sum(lineup_constraints.values())
        # elif isinstance(lineup_constraints, list):
        #     lineup_slot_count = sum(lineup_constraints)
        # else:
        #     lineup_slot_count = sum(constraint.max_count for constraint in lineup_constraints)

    def calc_loss(self, pred, target):
        """
        calculate the loss for a predictioned lineup
        relative to the target/optimal information
        """
        raise NotImplementedError()
        if sum(preds[i]) != len(targets[i]):
            # too many players selected
            loss += sum(target[i])

        raise NotImplementedError()

    def forward(self, preds, targets):
        loss = 0
        for i in range(preds.size(0)):
            loss += self.calc_loss(preds[i], targets[i])
        loss = loss / preds.size(0)
        return loss
