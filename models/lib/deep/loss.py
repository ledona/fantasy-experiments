from typing import cast

import torch.nn as nn
import pandas as pd
from fantasy_py.lineup.constraint import SportConstraints
from fantasy_py.lineup.knapsack import KnapsackConstraint


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

    _lineup_slot_count: int
    """the number of slots in the expected lineup"""

    def __init__(
        self,
        input_cols: list[str],
        target_cols: list[str],
        constraints: SportConstraints,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.target_cols = target_cols
        # self.input_cols = input_cols
        self.constraints = constraints

        if isinstance(constraints.lineup_constraints, dict):
            self._lineup_slot_count = sum(constraints.lineup_constraints.values())
        elif isinstance(constraints.lineup_constraints[0], int):
            self._lineup_slot_count = sum(cast(list[int], constraints.lineup_constraints))
        else:
            self._lineup_slot_count = sum(
                constraint.max_count
                for constraint in cast(list[KnapsackConstraint], constraints.lineup_constraints)
            )

    def calc_loss(self, pred: list[int], target_df: pd.DataFrame):
        """
        calculate the loss for a predictioned lineup
        relative to the target/optimal information

        pred: list-like of 0|1, each element is for a player and corresponds to\
            a row in target_df. 1=that player is in the lineup
        target_df: dataframe with all information for the slate
        """
        top_score = target_df.query("`in-lineup`")["fpts-historic"].sum()
        if (player_count_diff := abs(sum(pred) - self._lineup_slot_count)) > 0:
            return top_score + player_count_diff
        pred_df = target_df.drop(columns='in-lineup').assign(**{'in-lineup': pred})
        raise NotImplementedError()

    def forward(self, preds, targets):
        loss = 0
        for i in range(preds.size(0)):
            target_df = pd.DataFrame(targets[i], columns=self.target_cols)
            loss += self.calc_loss(preds[i], target_df)
        loss = loss / preds.size(0)
        return loss
