from itertools import product
from typing import cast

import numpy as np
import pandas as pd
import torch.nn as nn
from fantasy_py.lineup.constraint import SportConstraints
from fantasy_py.lineup.knapsack import KnapsackConstraint
from scipy.optimize import linear_sum_assignment

from .loader import DeepLineupDataset


class DeepLineupLoss(nn.Module):
    """
    Loss is the difference between the top lineup score and the predicted lineup score.
    Invalid lineups will have a negative score defined by the following rules. The rules
    are defined to prioritize for optimizing away from various types of lineup construction
    errors.

    1. If the number of players in the lineup is not equal to the required number
       (regardless of position) then the score is negative the number of missing or excess
       players
    2. If the number of players is correct but the lineup is invalid the score is
       -(.5 * the number of positions/slots not filled by the lineup)
    3. If the lineup is overbudget then the score is
       -(amount over budget) / cost_penalty_divider
    4. -(number of failed viability tests)

    the difference in score between the predicted lineup and top lineup
    of a slate. If the predicted lineup is invalid then the loss is increased by 1
    for every additional invalid player
    """

    _lineup_slot_count: int
    """the number of slots in the expected lineup"""

    _lineup_slot_pos: list[set[str]]
    """list of lineup slot positions based on constraints, each item
    is a list of positions that can be placed in the lineup position"""

    _cost_penalty_divider: float

    def __init__(
        self,
        dataset: DeepLineupDataset,
        constraints: SportConstraints,
        *args,
        **kwargs,
    ) -> None:
        """
        cost_penalty_divider: used for the over budget penalty
        """
        super().__init__(*args, **kwargs)
        self._cost_penalty_divider = dataset.cost_oom
        self.target_cols = dataset.target_cols
        self.constraints = constraints
        self._pos_cols = [col for col in self.target_cols if col.startswith("pos:")]
        assert self._pos_cols, "no player positions found in target cols"

        if isinstance(constraints.lineup_constraints, dict):
            self._lineup_slot_pos = []
            self._lineup_slot_count = sum(constraints.lineup_constraints.values())
            for pos, slots in constraints.lineup_constraints.items():
                self._lineup_slot_pos += [{pos} if isinstance(pos, str) else set(pos)] * slots
        elif isinstance(constraints.lineup_constraints[0], int):
            self._lineup_slot_count = len(constraints.lineup_constraints)
            self._lineup_slot_pos = []
            for pos, slots in enumerate(cast(list[int], constraints.lineup_constraints)):
                self._lineup_slot_pos += [{str(pos)}] * slots
        else:
            self._lineup_slot_pos = []
            self._lineup_slot_count = 0
            for constraint in cast(list[KnapsackConstraint], constraints.lineup_constraints):
                self._lineup_slot_count += constraint.max_count
                self._lineup_slot_pos += [
                    (
                        {str(constraint.classes)}
                        if isinstance(constraint.classes, (int, str))
                        else {str(pos) for pos in constraint.classes}
                    )
                ] * constraint.max_count

        assert len(self._lineup_slot_pos) == self._lineup_slot_count

        self._max_loss = dataset.sample_df_len - self._lineup_slot_count

    def _valid_lineup(self, pred_df: pd.DataFrame):
        """check that the predicted lineup is valid"""
        lineup_player_pos: dict[str, set[str]] = {}

        def player_pos_recorder(row):
            if pd.isna(row.player_id):
                return
            lineup_player_pos[f"p-{row.player_id}"] = {
                pos_col.split(":", 1)[1] for pos_col in self._pos_cols if row[pos_col]
            }

        pred_df.apply(player_pos_recorder, axis=1)
        assert len(lineup_player_pos) == self._lineup_slot_count

        # Create cost matrix
        cost_matrix = np.zeros((self._lineup_slot_count, self._lineup_slot_count))
        for (slot_i, slot_positions), (player_i, player_positions) in product(
            enumerate(self._lineup_slot_pos), enumerate(lineup_player_pos.values())
        ):
            if player_positions & slot_positions:
                cost_matrix[slot_i, player_i] = 1

        # solve
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
        unmatched_players = sum(cost_matrix[row_i, col_i] for row_i, col_i in zip(row_ind, col_ind))
        return self._lineup_slot_count - unmatched_players

    def _test_viabilities(self, df: pd.DataFrame):
        if self.constraints.knapsack_viability_testers is None:
            return 0

        failures = 0
        for tester in self.constraints.viability_testers:
            if not tester.is_valid(df):
                failures += 1

        return failures

    def calc_score(self, pred: list[int], target_df: pd.DataFrame):
        """
        calculate the score of the predicted lineup

        pred: list-like of 0|1, each element is for a player and corresponds to\
            a row in target_df. 1=that player is in the lineup
        target_df: dataframe with all information for the slate, including 'in-lineup'\
            column with 1 for players in the optimal lineup
        """
        if (player_count_diff := abs(sum(pred) - self._lineup_slot_count)) > 0:
            return -player_count_diff

        # dataframe just of the predicted lineup
        pred_df = (
            target_df.drop(columns="in-lineup").assign(**{"in-lineup": pred}).query("`in-lineup`")
        )

        # is this a valid lineup?
        if (failed_slots := self._valid_lineup(pred_df)) > 0:
            return -0.5 * failed_slots

        if (over_budget_by := pred_df.cost.sum() - self.constraints.budget) > 0:
            return -over_budget_by / self._cost_penalty_divider

        if (failed_viability_tests := self._test_viabilities(pred_df)) > 0:
            return -failed_viability_tests

        return pred_df["fpts-historic"].sum()

    def forward(self, preds, targets):
        """
        returns the mean loss across all predicted lineups, loss range is 0 to 1
        """
        scaled_adjusted_total_score = 0
        for i in range(preds.size(0)):
            target_df = pd.DataFrame(targets[i], columns=self.target_cols)
            score = self.calc_score(preds[i], target_df)
            adjusted_score = score + self._max_loss
            top_score = target_df.query("`in-lineup`")["fpts-historic"].sum()
            adjusted_top_score = top_score + self._max_loss
            scaled_adjusted_score = adjusted_score / adjusted_top_score
            scaled_adjusted_total_score += scaled_adjusted_score
        mean_adjusted_score = scaled_adjusted_total_score / preds.size(0)
        loss = 1 - mean_adjusted_score
        assert 0 <= loss <= 1

        return loss
