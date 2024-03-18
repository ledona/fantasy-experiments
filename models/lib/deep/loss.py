import sys
from itertools import product
from typing import cast

import numpy as np
import pandas as pd
import torch
from fantasy_py import log
from fantasy_py.lineup.constraint import SportConstraints
from fantasy_py.lineup.knapsack import KnapsackConstraint
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from .loader import DeepLineupDataset

_LOGGER = log.get_logger(__name__)


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

    _best_lineup_found: tuple[str, float]
    """description of the best lineup found, tuple of (description, score)"""

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
        self._best_lineup_found = ("", -sys.float_info.max)
        self.target_cols = list(dataset.target_cols)
        self.constraints = constraints
        self._pos_cols = [col for col in self.target_cols if col.startswith("pos:")]
        assert self._pos_cols, "no player positions found in target cols"

        self._cost_col_idx = self.target_cols.index("cost")
        self._hist_score_col_idx = self.target_cols.index("fpts-historic")
        self._in_top_lineup_col_idx = self.target_cols.index("in-lineup")

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
        return self._lineup_slot_count - cast(int, unmatched_players)

    def _test_viabilities(self, df: pd.DataFrame):
        if self.constraints.knapsack_viability_testers is None:
            return 0

        failures = 0
        for tester in self.constraints.viability_testers:
            if not tester.is_valid(df):
                failures += 1

        return failures

    def _update_best_lineup(self, reason, score):
        if score <= self._best_lineup_found[1]:
            return
        _LOGGER.info("New best lineup found. score=%f desc='%s'", score, reason)
        self._best_lineup_found = (reason, score)

    def _calc_slate_score(self, pred: Tensor, target: Tensor):
        raise NotImplementedError("create gen_lineup parameters")

        raise NotImplementedError("# call gen_lineup to create optimal lineup")

        raise NotImplementedError("# calculate historic score for optimal lineup")

    # def _calc_pred_scores(self, preds: Tensor, targets: Tensor):
    #     pred_probs, pred_indices = torch.topk(pred, self._lineup_slot_count)
    #     absolute_scores = target[
    #         torch.arange(pred.size(0)).unsqueeze(-1),
    #         pred_indices,
    #         self._hist_score_col_idx,
    #     ].sum(dim=1)
    #     return absolute_scores

    def _calc_score(self, pred: Tensor, target: Tensor):
        top_lineups_players_mask = target[:, :, self._in_top_lineup_col_idx] == 1
        top_lineups_masked_scores = (
            target[:, :, self._hist_score_col_idx] * top_lineups_players_mask.float()
        )
        top_lineups_scores = top_lineups_masked_scores.sum(dim=1)

        pred_scores = Tensor([self._calc_slate_score(pred, target) for pred, target in zip(pred, target)])
        # pred_scores = self._calc_pred_scores(pred, target)

        mean_score = torch.mean(pred_scores / top_lineups_scores)
        return mean_score

    def forward(self, preds: Tensor, targets: Tensor):
        """
        preds: tensor of shape (batch_size, inventory_size). batch_size=number of slates in batch\
            inventory_size=number of players in each slate. Values are 0-1, and are the probability\
            that each player is in the slate.
        targets: tensor of shage (batch_size, inventory_size, feature-count) which describes\
            the batch slates.
        returns (policy_gradient, reward), policy gradient is the gradient to use to update model\
            weights. Reward is the reward for the batch.
        """
        reward = self._calc_score(preds, targets)
        log_prob = torch.log(preds)
        policy_gradient = log_prob * reward
        return policy_gradient, float(reward)
