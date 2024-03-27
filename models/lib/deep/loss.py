import math
import os
import sys
from itertools import chain, product
from typing import cast

import dask
import numpy as np
import pandas as pd
import torch
from fantasy_py import log
from fantasy_py.lineup.constraint import SportConstraints
from fantasy_py.lineup.create_lineups import KnapsackInputData
from fantasy_py.lineup.do_gen_lineup import create_solver
from fantasy_py.lineup.fantasy_cost_aggregate import (
    FCAPlayerDict,
    FCAPlayerTeamInventoryMixin,
    FCATeamDict,
)
from fantasy_py.lineup.knapsack import KnapsackConstraint, KnapsackIdentityMapping, KnapsackItem
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from .loader import DeepLineupDataset

dask.config.set(scheduler="processes", num_workers=math.floor(os.cpu_count() * 0.75))
_LOGGER = log.get_logger(__name__)


class PTInv(FCAPlayerTeamInventoryMixin):
    def __init__(
        self, players: None | dict[int, FCAPlayerDict], teams: None | dict[int, FCATeamDict]
    ):
        self._players = players or {}
        self._teams = teams or {}

    def get_mi_player(self, id_: int):
        return self._players[id_]

    def get_mi_team(self, id_: int):
        return self._teams[id_]


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
        self._best_lineupclear_found = ("", -sys.float_info.max)
        self.target_cols = list(dataset.target_cols)
        self.constraints = constraints
        self._solver = create_solver(self.constraints, silent=True)

        self._pos_cols = {
            col: idx for idx, col in enumerate(self.target_cols) if col.startswith("pos:")
        }
        assert self._pos_cols, "no player positions found in target cols"

        self._cost_col_idx = self.target_cols.index("cost")
        self._player_idx = (
            self.target_cols.index("player_id") if "player_id" in self.target_cols else None
        )
        self._team_idx = self.target_cols.index("team_id")
        self._hist_score_col_idx = self.target_cols.index("fpts-historic")
        self._in_top_lineup_col_idx = self.target_cols.index("in-lineup")
        self._opp_id = self.target_cols.index("opp_id")

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

    def _gen_knapsack_data(self, probs: Tensor, target: Tensor):
        """
        The returned knapsack input data is in the same order as the probs and target
        tensor

        returns tuple of knapsack input data, player/team FCA inventory
        """
        knap_items: list[KnapsackItem] = []
        knap_mappings: list[KnapsackIdentityMapping] = []
        pid_to_i: dict[int, int] = {}
        tid_to_i: dict[int, int] = {}
        players: dict[int, FCAPlayerDict] = {}
        teams: dict[int, FCATeamDict] = {}

        for knap_idx, (value, player_info) in enumerate(zip(probs, target)):
            team_id = int(player_info[self._team_idx])
            if team_id == 0:
                # padding
                continue

            classes = [
                pos_col.split(":", 1)[1]
                for pos_col, idx in self._pos_cols.items()
                if player_info[idx]
            ]
            knap_item = KnapsackItem(classes, float(player_info[self._cost_col_idx]), float(value))
            knap_items.append(knap_item)
            if self._player_idx is not None and not torch.isnan(player_info[self._player_idx]):
                pid = int(player_info[self._player_idx])
                assert pid not in players and pid not in pid_to_i
                knap_mapping = KnapsackIdentityMapping.player_mapping(pid, float(value))
                pid_to_i[pid] = knap_idx
                players[pid] = {
                    "player_id": pid,
                    "team": str(team_id),
                    "team_id": team_id,
                    "cost": float(player_info[self._cost_col_idx]),
                    "positions": classes,
                    "opp_abbr": str(player_info[self._opp_id]),
                }
            else:
                assert team_id not in teams and team_id not in tid_to_i
                knap_mapping = KnapsackIdentityMapping.team_mapping(team_id, float(value))
                tid_to_i[team_id] = knap_idx
                teams[team_id] = {
                    "id": team_id,
                    "abbr": str(team_id),
                    "cost": float(player_info[self._cost_col_idx]),
                    "positions": classes,
                    "opp_abbr": str(player_info[self._opp_id]),
                }
            knap_mappings.append(knap_mapping)

        mpt_inv = PTInv(players, teams)

        return KnapsackInputData(knap_mappings, knap_items, pid_to_i, tid_to_i), mpt_inv

    def _calc_slate_score(self, pred: Tensor, target: Tensor):
        _LOGGER.info("preds score: started")
        knapsack_input, pt_inv = self._gen_knapsack_data(pred, target)

        solutions = self._solver.solve(
            knapsack_input.data,
            1,
            self.constraints.viability_testers,
            pt_inv,
            knapsack_input.mappings,
        )

        assert len(solutions) == 1
        pt_indices = list(chain(*solutions[0].items))
        _LOGGER.info("preds score:finished")
        return float(target[pt_indices, self._hist_score_col_idx].sum())

    def _calc_score(self, pred: Tensor, target: Tensor):
        top_lineups_players_mask = target[:, :, self._in_top_lineup_col_idx] == 1
        top_lineups_masked_scores = (
            target[:, :, self._hist_score_col_idx] * top_lineups_players_mask.float()
        )
        top_lineups_scores = top_lineups_masked_scores.sum(dim=1)

        dask_bag = dask.bag.from_sequence(zip(pred, target), npartitions=16)

        def func(bag_item: tuple[Tensor, Tensor]):
            return self._calc_slate_score(*bag_item)

        # if log.PROGRESS_REQUESTED:
        #     print()
        #     pbar = ProgressBar()
        #     pbar.register()
        pred_scores = Tensor(dask_bag.map(func).compute())

        assert not (pred_scores > top_lineups_scores).any()

        mean_score = torch.mean(pred_scores / top_lineups_scores)
        return mean_score

    def forward(self, preds: Tensor, targets: Tensor):
        """
        Calculate the policy reward and policy gradients, based on the policy predictions.
        Policy rewards/gradients returned instead of loss to support REINFORCE training

        preds: tensor of shape (batch_size, inventory_size). batch_size=number of slates in batch\
            inventory_size=number of players in each slate. Values are 0-1, and are the probability\
            that each player is in the slate.
        targets: tensor of shage (batch_size, inventory_size, feature-count) which describes\
            the batch slates.
        returns (gradients, reward)
        """
        reward = self._calc_score(preds, targets)
        log_prob = torch.log(preds)
        gradients = log_prob * reward
        return gradients, float(reward)
