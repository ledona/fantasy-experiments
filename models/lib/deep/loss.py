import sys
from itertools import chain
from typing import cast

import dask
import torch
from fantasy_py import log
from fantasy_py.lineup.constraint import SportConstraints
from fantasy_py.lineup.create_lineups import KnapsackInputData
from fantasy_py.lineup.deep import DeepLineupDataset
from fantasy_py.lineup.do_gen_lineup import create_solver
from fantasy_py.lineup.fantasy_cost_aggregate import (
    FCAPlayerDict,
    FCAPlayerTeamInventoryMixin,
    FCATeamDict,
)
from fantasy_py.lineup.knapsack import KnapsackConstraint, KnapsackIdentityMapping, KnapsackItem

_LOGGER = log.get_logger(__name__)


class PTInv(FCAPlayerTeamInventoryMixin):
    def __init__(
        self,
        sport: str,
        players: None | dict[int, FCAPlayerDict],
        teams: None | dict[int, FCATeamDict],
    ):
        self.sport_abbr = sport
        self._players = players or {}
        self._teams = teams or {}

    def get_mi_player(self, id_: int):
        return self._players[id_]

    def get_mi_team(self, id_: int):
        return self._teams[id_]


class DeepLineupLoss(torch.nn.Module):
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

    _best_lineup_found: tuple[str, float]
    """description of the best lineup found, tuple of (description, score)"""

    def __init__(
        self,
        sport: str,
        dataset: DeepLineupDataset,
        constraints: SportConstraints,
        *args,
        **kwargs,
    ) -> None:
        """
        cost_penalty_divider: used for the over budget penalty
        """
        super().__init__(*args, **kwargs)
        self.sport = sport
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

    def _update_best_lineup(self, reason, score):
        if score <= self._best_lineup_found[1]:
            return
        _LOGGER.info("New best lineup found. score=%f desc='%s'", score, reason)
        self._best_lineup_found = (reason, score)

    def _gen_knapsack_data(self, probs: torch.Tensor, target: torch.Tensor):
        """
        The returned knapsack input data is in the same order as the probs and target
        tensor

        returns tuple[knapsack input data, player/team FCA inventory]
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

        mpt_inv = PTInv(self.sport, players, teams)

        return KnapsackInputData(knap_mappings, knap_items, pid_to_i, tid_to_i), mpt_inv

    def _calc_slate_score(self, pred: torch.Tensor, target: torch.Tensor):
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
        return float(target[pt_indices, self._hist_score_col_idx].sum())

    def calc_score(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred: tensor of dims [S, P] where S is the number of samples and P is the number\
            of players per sample OR [P] if the prediction is for a single sample
        target: tensor with target data for all samples in pred

        returns for a single prediction, the score for that prediction, for a batch\
            the mean of the scores across the batch
        """
        # top_lineups_players_mask = target[:, :, self._in_top_lineup_col_idx] == 1
        # top_lineups_masked_scores = (
        #     target[:, :, self._hist_score_col_idx] * top_lineups_players_mask.float()
        # )
        # top_lineups_scores = top_lineups_masked_scores.sum(dim=1)

        if pred.ndim == 1:
            assert target.ndim == 2, "Expecting a single target tensor"
            return self._calc_slate_score(pred, target)

        assert pred.ndim == 2 and target.ndim == 3, "Expecting preds and targets in batch"
        dask_bag = dask.bag.from_sequence(zip(pred, target), npartitions=16)

        def func(bag_item: tuple[torch.Tensor, torch.Tensor]):
            return self._calc_slate_score(*bag_item)

        pred_scores = torch.Tensor(dask_bag.map(func).compute())
        return pred_scores.mean()

        # assert not (pred_scores > top_lineups_scores).any()

        # mean_score = torch.mean(pred_scores / top_lineups_scores)
        # return mean_score

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
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
        # REINFORCE reward
        reward = self.calc_score(preds, targets)
        log_prob = torch.log(preds)
        gradients = -log_prob * reward
        return gradients, float(reward)

    def backwards_(self, preds: torch.Tensor, loss_tensor: torch.Tensor):
        """
        apply whatever was gradients were returned in forward() to the model
        """
        # REINFORCE gradient update
        preds.backward(loss_tensor)
