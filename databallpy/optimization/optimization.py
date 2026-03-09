from abc import ABC, abstractmethod
from databallpy.game import Game
from dataclasses import dataclass
import pandas as pd
from enum import Enum
import pickle
from pathlib import Path
from databallpy.features.pitch_control import get_team_influence
from databallpy.utils.utils import sigmoid
import numpy as np
from databallpy.schemas.tracking_data import TrackingData

BASE_DIR = Path(__file__).resolve().parent


class ObjectiveType(str, Enum):
    GRID = "grid"  # computed for each grid cell
    PLAYER = "player"  # computed for each player


class ObjectiveTerm:
    def __init__(self, computation_type: ObjectiveType):
        self.computation_type = computation_type

    def compute(self, input_frame: pd.Series) -> float:
        raise NotImplementedError


class WeightedPitchControlObjective(ObjectiveTerm):
    # Computes the net pitch control for the defending team over the attacking team
    def __init__(
        self,
        grid: np.meshgrid,
        attacking_team: str,
        defending_team: str,
        defending_player_ids: list[str],
        attacking_team_influence: np.meshgrid,
        xt_array: np.ndarray | None = None,
    ):
        super().__init__(computation_type=ObjectiveType.GRID)
        self.grid = grid
        self.attacking_team = attacking_team
        self.defending_team = defending_team
        self.defending_player_ids = defending_player_ids
        self.attacking_team_influence = attacking_team_influence
        if not xt_array:
            with open(BASE_DIR / "xTArray.pkl", "rb") as f:
                self.xt_array = pickle.load(f)
        else:
            self.xt_array = xt_array
        if self.attacking_team == "away":
            self.xt_array = np.fliplr(self.xt_array)

    def compute(
        self,
        input_frame: pd.Series,
    ) -> float:
        team_influence_defending = get_team_influence(
            input_frame,
            col_ids=self.defending_player_ids,
            grid=self.grid,
            player_ball_distances=None,
        )
        # +ve is defending team, -ve is attacking team
        net_sigmoid_diff = sigmoid(
            team_influence_defending - self.attacking_team_influence, d=100
        )  # this makes the sigmoid steeper and more binary

        return sum(sum(self.xt_array * net_sigmoid_diff))


class PressureObjective(ObjectiveTerm):
    def __init__(
        self,
        game: Game,
        players_to_press: list[str] | None = None,
        attacking_player_ids: list[str] | None = None,
    ):
        if not attacking_player_ids and not players_to_press:
            raise ValueError(
                "Either players_to_press or attacking_player_ids must be provided"
            )

        super().__init__(computation_type=ObjectiveType.PLAYER)

        self.game = game
        self.players_to_press = (
            players_to_press if players_to_press else attacking_player_ids
        )

    def compute(self, input_frame: pd.Series) -> float:
        # pressure method only works on tracking data object, need to temporarily reconstruct it with the new data
        pressure_score = 0
        temp_tracking_df = pd.DataFrame(input_frame).T
        temp_tracking_df = TrackingData(
            temp_tracking_df.astype(self.game.tracking_data.dtypes)
        )

        for attacking_player in self.players_to_press:
            pressure = temp_tracking_df.get_pressure_on_player(
                temp_tracking_df.index[0], attacking_player, [106, 68], d_front=9
            )
            # compute the mean pressure on a player by dividing by number of players in p
            pressure_score += pressure / len(self.players_to_press)
        return pressure_score


@dataclass
class OptimizationResult:
    best_frame: pd.Series
    best_result: float


class OptimizationAlgorithm(ABC):
    @abstractmethod
    def __init__(
        self,
        game: Game,
        selected_frame_idx: int,
        objective_terms: list[ObjectiveTerm],
        weights: list[float],
    ):
        raise NotImplementedError

    @abstractmethod
    def run(self) -> OptimizationResult:
        raise NotImplementedError


class Constraints:
    pass


class Filters:
    pass


# def optimize_tracking_frame(
#     game: Game,
#     selected_frame_idx: int,
#     objective: ObjectiveFunction,
#     algorithm: OptimizationAlgorithm, # --> in the beginning only SimulatedAnnealing
#     constraints: Constraints, # Within optimization algo
#     filters: Filters # The variables the optimization algo is allowed to change
# ) -> OptimizationResult:
#     optimizer = OptimizationAlgorithm(game, selected_frame_idx)
#     return optimizer.run()
