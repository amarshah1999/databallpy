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


class Constraint(ABC):

    def compute_prerequisites(self, game: Game = None, frame: pd.Series = None) -> None:
        return None
    @abstractmethod
    def check(self, proposed_new_frame, player_id) -> bool:
        raise NotImplementedError
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

class TTIConstraint(Constraint):
    # TTI Implementation from https://github.com/devinpleuler/analytics-handbook/blob/master/soccer_analytics_handbook.ipynb
    def __init__(self, max_time_to_intercept_seconds: float = 1, reaction_time: float = 0.1, max_velocity: float = 5.0):
        self.max_time_to_intercept_seconds = max_time_to_intercept_seconds
        self.reaction_time = reaction_time
        self.max_velocity = max_velocity

    #FixMe - this is a lazy implementation
    def compute_prerequisites(self, game: Game = None, frame: pd.Series = None) -> None:
        self.player_to_starting_pos_and_vel_map = frame[
            [c + "_x" for c in game.get_column_ids()]
            + [c + "_y" for c in game.get_column_ids()]
            + [c + "_vx" for c in game.get_column_ids()]
            + [c + "_vy" for c in game.get_column_ids()]
        ]

    def tti(self, origin, destination, velocity):
        u = (origin + velocity) - origin
        v = destination - origin
        u_mag = np.sqrt(np.sum(u**2, axis=-1))
        v_mag = np.sqrt(np.sum(v**2, axis=-1))
        dot_product = np.sum(u * v, axis=-1)
        angle = np.arccos(dot_product / (u_mag * v_mag))
        r_reaction = origin + velocity * self.reaction_time
        d = destination - r_reaction
        t = (
            u_mag * angle / np.pi
            + self.reaction_time
            + np.linalg.norm(d, axis=-1) / self.max_velocity
        )

        return t

    def check(self, proposed_new_frame, player_id) -> bool:
        x_col, y_col, vx_col, vy_col = [player_id + suffix for suffix in ["_x", "_y", "_vx", "_vy"]]
        origin = np.array([self.player_to_starting_pos_and_vel_map[x_col], self.player_to_starting_pos_and_vel_map[y_col]])
        velocity = np.array([self.player_to_starting_pos_and_vel_map[vx_col], self.player_to_starting_pos_and_vel_map[vy_col]])
        destination = np.array([proposed_new_frame[x_col], proposed_new_frame[y_col]])
        
        return self.tti(origin, destination, velocity) < self.max_time_to_intercept_seconds


def optimize_tracking_frame(
    game: Game,
    selected_frame_idx: int,
    objective_terms: list[ObjectiveTerm],
    weights: list[float],
    constraints: list[Constraint],
    algorithm: OptimizationAlgorithm,
) -> OptimizationResult:
    optimizer = algorithm(game, selected_frame_idx, objective_terms, weights, constraints)
    return optimizer.run()
