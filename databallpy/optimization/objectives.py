import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from databallpy.features.pitch_control import get_team_influence
from databallpy.game import Game
from databallpy.optimization.optimization import ObjectiveTerm, ObjectiveType
from databallpy.schemas.tracking_data import TrackingData
from databallpy.utils.utils import sigmoid
from scipy.ndimage import zoom

XT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "open_play_xT.npy"


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
        if xt_array is None:
            open_play_xt = np.load(XT_MODEL_PATH)
            # we are using (y, x) orientation instead of (x, y) so that it matches
            self.xt_array = zoom(open_play_xt, (106 / 264, 68 / 196), order=1).T

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
