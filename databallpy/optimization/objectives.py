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
        game: Game,
        frame: pd.Series,
        xt_array: np.ndarray | None = None,
    ):
        super().__init__(computation_type=ObjectiveType.GRID)
        self.grid = np.meshgrid(
            np.linspace(
                -game.pitch_dimensions[0] / 2, game.pitch_dimensions[0] / 2, 106
            ),
            np.linspace(-game.pitch_dimensions[1] / 2, game.pitch_dimensions[1] / 2, 68),
        )

        self.attacking_team = frame["team_possession"]
        self.defending_team = "home" if self.attacking_team == "away" else "away"
        self.defending_player_ids = game.get_column_ids(team=self.defending_team)

        self.attacking_team_influence = get_team_influence(
            frame,
            col_ids=game.get_column_ids(team=self.attacking_team),
            grid=self.grid,
            player_ball_distances=None,
        )
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
        frame: pd.Series,
        players_to_press: list[str] | None = None,
    ):
        super().__init__(computation_type=ObjectiveType.PLAYER)

        self.game = game
        self.players_to_press = (
            players_to_press
            if players_to_press
            else game.get_column_ids(team=frame["team_possession"])
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
            # compute the mean pressure on a player by dividing by number of players being pressed
            pressure_score += pressure / len(self.players_to_press)
        return pressure_score
