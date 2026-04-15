import numpy as np
import pandas as pd

from databallpy.game import Game
from databallpy.optimization.optimization import Constraint


class TTIConstraint(Constraint):

    def __init__(
        self,
        max_time_to_intercept_seconds: float = 1,
        reaction_time: float = 0.1,
        max_velocity: float = 5.0,
    ):
        self.max_time_to_intercept_seconds = max_time_to_intercept_seconds
        self.reaction_time = reaction_time
        self.max_velocity = max_velocity

    # FixMe - this is a lazy implementation
    def compute_prerequisites(self, game: Game, frame: pd.Series) -> None:
        self.player_to_starting_pos_and_vel_map = frame[
            [c + "_x" for c in game.get_column_ids()]
            + [c + "_y" for c in game.get_column_ids()]
            + [c + "_vx" for c in game.get_column_ids()]
            + [c + "_vy" for c in game.get_column_ids()]
        ]

    # TTI Implementation from https://github.com/devinpleuler/analytics-handbook/blob/master/soccer_analytics_handbook.ipynb
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
        x_col, y_col, vx_col, vy_col = [
            player_id + suffix for suffix in ["_x", "_y", "_vx", "_vy"]
        ]
        origin = np.array(
            [
                self.player_to_starting_pos_and_vel_map[x_col],
                self.player_to_starting_pos_and_vel_map[y_col],
            ]
        )
        velocity = np.array(
            [
                self.player_to_starting_pos_and_vel_map[vx_col],
                self.player_to_starting_pos_and_vel_map[vy_col],
            ]
        )
        destination = np.array([proposed_new_frame[x_col], proposed_new_frame[y_col]])

        return (
            self.tti(origin, destination, velocity) < self.max_time_to_intercept_seconds
        )
