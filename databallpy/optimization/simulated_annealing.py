# Annealer
import random
import math
import pandas as pd
import numpy as np
from databallpy.features.pitch_control import get_team_influence
from databallpy.schemas.tracking_data import TrackingData
from databallpy.optimization.optimization import (
    OptimizationAlgorithm,
    OptimizationResult,
    ObjectiveTerm,
)
from copy import deepcopy
from databallpy import Game
from databallpy.utils.utils import sigmoid

import pickle

from pathlib import Path


class SimulatedAnnealing(OptimizationAlgorithm):
    def __init__(
        self,
        game: Game,
        selected_frame_idx: int,
        objective_terms: list[ObjectiveTerm],
        weights: list[float],
        distance_perturbation=0.2,
        num_iterations=1000,
        max_tti=1,
    ):
        # data
        self.game = game
        self.frame = game.tracking_data[
            game.tracking_data["frame"] == selected_frame_idx
        ].iloc[0]

        self.attacking_team = self.frame["team_possession"]
        self.defending_team = (
            "home" if self.frame["team_possession"] == "away" else "away"
        )

        # annealing params
        self.distance_perturbation = distance_perturbation  # how much to move the player
        self.p_0 = 0.5
        self.T_0 = -100 / (math.log(self.p_0))
        self.T = self.T_0
        self.cooling_rate = 0.9
        self.num_iterations = num_iterations
        self.max_tti = max_tti

        self.grid = np.meshgrid(
            np.linspace(
                -self.game.pitch_dimensions[0] / 2,
                self.game.pitch_dimensions[0] / 2,
                106,
            ),
            np.linspace(
                -self.game.pitch_dimensions[1] / 2, self.game.pitch_dimensions[1] / 2, 68
            ),
        )

        self.defending_player_ids = self.game.get_column_ids(team=self.defending_team)
        self.attacking_player_ids = self.game.get_column_ids(team=self.attacking_team)

        self.player_to_starting_pos_and_vel_map = self.frame[
            [c + "_x" for c in game.get_column_ids()]
            + [c + "_y" for c in game.get_column_ids()]
            + [c + "_vx" for c in game.get_column_ids()]
            + [c + "_vy" for c in game.get_column_ids()]
        ]

        self.objective_terms = objective_terms
        self.weights = weights


    # TTI Implementation from https://github.com/devinpleuler/analytics-handbook/blob/master/soccer_analytics_handbook.ipynb
    def tti(self, origin, destination, velocity, reaction_time, max_velocity=5.0):
        u = (origin + velocity) - origin
        v = destination - origin
        u_mag = np.sqrt(np.sum(u**2, axis=-1))
        v_mag = np.sqrt(np.sum(v**2, axis=-1))
        dot_product = np.sum(u * v, axis=-1)
        angle = np.arccos(dot_product / (u_mag * v_mag))
        r_reaction = origin + velocity * reaction_time
        d = destination - r_reaction
        t = (
            u_mag * angle / np.pi
            + reaction_time
            + np.linalg.norm(d, axis=-1) / max_velocity
        )

        return t

    def perturbation(self, input_frame):
        new_frame = deepcopy(input_frame)
        # randomly choose 1 of the defenders
        # TODO see if we can cache the unselected players to avoid recomputing every time
        self.selected_players = random.sample(self.defending_player_ids, 1)
        for c in self.selected_players:
            # move each player up to a maximal distance from their starting positions
            x_col = c + "_x"
            y_col = c + "_y"
            vx_col = c + "_vx"
            vy_col = c + "_vy"
            player_initial_x_pos = self.player_to_starting_pos_and_vel_map[x_col]
            player_initial_y_pos = self.player_to_starting_pos_and_vel_map[y_col]
            player_initial_vx = self.player_to_starting_pos_and_vel_map[vx_col]
            player_initial_vy = self.player_to_starting_pos_and_vel_map[vy_col]
            new_x_pos = new_frame[x_col] + random.uniform(
                -self.distance_perturbation, self.distance_perturbation
            )
            new_y_pos = new_frame[y_col] + random.uniform(
                -self.distance_perturbation, self.distance_perturbation
            )

            # check if the new position is reachable within max time to intercept (tti)
            if (
                self.tti(
                    np.array([player_initial_x_pos, player_initial_y_pos]),
                    np.array([new_x_pos, new_y_pos]),
                    np.array([player_initial_vx, player_initial_vy]),
                    0.1,
                )
                < self.max_tti
            ):
                new_frame[y_col] = new_y_pos
                new_frame[x_col] = new_x_pos

        return new_frame

    def compute_objective(self, input_frame):
        objective_total = 0
        for i, objective_term in enumerate(self.objective_terms):
            score = objective_term.compute(input_frame)
            objective_total += score * self.weights[i]

        # take the overall sum of geospatial terms
        return objective_total

    def run(self):
        best_score = 0
        best_solution = deepcopy(self.frame)
        latest_frame = deepcopy(self.frame)
        last_checkpoint_score = 0
        T = self.T
        print("running annealer")
        for i in range(1, self.num_iterations):
            perturbed_frame = self.perturbation(latest_frame)
            new_score = self.compute_objective(perturbed_frame)
            # take the new result if it's better, or randomly take a worse result with decaying probability
            if (new_score > best_score) or math.exp(
                (new_score - best_score) / T
            ) > random.random():
                latest_frame = perturbed_frame
            if new_score > best_score:
                best_score = new_score
                best_solution = perturbed_frame
                latest_frame = perturbed_frame
            T = T * self.cooling_rate
            if i % 200 == 0:
                print(f"iteration {i} / {self.num_iterations} best_score {best_score}")
                if last_checkpoint_score == best_score:
                    print("No improvement found in last 200 iterations... exiting.")
                    break
                last_checkpoint_score = best_score

        print(f"best score {best_score}")
        return OptimizationResult(best_frame=best_solution, best_result=best_score)
