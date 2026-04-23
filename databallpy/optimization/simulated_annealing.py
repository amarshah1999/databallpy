# Annealer
import math
import random
from copy import deepcopy

from databallpy import Game
from databallpy.optimization.optimization import (
    Constraint,
    ObjectiveTerm,
    OptimizationAlgorithm,
    OptimizationResult,
)
from databallpy.utils.logging import create_logger

LOGGER = create_logger(__name__)


class SimulatedAnnealing(OptimizationAlgorithm):
    def __init__(
        self,
        game: Game,
        selected_frame_idx: int,
        objective_terms: list[ObjectiveTerm],
        weights: list[float],
        constraints: list[Constraint] = [],
        defending_players_to_optimize: list[str] | None = None,
        distance_perturbation: float = 0.2,
        num_iterations: int = 1000,
    ):
        # data
        self.game = game
        self.frame = game.tracking_data[
            game.tracking_data["frame"] == selected_frame_idx
        ].iloc[0]

        # annealing params, per https://www.geeksforgeeks.org/artificial-intelligence/what-is-simulated-annealing/
        self.distance_perturbation = distance_perturbation
        self.p_0 = 0.5
        self.T_0 = -100 / (math.log(self.p_0))
        self.T = self.T_0
        self.cooling_rate = 0.9
        self.num_iterations = num_iterations

        self.defending_players_to_optimize = (
            defending_players_to_optimize
            if defending_players_to_optimize
            else self.game.get_column_ids(
                team="home" if self.frame["team_possession"] == "away" else "away"
            )
        )

        self.objective_terms = objective_terms
        self.weights = weights
        self.constraints = constraints
        for constraint in self.constraints:
            constraint.compute_prerequisites(game=self.game, frame=self.frame)

    def perturbation(self, input_frame):
        proposed_new_frame = deepcopy(input_frame)
        # randomly choose 1 of the defenders
        # TODO see if we can cache the unselected players to avoid recomputing every time
        self.selected_players = random.sample(self.defending_players_to_optimize, 1)
        for player_id in self.selected_players:
            # move each player up to a maximal distance from their starting positions
            x_col = player_id + "_x"
            y_col = player_id + "_y"

            new_x_pos = proposed_new_frame[x_col] + random.uniform(
                -self.distance_perturbation, self.distance_perturbation
            )
            new_y_pos = proposed_new_frame[y_col] + random.uniform(
                -self.distance_perturbation, self.distance_perturbation
            )
            proposed_new_frame[y_col] = new_y_pos
            proposed_new_frame[x_col] = new_x_pos
            # check if the new position is reachable within max time to intercept (tti)
            if all(
                constraint.check(proposed_new_frame, player_id)
                for constraint in self.constraints
            ):
                return proposed_new_frame
            return input_frame

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
                LOGGER.info(
                    "Iteration %s / %s, best_score=%s",
                    i,
                    self.num_iterations,
                    best_score,
                )
                if last_checkpoint_score == best_score:
                    LOGGER.warning(
                        "No improvement found in last 200 iterations. Exiting early."
                    )
                    break
                last_checkpoint_score = best_score

        LOGGER.info("Finished simulated annealing. Best score=%s", best_score)
        return OptimizationResult(best_frame=best_solution, best_result=best_score)
