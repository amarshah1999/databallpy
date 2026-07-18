import math
import unittest
from unittest.mock import MagicMock

import pandas as pd

from databallpy.optimization.optimization import (
    Constraint,
    ObjectiveTerm,
    ObjectiveType,
    OptimizationResult,
)
from databallpy.optimization.simulated_annealing import SimulatedAnnealing
from databallpy.utils.get_game import get_game


def _load_test_game():
    return get_game(
        tracking_data_loc="tests/test_data/tracab_td_test.dat",
        tracking_metadata_loc="tests/test_data/tracab_metadata_test.xml",
        tracking_data_provider="tracab",
        event_data_loc="tests/test_data/f24_test.xml",
        event_metadata_loc="tests/test_data/f7_test.xml",
        event_data_provider="opta",
        check_quality=False,
    )


class _ConstantObjective(ObjectiveTerm):
    def __init__(self, value: float):
        super().__init__(ObjectiveType.PLAYER)
        self.value = value

    def compute(self, input_frame: pd.Series) -> float:
        return self.value


class _FixedConstraint(Constraint):
    def __init__(self, result: bool):
        self.result = result
        self.checked_players = []

    def check(self, proposed_new_frame, player_id) -> bool:
        self.checked_players.append(player_id)
        return self.result


class TestSimulatedAnnealing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = _load_test_game()

    def setUp(self):
        self.game.get_column_ids = MagicMock(return_value=["home_34"])
        self.player_id = "home_34"

    def _make_sa(self, **overrides) -> SimulatedAnnealing:
        defaults = dict(
            game=self.game,
            selected_frame_idx=1,
            objective_terms=[_ConstantObjective(1.0)],
            weights=[1.0],
            constraints=[],
            defending_players_to_optimize=[self.player_id],
            num_iterations=10,
            patience=5,
            random_state=0,
            verbose=False,
        )
        defaults.update(overrides)
        return SimulatedAnnealing(**defaults)

    def test_init_sets_parameters(self):
        sa = self._make_sa(num_iterations=100, patience=20, distance_perturbation=0.3)

        self.assertEqual(sa.distance_perturbation, 0.3)
        self.assertEqual(sa.p_0, 0.5)
        self.assertAlmostEqual(sa.T_0, -100 / math.log(0.5))
        self.assertEqual(sa.T, sa.T_0)
        self.assertEqual(sa.cooling_rate, 0.9)
        self.assertEqual(sa.num_iterations, 100)
        self.assertEqual(sa.patience, 20)

    def test_init_log_interval(self):
        cases = [
            ("default_large", 100, None, 20),
            ("default_small", 3, None, 1),
            ("explicit", 100, 7, 7),
        ]
        for name, num_iterations, log_interval, expected in cases:
            with self.subTest(name=name):
                sa = self._make_sa(
                    num_iterations=num_iterations, log_interval=log_interval
                )
                self.assertEqual(sa.log_interval, expected)

    def test_init_mismatched_weights_raises(self):
        with self.assertRaises(ValueError):
            self._make_sa(objective_terms=[_ConstantObjective(1.0)], weights=[1.0, 2.0])

    def test_init_uses_explicit_defending_players(self):
        sa = self._make_sa(defending_players_to_optimize=["home_1", "home_2"])
        self.assertEqual(sa.defending_players_to_optimize, ["home_1", "home_2"])
        self.game.get_column_ids.assert_not_called()

    def test_init_default_defending_players_is_opposite_team(self):
        # the defending team is the one NOT in possession
        cases = [("away", "home"), ("home", "away")]
        for possession, expected_team in cases:
            with self.subTest(possession=possession):
                self.game.get_column_ids = MagicMock(return_value=["home_34"])
                original = self.game.tracking_data.loc[1, "team_possession"]
                self.game.tracking_data.loc[1, "team_possession"] = possession
                try:
                    sa = self._make_sa(defending_players_to_optimize=None)
                finally:
                    self.game.tracking_data.loc[1, "team_possession"] = original
                self.game.get_column_ids.assert_called_once_with(team=expected_team)
                self.assertEqual(sa.defending_players_to_optimize, ["home_34"])

    def test_compute_objective(self):
        cases = [
            ("single_term", [2.0], [3.0], 6.0),
            ("multiple_terms", [2.0, 4.0], [1.0, 0.5], 4.0),
            ("zero_weight", [5.0, 9.0], [0.0, 1.0], 9.0),
        ]
        for name, values, weights, expected in cases:
            with self.subTest(name=name):
                sa = self._make_sa(
                    objective_terms=[_ConstantObjective(v) for v in values],
                    weights=weights,
                )
                result = sa.compute_objective(self.game.tracking_data.loc[1].copy())
                self.assertAlmostEqual(result, expected)

    def test_perturbation(self):
        # (name, constraint, expect_moved) - with no/failing/passing constraints
        cases = [
            ("no_constraints", None, True),
            ("passing_constraint", _FixedConstraint(True), True),
            ("failing_constraint", _FixedConstraint(False), False),
        ]
        for name, constraint, expect_moved in cases:
            with self.subTest(name=name):
                constraints = [constraint] if constraint is not None else []
                sa = self._make_sa(constraints=constraints, distance_perturbation=0.2)

                frame = self.game.tracking_data.loc[1].copy()
                frame[f"{self.player_id}_x"] = 10.0
                frame[f"{self.player_id}_y"] = -5.0
                result = sa.perturbation(frame)

                if expect_moved:
                    # moved by at most distance_perturbation in each direction
                    self.assertLessEqual(abs(result[f"{self.player_id}_x"] - 10.0), 0.2)
                    self.assertLessEqual(
                        abs(result[f"{self.player_id}_y"] - (-5.0)), 0.2
                    )
                else:
                    # a failing constraint means the original frame is returned unchanged
                    self.assertIs(result, frame)
                    self.assertEqual(result[f"{self.player_id}_x"], 10.0)

                if constraint is not None:
                    self.assertEqual(constraint.checked_players, [self.player_id])

    def test_run_returns_result_with_best_score(self):
        sa = self._make_sa(
            objective_terms=[_ConstantObjective(5.0)],
            weights=[1.0],
            num_iterations=5,
            patience=100,
        )
        result = sa.run()

        self.assertIsInstance(result, OptimizationResult)
        self.assertIsInstance(result.best_frame, pd.Series)
        # a constant positive objective is accepted as the best score on iteration 1
        self.assertEqual(result.best_result, 5.0)

    def test_run_early_stopping_when_no_improvement(self):
        spy_objective = _ConstantObjective(0.0)
        spy_objective.compute = MagicMock(return_value=0.0)
        sa = self._make_sa(
            objective_terms=[spy_objective],
            weights=[1.0],
            num_iterations=1000,
            patience=2,
        )
        result = sa.run()

        self.assertIsInstance(result, OptimizationResult)
        # with no improvement, the run breaks at the first patience checkpoint (i == 2)
        self.assertEqual(spy_objective.compute.call_count, 2)
