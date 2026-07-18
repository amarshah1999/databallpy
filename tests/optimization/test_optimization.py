import unittest
from unittest.mock import MagicMock

import pandas as pd

from databallpy.optimization.optimization import (
    Constraint,
    ObjectiveTerm,
    ObjectiveType,
    OptimizationAlgorithm,
    OptimizationResult,
    optimize_tracking_frame,
)
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


class _DummyAlgorithm(OptimizationAlgorithm):
    def __init__(
        self,
        game,
        selected_frame_idx,
        objective_terms,
        weights,
        constraints=None,
    ):
        super().__init__(
            game=game,
            selected_frame_idx=selected_frame_idx,
            objective_terms=objective_terms,
            weights=weights,
            constraints=constraints,
        )

    def run(self):
        return OptimizationResult(best_frame=self.frame, best_result=0.0)


class TestOptimizationAlgorithm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = _load_test_game()

    def test_init_stores_attributes(self):
        terms = [_ConstantObjective(1.0)]
        weights = [1.0]
        algorithm = _DummyAlgorithm(self.game, 1, terms, weights)

        self.assertIs(algorithm.game, self.game)
        self.assertEqual(algorithm.objective_terms, terms)
        self.assertEqual(algorithm.weights, weights)
        # frame is the single row selected from the tracking data
        expected_frame = self.game.tracking_data.loc[[1]].iloc[0]
        self.assertTrue(algorithm.frame.equals(expected_frame))

    def test_init_defaults_constraints_to_empty_list(self):
        algorithm = _DummyAlgorithm(self.game, 1, [_ConstantObjective(1.0)], [1.0])
        self.assertEqual(algorithm.constraints, [])

    def test_init_mismatched_weights_raises(self):
        cases = [
            ("too_few_weights", [_ConstantObjective(1.0)], []),
            (
                "too_many_weights",
                [_ConstantObjective(1.0)],
                [1.0, 2.0],
            ),
        ]
        for name, terms, weights in cases:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    _DummyAlgorithm(self.game, 1, terms, weights)


class TestOptimizeTrackingFrame(unittest.TestCase):
    def test_runs_algorithm_and_returns_its_result(self):
        game = MagicMock()
        terms = [_ConstantObjective(1.0)]
        weights = [1.0]
        constraints = []
        expected_result = OptimizationResult(
            best_frame=pd.Series(dtype=float), best_result=42.0
        )

        algorithm = MagicMock()
        algorithm.__name__ = "SpyAlgorithm"  # optimize_tracking_frame logs this
        algorithm.return_value.run.return_value = expected_result

        result = optimize_tracking_frame(
            game=game,
            selected_frame_idx=3,
            objective_terms=terms,
            weights=weights,
            constraints=constraints,
            algorithm=algorithm,
            num_iterations=5,
            random_state=1,
        )

        # the extra kwargs must be forwarded to the algorithm constructor
        algorithm.assert_called_once_with(
            game=game,
            selected_frame_idx=3,
            objective_terms=terms,
            weights=weights,
            constraints=constraints,
            num_iterations=5,
            random_state=1,
        )
        algorithm.return_value.run.assert_called_once_with()
        self.assertIs(result, expected_result)


