import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from databallpy.optimization.objectives import (
    PressureObjective,
    WeightedPitchControlObjective,
)
from databallpy.optimization.optimization import ObjectiveType
from databallpy.utils.get_game import get_game

GRID_SHAPE = (68, 106)


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


class TestWeightedPitchControlObjective(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = _load_test_game()

    def setUp(self):
        self.game.get_column_ids = MagicMock(return_value=["home_1", "away_1"])
        self.frame = self.game.tracking_data.loc[1].copy()
        self.frame["team_possession"] = "home"
        self.xt_array = np.full(GRID_SHAPE, 0.5)

    def test_init_away_flips_xt_array(self):
        self.frame["team_possession"] = "away"
        with patch(
            "databallpy.optimization.objectives.get_team_influence",
            return_value=np.ones(GRID_SHAPE),
        ):
            objective = WeightedPitchControlObjective(
                self.game, self.frame, xt_array=self.xt_array
            )

        self.assertEqual(objective.attacking_team, "away")
        self.assertEqual(objective.defending_team, "home")
        np.testing.assert_array_equal(objective.xt_array, np.fliplr(self.xt_array))

    def test_init_loads_default_xt_array_when_none(self):
        with (
            patch(
                "databallpy.optimization.objectives.get_team_influence",
                return_value=np.ones(GRID_SHAPE),
            ),
            patch(
                "databallpy.optimization.objectives.np.load",
                return_value=np.ones((264, 196)),
            ) as mock_load,
        ):
            objective = WeightedPitchControlObjective(self.game, self.frame)

        mock_load.assert_called_once()
        # loaded array is zoomed to the grid resolution and transposed to (y, x)
        self.assertEqual(objective.xt_array.shape, GRID_SHAPE)

    def test_compute(self):
        xt_array = np.full(GRID_SHAPE, 0.5)
        total_xt = float(np.sum(xt_array))
        cases = [
            ("defending_dominates", 1.0, 2.0, "home", total_xt),
            ("attacking_dominates", 2.0, 1.0, "home", 0.0),
            ("equal_influence", 1.0, 1.0, "home", total_xt / 2),
            ("away_possession", 1.0, 2.0, "away", total_xt),
        ]
        for name, attacking, defending, team_possession, expected in cases:
            with self.subTest(name=name):
                frame = self.frame.copy()
                frame["team_possession"] = team_possession
                with patch(
                    "databallpy.optimization.objectives.get_team_influence",
                    side_effect=[
                        np.full(GRID_SHAPE, attacking),
                        np.full(GRID_SHAPE, defending),
                    ],
                ):
                    objective = WeightedPitchControlObjective(
                        self.game, frame, xt_array=xt_array
                    )
                    result = objective.compute(frame)
                self.assertAlmostEqual(result, expected)


class TestPressureObjective(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = _load_test_game()

    def setUp(self):
        self.frame = self.game.tracking_data.loc[1].copy()
        self.frame["team_possession"] = "home"

    def test_init_default_players_to_press(self):
        self.game.get_column_ids = MagicMock(return_value=["home_1", "home_2"])
        objective = PressureObjective(self.game, self.frame)

        self.assertEqual(objective.computation_type, ObjectiveType.PLAYER)
        self.assertIs(objective.game, self.game)
        self.assertEqual(objective.players_to_press, ["home_1", "home_2"])
        self.game.get_column_ids.assert_called_once_with(team="home")

    def test_init_explicit_players_to_press(self):
        players = ["away_5", "away_9"]
        objective = PressureObjective(self.game, self.frame, players_to_press=players)
        self.assertEqual(objective.players_to_press, players)

    def test_compute(self):
        # pressure is the mean of the per-player pressure over the pressed players
        cases = [
            ("two_players", ["home_1", "home_2"], [2.0, 4.0], 3.0),
            ("single_player", ["home_1"], [7.5], 7.5),
            ("uniform_pressure", ["home_1", "home_2", "home_3"], [3.0, 3.0, 3.0], 3.0),
        ]
        for name, players, pressures, expected in cases:
            with self.subTest(name=name):
                objective = PressureObjective(
                    self.game, self.frame, players_to_press=players
                )
                with patch(
                    "databallpy.optimization.objectives.TrackingData"
                ) as mock_tracking_data:
                    instance = mock_tracking_data.return_value
                    instance.index = [self.frame.name]
                    instance.get_pressure_on_player.side_effect = pressures
                    result = objective.compute(self.frame)
                self.assertAlmostEqual(result, expected)
