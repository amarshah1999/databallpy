from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from databallpy.game import Game
from databallpy.utils.logging import create_logger

LOGGER = create_logger(__name__)


class ObjectiveType(str, Enum):
    GRID = "grid"  # computed for each grid cell
    PLAYER = "player"  # computed for each player


class ObjectiveTerm:
    def __init__(self, computation_type: ObjectiveType):
        self.computation_type = computation_type

    def compute(self, input_frame: pd.Series) -> float:
        raise NotImplementedError


@dataclass
class OptimizationResult:
    best_frame: pd.Series
    best_result: float


class Constraint(ABC):
    def compute_prerequisites(self, game: Game = None, frame: pd.Series = None) -> None:
        return None

    @abstractmethod
    def check(self, proposed_new_frame, player_id)  -> bool:
        raise NotImplementedError


class OptimizationAlgorithm(ABC):
    @abstractmethod
    def __init__(
        self,
        game: Game,
        selected_frame_idx: int,
        objective_terms: list[ObjectiveTerm],
        weights: list[float],
        constraints: list[Constraint] | None = None,
        **kwargs: Any,
    ):
        raise NotImplementedError

    @abstractmethod
    def run(self) -> OptimizationResult:
        raise NotImplementedError


# def optimize_tracking_frame(
#     game: Game,
#     selected_frame_idx: int,
#     objective_terms: list[ObjectiveTerm],
#     weights: list[float],
#     constraints: list[Constraint],
#     algorithm: type[OptimizationAlgorithm],
# ) -> OptimizationResult:
#     LOGGER.info("Running optimization with %s", algorithm.__name__)

#     optimizer = algorithm(
#         game=game,
#         selected_frame_idx=selected_frame_idx,
#         objective_terms=objective_terms,
#         weights=weights,
#         constraints=constraints,
#     )
#     return optimizer.run()

