from abc import ABC, abstractmethod
from databallpy.game import Game
from dataclasses import dataclass
import pandas as pd

class ObjectiveFunction(ABC):
    pass

@dataclass
class OptimizationResult:
    best_frame: pd.Series
    best_result: float

class OptimizationAlgorithm(ABC):
    @abstractmethod
    def __init__(self, game: Game, selected_frame_idx: int):
        pass
    @abstractmethod
    def run(self) -> OptimizationResult:
        pass

class Constraints:
    pass

class Filters:
    pass



def optimize_tracking_frame(
    game: Game,
    selected_frame_idx: int,
    objective: ObjectiveFunction,
    algorithm: OptimizationAlgorithm, # --> in the beginning only SimulatedAnnealing
    constraints: Constraints, # Within optimization algo
    filters: Filters # The variables the optimization algo is allowed to change
) -> OptimizationResult:
    optimizer = OptimizationAlgorithm(game, selected_frame_idx)
    return optimizer.run()