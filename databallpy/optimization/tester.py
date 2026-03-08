#imports

from databallpy.optimization.simulated_annealing import SimulatedAnnealing

from kloppy import skillcorner
from databallpy import get_game_from_kloppy
from databallpy.visualize import plot_soccer_pitch, plot_tracking_data
from databallpy.features.pitch_control import get_pitch_control_single_frame

import pandas as pd
# import accessible_space
import pickle

import matplotlib.pyplot as plt

match_id = 1886347
tracking_data_github_url = f"https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl"
meta_data_github_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_match.json"

dataset = skillcorner.load(
    meta_data=meta_data_github_url,
    raw_data=tracking_data_github_url,
    # Optional Parameters
    coordinates="skillcorner",
    limit=500, 
)


#actual script
game = get_game_from_kloppy(dataset)
game.tracking_data.add_velocity(game.get_column_ids() + ["ball"], allow_overwrite=True)
game.tracking_data.add_individual_player_possession()
selected_frame_idx = 210 

pc = get_pitch_control_single_frame(game.tracking_data[game.tracking_data['frame']==selected_frame_idx].iloc[0],
            game.pitch_dimensions,
            n_x_bins=106, n_y_bins=68,
        )
print(f"total pitch control: {sum(sum(pc))}")

annealer = SimulatedAnnealing(
            game, 
            selected_frame_idx, 
            distance_perturbation=0.2, # how many yards to randomly move the player on each iteration 
            max_tti= 2, # the maximum time to intercept, i.e. don't move a player if they can't reach that space within 1s
            num_iterations=1000, # the number of perturbations to do
            weighted_pitch_control_parameter = 1 
        )

result = annealer.run()
print(result)