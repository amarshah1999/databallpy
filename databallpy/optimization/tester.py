from databallpy.optimization.simulated_annealing import SimulatedAnnealing
from kloppy import skillcorner
from databallpy import get_game_from_kloppy
import matplotlib.patches as mpatches

from databallpy.visualize import (
    _plot_heatmap_overlay,
    _plot_single_frame,
    _plot_velocities,
    pick_bw_for_contrast,
    plot_soccer_pitch,
    plot_tracking_data,
)
from matplotlib.colors import to_rgb
from databallpy.features.pitch_control import (
    get_pitch_control_single_frame,
    get_team_influence,
)
from databallpy.optimization.optimization import (
    WeightedPitchControlObjective,
    PressureObjective,
    TTIConstraint,
    optimize_tracking_frame,
)
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from copy import deepcopy


def generate_fig(game, frame_idx, save_path=None):
    frame = game.tracking_data[game.tracking_data["frame"] == frame_idx].iloc[0]
    pitch_control = get_pitch_control_single_frame(
        frame,
        game.pitch_dimensions,
        n_x_bins=106,
        n_y_bins=68,
    )
    cmap_red_green = LinearSegmentedColormap.from_list(
        "reds", [(0, 1, 0, 1), (0.5, 0.5, 0, 0), (1, 0, 0, 1)]
    )

    fig, ax = plot_soccer_pitch(field_dimen=game.pitch_dimensions, pitch_color="white")
    idx_relative_to_game = game.tracking_data[
        game.tracking_data["frame"] == frame_idx
    ].index[0]
    fig, ax = plot_tracking_data(
        game,
        idx_relative_to_game,
        team_colors=["green", "red"],
        ax=ax,
        fig=fig,
        heatmap_overlay=pitch_control,
        overlay_cmap=cmap_red_green,
        add_velocities=True,
    )
    if save_path:
        fig.savefig(save_path)
    return fig, ax

def diff_frames(game_1, frame_idx_1, game_2, frame_idx_2, save_path=None):
    # frame_1 = game_1.tracking_data[game_1.tracking_data["frame"] == frame_idx_1].iloc[0]
    # frame_2 = game_2.tracking_data[game_2.tracking_data["frame"] == frame_idx_2].iloc[0]
    # pitch_control_1 = get_pitch_control_single_frame(
    #     frame_1,
    #     game_1.pitch_dimensions,
    #     n_x_bins=106,
    #     n_y_bins=68,
    # )
    # pitch_control_2 = get_pitch_control_single_frame(
    #     frame_2,
    #     game_2.pitch_dimensions,
    #     n_x_bins=106,
    #     n_y_bins=68,
    # )
    # cmap_red_green = LinearSegmentedColormap.from_list(
    #     "reds", [(0, 1, 0, 1), (0.5, 0.5, 0, 0), (1, 0, 0, 1)]
    # )

    fig, ax = plot_soccer_pitch(
        field_dimen=game_1.pitch_dimensions, pitch_color="white"
    )
    idx_relative_to_game_1 = game_1.tracking_data[
        game_1.tracking_data["frame"] == frame_idx_1
    ].index[0]
    idx_relative_to_game_2 = game_2.tracking_data[
        game_2.tracking_data["frame"] == frame_idx_2
    ].index[0]

    before_alpha = 0.2

    fig, ax = plot_tracking_data(
        game_1,
        idx_relative_to_game_1,
        team_colors=["green", "red"],
        ax=ax,
        fig=fig,
        alpha=before_alpha,
        # heatmap_overlay=pitch_control_1,
        # overlay_cmap=cmap_red_green,
        # add_velocities=False,
    )
    fig, ax = plot_tracking_data(
        game_2,
        idx_relative_to_game_2,
        team_colors=["green", "red"],
        ax=ax,
        fig=fig,
        # heatmap_overlay=pitch_control_2,
        # overlay_cmap=cmap_red_green,
        # add_velocities=False,
    )

    after_legend_markers = (
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=(0.0, 1.0, 0.0, 1.0),
            markeredgecolor="green",
            markersize=9,
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=(1.0, 0.0, 0.0, 1.0),
            markeredgecolor="red",
            markersize=9,
            linestyle="None",
        ),
    )
    before_legend_markers = (
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=(0.0, 1.0, 0.0, before_alpha),
            markeredgecolor=(0.0, 0.5, 0.0, 0.9),
            markersize=9,
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=(1.0, 0.0, 0.0, before_alpha),
            markeredgecolor=(0.7, 0.0, 0.0, 0.9),
            markersize=9,
            linestyle="None",
        ),
    )
    ax.legend(
        [before_legend_markers, after_legend_markers],
        ["Before Optimization", "After Optimization"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="lower right",
        framealpha=0.95,
    ).set_zorder(20)

    if save_path:
        fig.savefig(save_path)
    return fig, ax

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


# actual script
game = get_game_from_kloppy(dataset)
game.tracking_data.add_velocity(game.get_column_ids() + ["ball"], allow_overwrite=True)
game.tracking_data.add_individual_player_possession()
selected_frame_idx = 210

pc = get_pitch_control_single_frame(
    game.tracking_data[game.tracking_data["frame"] == selected_frame_idx].iloc[0],
    game.pitch_dimensions,
    n_x_bins=106,
    n_y_bins=68,
)
print(f"total pitch control: {sum(sum(pc))}")

grid = np.meshgrid(
    np.linspace(
        -game.pitch_dimensions[0] / 2,
        game.pitch_dimensions[0] / 2,
        106,
    ),
    np.linspace(-game.pitch_dimensions[1] / 2, game.pitch_dimensions[1] / 2, 68),
)
frame = game.tracking_data[game.tracking_data["frame"] == selected_frame_idx].iloc[0]
attacking_team = frame["team_possession"]
defending_team = "home" if frame["team_possession"] == "away" else "away"
attacking_team_influence = get_team_influence(
    frame,
    col_ids=game.get_column_ids(team=attacking_team),
    grid=grid,
    player_ball_distances=None,
)

result = optimize_tracking_frame(
    game, 
    selected_frame_idx, 
    objective_terms = [
        WeightedPitchControlObjective(
            grid=grid,
            attacking_team=attacking_team,
            defending_team=defending_team,
            attacking_team_influence=attacking_team_influence,
            defending_player_ids=game.get_column_ids(team=defending_team),
        ),
        PressureObjective(
            attacking_player_ids=game.get_column_ids(team=defending_team), game=game
        ),
    ], 
    weights = [1,1], 
    constraints = [TTIConstraint()],
    algorithm = SimulatedAnnealing,
)
print(result.best_result)
new_game = deepcopy(game)
new_game.tracking_data = pd.DataFrame(result.best_frame).transpose().reset_index()
# generate_fig(new_game, selected_frame_idx, save_path="test.png")
diff_frames(game, selected_frame_idx, new_game, selected_frame_idx, save_path="test_diff.png")
