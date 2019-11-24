
import numpy as np


def calculate_rewards(timestep, rew_type):
    if rew_type == "sparse":
        return timestep.reward
    elif rew_type == "neg_distance":
        obs = timestep.observation
        return [- 2 * np.sqrt(np.sum(np.square(o['opponent_goal_mid'] - \
                                               o['ball_ego_position']))) - \
                np.sqrt(np.sum(np.square(o['ball_ego_position']))) for o in obs]
    else:
        raise ValueError("Inavlid reward type")
