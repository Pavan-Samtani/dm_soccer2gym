
import numpy as np


def calculate_rewards(timestep, rew_type):
    if rew_type == "sparse":
        return timestep.reward
    elif rew_type == "neg_distance":
        obs = timestep.observation
        norm_dist_goal = [o['opponent_goal_mid'][:, :2] / np.array([[32., 24.]]) for o in obs]
        norm_dist_ball = [o['ball_ego_position'][:, :2] / np.array([[32., 24.]]) for o in obs]
        return [(- 2 * np.sum(np.square(norm_dist_goal[i] - norm_dist_ball[i])) - \
                np.sum(np.square(norm_dist_ball[i]))) * (1 - (timestep.reward[i] == 1.)) \
                for i in range(len(obs))]
    else:
        raise ValueError("Inavlid reward type")
