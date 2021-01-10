
from collections import OrderedDict
from gym import core, spaces
from dm_control.locomotion.soccer import camera
from dm_soccer2gym.loader import teams_load
from dm_env import specs
from gym.utils import seeding
import gym
import numpy as np
import sys


sigmoid = lambda x: 1 / (1 + np.exp(-x))

arctan_yx = lambda x, y: (np.arctan(np.divide(y, x) if x != 0 and y != 0 else 0) + np.pi * (x < 0)) % (2 * np.pi)

polar_mod = lambda x: np.sqrt(np.sum(np.square(x)))
polar_ang = lambda x: arctan_yx(x[0, 0], x[0, 1])
cos_vector = lambda x, y: np.dot(np.ravel(x), np.ravel(y)) / (np.linalg.norm(x) * np.linalg.norm(y))

sqrt_2 = np.sqrt(2)

class DmcDiscrete(gym.spaces.Discrete):
    def __init__(self, _minimum, _maximum):
        super().__init__(_maximum - _minimum)
        self.offset = _minimum


def convertSpec2Space(spec, clip_inf=False):
    if not isinstance(spec, list):
        if spec.dtype == np.int:
            # Discrete
            return DmcDiscrete(spec.minimum, spec.maximum)
        # Box
        else:
            if type(spec) is specs.Array:
                return spaces.Box(-np.inf, np.inf, shape=spec.shape)
            elif type(spec) is specs.BoundedArray:
                _min = spec.minimum
                _max = spec.maximum
                if clip_inf:
                    _min = np.clip(spec.minimum, -sys.float_info.max, sys.float_info.max)
                    _max = np.clip(spec.maximum, -sys.float_info.max, sys.float_info.max)

                if np.isscalar(_min) and np.isscalar(_max):
                    # same min and max for every element
                    return spaces.Box(_min, _max, shape=spec.shape)
                else:
                    # different min and max for every element
                    return spaces.Box(_min + np.zeros(spec.shape),
                                      _max + np.zeros(spec.shape))
            else:
                raise ValueError('Unknown spec!')
    elif isinstance(spec, list):
        return convertSpec2Space(spec[0])
    else:
        raise ValueError('Unknown spec!')


def convertOrderedDict2Space(odict):
    if not isinstance(odict, list):
        if len(odict.keys()) == 1:
            # no concatenation
            return convertSpec2Space(list(odict.values())[0])
        else:
            # concatentation
            numdim = sum([np.int(np.prod(odict[key].shape)) for key in odict])
            return spaces.Box(-np.inf, np.inf, shape=(numdim,))
    else:
        return convertOrderedDict2Space(odict[0])


def convertObservation(spec_obs):
    if not isinstance(spec_obs, list):
        if len(spec_obs.keys()) == 1:
            # no concatenation
            return list(spec_obs.values())[0]
        else:
            # concatentation
            numdim = sum([np.int(np.prod(spec_obs[key].shape)) for key in spec_obs])
            space_obs = np.zeros((numdim,))
            i = 0
            for key in spec_obs:
                space_obs[i:i+np.prod(spec_obs[key].shape)] = spec_obs[key].ravel()
                i += np.prod(spec_obs[key].shape)
            return space_obs
    else:
        return [convertObservation(x) for x in spec_obs]


def dis_space(action_space, flag):
    if not flag or isinstance(action_space, spaces.Discrete):
        return action_space, None
    else:
        a_s = action_space
        num = a_s.shape[0]
        return (DmcDiscrete(_minimum=0, _maximum=int(flag ** num)),
               [np.linspace(a_s.low[i], a_s.high[i], num=flag, endpoint=True) for i in range(num)])


class DmSoccerWrapper(core.Env):

    def __init__(self, team_1, team_2, task_kwargs={}, render_mode_list=None):

        self.team_1 = team_1
        self.team_2 = team_2
        self.num_players = team_1 + team_2

        random_state = task_kwargs.get("random_state", None)
        disable_walker_contacts = task_kwargs.get("disable_walker_contacts", True)

        self.time_limit = task_kwargs.get("time_limit", 45.)        
        self.control_timestep = task_kwargs.get("control_timestep", 0.025)
        self.rew_type = task_kwargs.get("rew_type", "sparse")
        self.disable_jump = task_kwargs.get("disable_jump", False)
        self.observables = task_kwargs.get("observables", "core")
        self.flags = np.array([False for i in range(self.num_players)])

        if render_mode_list is not None:
            tracking_cameras = []
            for min_distance in render_mode_list:
                tracking_cameras.append(
                    camera.MultiplayerTrackingCamera(
                        min_distance=min_distance,
                        distance_factor=1,
                        smoothing_update_speed=0.1,
                        width=720,
                        height=360,
                    ))
        
        else:
            tracking_cameras = ()

        self.tracking_cameras = tracking_cameras

        self.render_mode_list = render_mode_list
            
        self.dmcenv = teams_load(home_team_size=team_1, away_team_size=team_2,
                                 time_limit=self.time_limit, random_state=random_state,
                                 disable_walker_contacts=disable_walker_contacts,
                                 control_timestep=self.control_timestep,
                                 observables=self.observables,
                                 tracking_cameras=self.tracking_cameras)

        # convert spec to space, discrete actions and disable jump if required
        self.action_space = convertSpec2Space(self.dmcenv.action_spec(), clip_inf=True)
        
        if self.disable_jump:
            _shape = (self.action_space.shape[0] - 1,)
            _low = self.action_space.low[:-1]
            _high = self.action_space.high[:-1]
            self.action_space = spaces.Box(_low, _high)

        self.observation_space = convertOrderedDict2Space(self.dmcenv.observation_spec())

        # set seed
        # self.seed()

    def getObservation(self):
        return convertObservation(self.timestep.observation)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.timestep = self.dmcenv.reset()
        self.set_vals()
        return self.getObservation()

    def get_end(self):
        return self.timestep.last()

    def step(self, a):
        
        a_ = a.copy()
        if self.disable_jump:
            for j in range(len(a_)):
                a_[j] = np.concatenate([a_[j], np.array([0], dtype=np.float32)])
                
        self.timestep = self.dmcenv.step(a_)

        return self.getObservation(), self.calculate_rewards(), self.get_end(), {}

    def set_vals(self):
    
        obs = self.timestep.observation

        self.got_kickable_rew = np.array([False for o in obs])
        ball_pos = [o['ball_ego_position'][:, :2] for o in obs]
        goal_pos = [o['opponent_goal_mid'][:, :2] for o in obs]
        self.old_ball_dist = np.array([polar_mod(ball_pos[i]) for i in range(self.num_players)])
        self.old_ball_goal_dist = np.array([polar_mod(goal_pos[i] - ball_pos[i]) \
                                            for i in range(self.num_players)])

    def calculate_rewards(self):

        return self.timestep.reward

    def render(self):

        return [cam.render() for cam in self.tracking_cameras]


"""
class DmReachWrapper(DmSoccerWrapper):
    
    def __init__(self, team_1, team_2, task_kwargs={}, render_mode_list=None):

        super().__init__(team_1, team_2, task_kwargs, render_mode_list)
        self.dist_thresh = task_kwargs.get("dist_thresh", 0.03)
        self.observation_space = spaces.Box(0, 1, shape=(6,))

    def set_vals(self):
    
        obs = self.timestep.observation
        fl = self.timestep.observation[0]["field_front_left"][:, :2]
        br = self.timestep.observation[0]["field_back_right"][:, :2]

        self.max_dist = polar_mod(fl - br)

        self.got_kickable_rew = np.array([False for o in obs])
        ball_pos = [o['ball_ego_position'][:, :2] for o in obs]
        self.old_ball_dist = np.array([polar_mod(ball_pos[i]) for i in range(self.num_players)]) / self.max_dist

    def get_end(self):

        return np.any(self.got_kickable_rew) | self.timestep.last()

    def getObservation(self):
        
        obs = self.timestep.observation
        cut_obs = []
        
        for o in obs:
            ball_pos = -o['ball_ego_position'][:, :2]
            actual_vel = o["sensors_velocimeter"][:, :2]
            actual_ac = o["sensors_accelerometer"][:, :2]
            ball_dist_scaled = np.array([(polar_mod(ball_pos) / self.max_dist)])
            ball_angle_scaled = np.array([polar_ang(ball_pos) / (2 * np.pi)])
            vel_norm_scaled = np.array([polar_mod(np.tanh(actual_vel)) / sqrt_2])
            vel_ang_scaled = np.array([polar_ang(actual_vel) / (2 * np.pi)])
            ac_norm_scaled = np.array([polar_mod(np.tanh(actual_ac)) / sqrt_2])
            ac_ang_scaled = np.array([polar_ang(actual_ac) / (2 * np.pi)])
            cut_obs.append(OrderedDict({"ball_dist_scaled": ball_dist_scaled, "ball_angle_scaled": ball_angle_scaled,
                            		   "vel_norm_scaled": vel_norm_scaled, "vel_ang_scaled": vel_ang_scaled,
                            		   "ac_norm_scaled": ac_norm_scaled, "ac_ang_scaled": ac_ang_scaled}))       

        return convertObservation(cut_obs)

    def calculate_rewards(self):

        obs = self.timestep.observation
        ball_pos = [o['ball_ego_position'][:, :2] for o in obs]

        ball_dist = np.array([polar_mod(ball_pos[i]) for i in range(self.num_players)]) / self.max_dist

        if self.rew_type == "sparse":

            kickable = ball_dist < self.dist_thresh
            rewards = 50 * np.float32(kickable * (1 - self.got_kickable_rew))

            self.old_ball_dist = ball_dist
            self.got_kickable_rew = kickable | self.got_kickable_rew    
    
            return rewards.tolist()

        elif self.rew_type == "quick_reach":

            kickable = ball_dist < self.dist_thresh

            rewards = np.zeros(self.num_players)
            for j in range(self.num_players):
                if not kickable[j]:
                    rewards[j] = (self.old_ball_dist[j] - ball_dist[j]) - 1
                else:
                    rewards[j] = 50
            
            self.old_ball_dist = ball_dist
            self.got_kickable_rew = kickable | self.got_kickable_rew      

            return rewards.tolist()

        else:
            raise ValueError("Invalid reward type")
"""


class DmGoalWrapper(DmSoccerWrapper):

    def __init__(self, team_1, team_2, task_kwargs={}, render_mode_list=None):
        super().__init__(team_1, team_2, task_kwargs, render_mode_list)
        self.dist_thresh = task_kwargs.get("dist_thresh", 0.03)
        self.observation_space = spaces.Box(0, 1, shape=(18 + (self.num_players - 1) * 6,))
        # self.observation_space = spaces.Box(0, 1, shape=(18 + (self.num_players - 1) * 4,))

    def set_vals(self):
    
        obs = self.timestep.observation
        fl = self.timestep.observation[0]["field_front_left"][:, :2]
        br = self.timestep.observation[0]["field_back_right"][:, :2]

        self.max_dist = polar_mod(fl - br)

        self.got_kickable_rew = np.array([False for _ in range(self.num_players)])
        self.old_ball_dist = []
        self.old_ball_op_dist = []
        self.old_ball_team_dist = []

        for i in range(self.num_players):
            ball_pos = -obs[i]['ball_ego_position'][:, :2]
            op_goal_pos = -obs[i]["opponent_goal_mid"][:, :2]
            tm_goal_pos = -obs[i]["team_goal_mid"][:, :2]

            ball_op_goal_pos = -ball_pos + op_goal_pos
            ball_team_goal_pos = -ball_pos + tm_goal_pos
            self.old_ball_dist.append(polar_mod(ball_pos) / self.max_dist)
            self.old_ball_op_dist.append(polar_mod(ball_op_goal_pos) / self.max_dist)
            self.old_ball_team_dist.append(polar_mod(ball_team_goal_pos) / self.max_dist)

        self.old_ball_dist = np.array(self.old_ball_dist)
        self.old_ball_op_goal_dist = np.array(self.old_ball_op_dist)
        self.old_ball_team_goal_dist = np.array(self.old_ball_team_dist)

    def getObservation(self):
        
        obs = self.timestep.observation
        cut_obs = []
        ball_pos_all = [-o['ball_ego_position'][:, :2] for o in obs]
        ball_dist_scaled_all = np.array([polar_mod(ball_pos) for ball_pos in ball_pos_all]) / self.max_dist
        kickable = ball_dist_scaled_all < self.dist_thresh
        kickable_ever = self.got_kickable_rew

        ctr = 0
        for o in obs:
            ball_pos = ball_pos_all[ctr]
            ball_vel = o["ball_ego_linear_velocity"][:, :2]
            op_goal_pos = -o["opponent_goal_mid"][:, :2]
            team_goal_pos = -o["team_goal_mid"][:, :2]

            actual_vel = o["sensors_velocimeter"][:, :2]
            actual_ac = o["sensors_accelerometer"][:, :2]
            ball_op_goal_pos = -ball_pos + op_goal_pos
            ball_team_goal_pos = -ball_pos + team_goal_pos
            ball_goal_vel = o["stats_vel_ball_to_goal"]
            ball_dist_scaled = np.array([ball_dist_scaled_all[ctr]])

            cut_obs.append(OrderedDict({"ball_dist_scaled": ball_dist_scaled, 
                            		   "ball_angle_scaled": np.array([polar_ang(ball_pos) / (2 * np.pi)]),
                                       "vel_norm_scaled": np.array([polar_mod(np.tanh(actual_vel)) / sqrt_2]), 
                                       "vel_ang_scaled": np.array([polar_ang(actual_vel) / (2 * np.pi)]),
                            		   "ac_norm_scaled": np.array([polar_mod(np.tanh(actual_ac)) / sqrt_2]), 
                                       "ac_ang_scaled": np.array([polar_ang(actual_ac) / (2 * np.pi)]),
                                       "op_goal_dist_scaled": np.array([(polar_mod(op_goal_pos) / self.max_dist)]),
                                       "op_goal_angle_scaled": np.array([polar_ang(op_goal_pos) / (2 * np.pi)]),
                            		   "team_goal_dist_scaled": np.array([(polar_mod(team_goal_pos) / self.max_dist)]),
                            		   "team_goal_angle_scaled": np.array([polar_ang(team_goal_pos) / (2 * np.pi)]),
                            		   "ball_vel_scaled": np.array([(polar_mod(np.tanh(ball_vel)) / sqrt_2)]), 
                                       "ball_vel_angle_scaled": np.array([polar_ang(ball_vel) / (2 * np.pi)]),
                            		   "ball_op_goal_dist_scaled": np.array([(polar_mod(ball_op_goal_pos) / self.max_dist)]),
                            		   "ball_op_goal_angle_scaled": np.array([polar_ang(ball_op_goal_pos) / (2 * np.pi)]),
                            		   "ball_team_goal_dist_scaled": np.array([(polar_mod(ball_team_goal_pos) / self.max_dist)]),
                            		   "ball_team_goal_angle_scaled": np.array([polar_ang(ball_team_goal_pos) / (2 * np.pi)]),
                            		   "ball_goal_vel": np.array([sigmoid(ball_goal_vel)]),
                            		   "kickable_ever": np.float32(np.array([kickable_ever[ctr]]))}))

            for player in range(self.team_2):
                opponent_pos = -o[f"opponent_{player}_ego_position"][:, :2]
                opponent_vel = o[f"opponent_{player}_ego_linear_velocity"][:, :2]
                opponent_ball_pos = -opponent_pos + ball_pos

                cut_obs[-1][f"opponent_{player}_dist_scaled"] = np.array([(polar_mod(opponent_pos) / self.max_dist)])
                cut_obs[-1][f"opponent_{player}_angle_scaled"] = np.array([(polar_ang(opponent_pos) / (2 * np.pi))])
                cut_obs[-1][f"opponent_{player}_vel_scaled"] = np.array([(polar_mod(np.tanh(opponent_vel)) / sqrt_2)])
                cut_obs[-1][f"opponent_{player}_vel_angle_scaled"] = np.array([(polar_ang(opponent_vel) / (2 * np.pi))])
                cut_obs[-1][f"opponent_{player}_ball_dist_scaled"] = np.array([(polar_mod(opponent_ball_pos) / self.max_dist)])
                cut_obs[-1][f"opponent_{player}_ball_angle_scaled"] = np.array([(polar_ang(opponent_ball_pos) / (2 * np.pi))])

            for player in range(self.team_1 - 1):
                teammate_pos = -o[f"teammate_{player}_ego_position"][:, :2]
                teammate_vel = o[f"teammate_{player}_ego_linear_velocity"][:, :2]
                teammate_ball_pos = -teammate_pos + ball_pos

                cut_obs[-1][f"teammate_{player}_dist_scaled"] = np.array([(polar_mod(teammate_pos) / self.max_dist)])
                cut_obs[-1][f"teammate_{player}_angle_scaled"] = np.array([(polar_ang(teammate_pos) / (2 * np.pi))])
                cut_obs[-1][f"teammate_{player}_vel_scaled"] = np.array([(polar_mod(np.tanh(teammate_vel)) / sqrt_2)])
                cut_obs[-1][f"teammate_{player}_vel_angle_scaled"] = np.array([(polar_ang(teammate_vel) / (2 * np.pi))])
                cut_obs[-1][f"teammate_{player}_ball_dist_scaled"] = np.array([(polar_mod(teammate_ball_pos) / self.max_dist)])
                cut_obs[-1][f"teammate_{player}_ball_angle_scaled"] = np.array([(polar_ang(teammate_ball_pos) / (2 * np.pi))])

            ctr += 1

        return np.clip(convertObservation(cut_obs), -1, 1)

    def calculate_rewards(self):

        if self.rew_type == "sparse":
            rewards = self.timestep.reward
    
            return rewards
        
        """
        elif self.rew_type == "simple":
            obs = self.timestep.observation
            ball_pos = [-o['ball_ego_position'][:, :2] for o in obs] 

            ball_dist = np.array([polar_mod(ball_pos[i]) for i in range(self.num_players)]) / self.max_dist
            kickable = ball_dist < self.dist_thresh

            rewards = (100 * np.array(self.timestep.reward))
            if not(np.any(rewards)):
                rewards -= 1.
                kickable_now_first = (kickable * (1 - self.got_kickable_rew))
                rewards += (self.old_ball_dist - ball_dist) * (1 - kickable_now_first) + 10 * kickable_now_first
            
            self.old_ball_dist = ball_dist
            self.got_kickable_rew = kickable | self.got_kickable_rew      

            return rewards.tolist()
        """

        if self.rew_type == "simple_v2":
            obs = self.timestep.observation
            ball_pos = [-o['ball_ego_position'][:, :2] for o in obs] 
            ball_op_goal_pos = [-ball_pos[i] - obs[i]["opponent_goal_mid"][:, :2] for i in range(self.num_players)]
            ball_team_goal_pos = [-ball_pos[i] - obs[i]["team_goal_mid"][:, :2] for i in range(self.num_players)]

            ball_dist = np.array([polar_mod(ball_pos[i]) for i in range(self.num_players)]) / self.max_dist
            ball_op_goal_dist = np.array([polar_mod(ball_op_goal_pos[i]) for i in range(self.num_players)]) / self.max_dist
            ball_team_goal_dist = np.array([polar_mod(ball_team_goal_pos[i]) for i in range(self.num_players)]) / self.max_dist
            kickable = ball_dist < self.dist_thresh

            val_1 = (int(self.time_limit / self.control_timestep) + 1) / 10
            val_2 = val_1 / 10

            rewards = (val_1 * np.array(self.timestep.reward))
            if not(np.any(rewards)):
                rewards -= 0.1
                kickable_now_first = (kickable * (1 - self.got_kickable_rew))
                rewards += 1.2 * ((self.old_ball_op_goal_dist - ball_op_goal_dist) - (self.old_ball_team_goal_dist - ball_team_goal_dist)) \
                           * self.got_kickable_rew + ((self.old_ball_dist - ball_dist) * (1 - kickable_now_first) + val_2 * kickable_now_first) \
                           * (1 - self.got_kickable_rew)
            
            self.old_ball_dist = ball_dist
            self.old_ball_op_goal_dist = ball_op_goal_dist
            self.old_ball_team_goal_dist = ball_team_goal_dist
            self.got_kickable_rew = kickable | self.got_kickable_rew      

            return rewards.tolist()

        else:
            raise ValueError("Invalid reward type")

