
from gym import core, spaces
from dm_soccer2gym.loader import teams_load
from dm_env import specs
from gym.utils import seeding
import gym
from dm_soccer2gym.viewer import DmControlViewer
import numpy as np
import sys

simple_act_dict = {0: np.array([1, 0, 0]),
                   1: np.array([-1, 0, 0]), 
                   2: np.array([0, 1, 0]),
                   3: np.array([0, -1, 0]), 
                   4: np.array([0, 0, 0])}

sigmoid = lambda x: 1 / (1 + np.exp(-x))

arctan_yx = lambda x, y: (np.arctan(np.divide(y, x) if x != 0 and y != 0 else 0) + np.pi * (x < 0)) % (2 * np.pi)

polar_mod = lambda x: np.sqrt(np.sum(np.square(x)))
polar_ang = lambda x: arctan_yx(x[0, 1], x[0, 0])

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
        time_limit = task_kwargs.get("time_limit", 45.)
        random_state = task_kwargs.get("random_state", None)
        disable_walker_contacts = task_kwargs.get("disable_walker_contacts", True)
        self.rew_type = task_kwargs.get("rew_type", "sparse")
        self.disable_jump = task_kwargs.get("disable_jump", False)
        self.discrete_actions = task_kwargs.get("discrete_actions", 0)
        self.simple_actions = task_kwargs.get("simple_actions", False)
        self.flags = np.array([False for i in range(self.num_players)])
            
        self.dmcenv = teams_load(home_team_size=team_1, away_team_size=team_2,
                                 time_limit=time_limit, random_state=random_state,
                                 disable_walker_contacts=disable_walker_contacts)

        # convert spec to space, discrete actions and disable jump if required
        self.action_space = convertSpec2Space(self.dmcenv.action_spec(), clip_inf=True)
        self.discrete = isinstance(self.action_space, spaces.Discrete)
        if not(self.simple_actions):
            if self.disable_jump:
                _shape = (self.action_space.shape[0] - 1,)
                _low = self.action_space.low[:-1]
                _high = self.action_space.high[:-1]
                self.action_space = spaces.Box(_low, _high)
            self.action_space, self.vals = dis_space(self.action_space, self.discrete_actions)
        else:
            self.action_space = DmcDiscrete(_minimum=0, _maximum=len(simple_act_dict.keys()))

        self.observation_space = convertOrderedDict2Space(self.dmcenv.observation_spec())

        if render_mode_list is not None:
            self.metadata['render.modes'] = list(render_mode_list.keys())
            self.viewer = {key: None for key in render_mode_list.keys()}
        else:
            self.metadata['render.modes'] = []

        self.render_mode_list = render_mode_list

        # set seed
        self.seed()

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
        
        if type(self.action_space) == DmcDiscrete:
            a_ = a.copy()
            a_ = [x + self.action_space.offset for x in a_]
            if not(self.discrete):
                if not (self.simple_actions):
                    num = self.dmcenv.action_spec()[0].shape[0] - int(self.disable_jump)
                    aux_all = [np.zeros(num) for _ in range(self.num_players)]
                    for j in range(self.num_players):
                        aux = aux_all[j]
                        div = self.discrete_actions ** (num - 1)
                        total = a_[j]
                        for i in range(num):
                            aux[i] = self.vals[i][int(total / div)]
                            total = total % div
                            div /= self.discrete_actions
                        if self.disable_jump:
                            aux = np.concatenate([aux, np.array([0], dtype=np.float32)])
                        aux_all[j] = aux
                    a_ = aux_all
                else:
                    a_ = [simple_act_dict[i] for i in a_]
            self.timestep = self.dmcenv.step(a_)
        
        else:
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

        if self.rew_type == "sparse":
            return [x * 100 for x in self.timestep.reward]

        elif self.rew_type == "neg_distance_ball_goal":
            obs = self.timestep.observation
            goal_pos = [o['opponent_goal_mid'][:, :2] for o in obs]
            ball_pos = [o['ball_ego_position'][:, :2] for o in obs]

            return [- (polar_mod(ball_pos[i]) + \
                       polar_mod(goal_pos[i])) * \
                    (1 - (self.timestep.reward[i] == 1.)) for i in range(self.num_players)]

        elif self.rew_type == "vel_ball_goal":
            obs = self.timestep.observation
            vel_goal = [o['stats_vel_ball_to_goal'][0] for o in obs]
            vel_ball = [o['stats_vel_to_ball'][0] for o in obs]

            return [(vel_ball[i] + vel_goal[i] + \
                     100 * self.timestep.reward[i]) for i in range(self.num_players)]

        elif self.rew_type == "openai_empty_goal":

            obs = self.timestep.observation
            goal_pos = [o['opponent_goal_mid'][:, :2] for o in obs]
            ball_pos = [o['ball_ego_position'][:, :2] for o in obs]

            ball_dist = np.array([polar_mod(ball_pos[i]) for i in range(self.num_players)])
            ball_goal_dist = np.array([polar_mod(goal_pos[i] - ball_pos[i]) \
                                       for i in range(self.num_players)])

            kickable = ball_dist < 0.5

            rewards = np.array(self.timestep.reward.copy()) * 5
            rewards += (self.old_ball_dist - ball_dist)
            rewards += 0.6 * (self.old_ball_goal_dist - ball_goal_dist)
            rewards += np.float32(kickable * (1 - self.got_kickable_rew))
            
            self.old_ball_dist = ball_dist
            self.old_ball_goal_dist = ball_goal_dist
            self.got_kickable_rew = kickable | self.got_kickable_rew      

            return rewards.tolist()

        else:
            raise ValueError("Invalid reward type")

    def render(self, mode='human_rgb_array', close=False):

        self.pixels = self.dmcenv.physics.render(**self.render_mode_list[mode]['render_kwargs'])
        if close:
            if self.viewer[mode] is not None:
                self._get_viewer(mode).close()
                self.viewer[mode] = None
            return
        elif self.render_mode_list[mode]['show']:
            self._get_viewer(mode).update(self.pixels)

        if self.render_mode_list[mode]['return_pixel']:
            return self.pixels

    def _get_viewer(self, mode):
        if self.viewer[mode] is None:
            self.viewer[mode] = DmControlViewer(self.pixels.shape[1], self.pixels.shape[0],
                                                self.render_mode_list[mode]['render_kwargs']['depth'])
        return self.viewer[mode]


class DmReachWrapper(DmSoccerWrapper):
    
    def __init__(self, team_1, team_2, task_kwargs={}, render_mode_list=None):

        super().__init__(team_1, team_2, task_kwargs, render_mode_list)
        self.dist_thresh = task_kwargs.get("dist_thresh", 0.03)
        self.observation_space = spaces.Box(0, 1, shape=(6,))

    def set_vals(self):
    
        obs = self.timestep.observation
        self.max_dist = polar_mod(self.dmcenv.task.arena.size)

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
            cut_obs.append({"ball_dist_scaled": ball_dist_scaled, "ball_angle_scaled": ball_angle_scaled,
                            "vel_norm_scaled": vel_norm_scaled, "vel_ang_scaled": vel_ang_scaled,
                            "ac_norm_scaled": ac_norm_scaled, "ac_ang_scaled": ac_ang_scaled})        

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
            print(kickable, self.got_kickable_rew)   
    
            return rewards.tolist()

        elif self.rew_type == "quick_reach":

            kickable = ball_dist < self.dist_thresh

            rewards = np.zeros(self.num_players)
            for j in range(self.num_players):
                if not kickable[j]:
                    rewards[j] = (self.old_ball_dist[j] - ball_dist[j]) - 1
                else:
                    rewards[j] = 50 * np.float32(kickable[j])
            
            self.old_ball_dist = ball_dist
            self.got_kickable_rew = kickable | self.got_kickable_rew      

            return rewards.tolist()

        else:
            raise ValueError("Invalid reward type")


class DmGoalWrapper(DmSoccerWrapper):

    def __init__(self, team_1, team_2, task_kwargs={}, render_mode_list=None):
        super().__init__(team_1, team_2, task_kwargs, render_mode_list)
        self.dist_thresh = task_kwargs.get("dist_thresh", 0.03)
        self.observation_space = spaces.Box(0, 1, shape=(13 + (self.num_players - 1) * 6,))

    def set_vals(self):
    
        obs = self.timestep.observation
        self.max_dist = polar_mod(self.dmcenv.task.arena.size)

        self.got_kickable_rew = np.array([False for o in obs])
        ball_pos = [o['ball_ego_position'][:, :2] for o in obs]
        ball_op_goal_pos = [ball_pos[i] - obs[i]["opponent_goal_mid"][:, :2] for i in range(self.num_players)]
        ball_team_goal_pos = [ball_pos[i] - obs[i]["team_goal_mid"][:, :2] for i in range(self.num_players)]
        self.old_ball_dist = np.array([polar_mod(ball_pos[i]) for i in range(self.num_players)]) / self.max_dist
        self.old_ball_op_goal_dist = np.array([polar_mod(ball_op_goal_pos[i]) for i in range(self.num_players)]) / self.max_dist
        self.old_ball_team_goal_dist = np.array([polar_mod(ball_team_goal_pos[i]) for i in range(self.num_players)]) / self.max_dist

    def getObservation(self):
        
        obs = self.timestep.observation
        cut_obs = []
        
        for o in obs:
            ball_pos = -o['ball_ego_position'][:, :2]
            ball_vel = o["ball_ego_linear_velocity"][:, :2]
            op_goal_pos = -o["opponent_goal_mid"][:, :2]
            team_goal_pos = -o["team_goal_mid"][:, :2]
            actual_vel = o["sensors_velocimeter"][:, :2]
            actual_ac = o["sensors_accelerometer"][:, :2]
            ball_op_goal_pos = -ball_pos + op_goal_pos
            ball_team_goal_pos = -ball_pos + team_goal_pos
            ball_goal_vel = o["stats_vel_ball_to_goal"]

            cut_obs.append({"ball_dist_scaled": np.array([(polar_mod(ball_pos) / self.max_dist)]), 
                            "ball_angle_scaled": np.array([polar_ang(ball_pos) / (2 * np.pi)]),
                            "ball_vel_scaled": np.array([(polar_mod(np.tanh(ball_pos)) / sqrt_2)]), 
                            "ball_vel_angle_scaled": np.array([polar_ang(ball_pos) / (2 * np.pi)]),
                            "op_goal_dist_scaled": np.array([(polar_mod(op_goal_pos) / self.max_dist)]),
                            "op_goal_angle_scaled": np.array([polar_ang(op_goal_pos) / (2 * np.pi)]),
                            "team_goal_dist_scaled": np.array([(polar_mod(team_goal_pos) / self.max_dist)]),
                            "team_goal_angle_scaled": np.array([polar_ang(team_goal_pos) / (2 * np.pi)]),
                            "vel_norm_scaled": np.array([polar_mod(np.tanh(actual_vel)) / sqrt_2]), 
                            "vel_ang_scaled": np.array([polar_ang(actual_vel) / (2 * np.pi)]),
                            "ac_norm_scaled": np.array([polar_mod(np.tanh(actual_ac)) / sqrt_2]), 
                            "ac_ang_scaled": np.array([polar_ang(actual_ac) / (2 * np.pi)]),
                            "ball_op_goal_dist_scaled": np.array([(polar_mod(ball_op_goal_pos) / self.max_dist)]),
                            "ball_op_goal_angle_scaled": np.array([polar_ang(ball_op_goal_pos) / (2 * np.pi)]),
                            "ball_team_goal_dist_scaled": np.array([(polar_mod(ball_team_goal_pos) / self.max_dist)]),
                            "ball_team_goal_angle_scaled": np.array([polar_ang(ball_team_goal_pos) / (2 * np.pi)]),
                            "ball_goal_vel": np.array([sigmoid(ball_goal_vel)])})

            for player in range(self.team_1 - 1):
                teammate_pos = -o[f"teammate_{player}_ego_position"][:, :2]
                teammate_vel = o[f"teammate_{player}_ego_linear_velocity"][:, :2]
                teammate_ball_pos = -teammate_pos + ball_pos

                cut_obs[-1][f"teammate_{player}_dist_scaled"] = np.array([(polar_mod(teammate_pos) / self.max_dist)])
                cut_obs[-1][f"teammate_{player}_angle_scaled"] = np.array([(polar_ang(teammate_pos) / (2 * np.pi))])
                cut_obs[-1][f"teammate_{player}_vel_scaled"] = np.array([(polar_mod(np.tanh(teammate_pos)) / sqrt_2)])
                cut_obs[-1][f"teammate_{player}_vel_angle_scaled"] = np.array([(polar_ang(teammate_pos) / (2 * np.pi))])
                cut_obs[-1][f"teammate_{player}_ball_dist_scaled"] = np.array([(polar_mod(teammate_ball_pos) / self.max_dist)])
                cut_obs[-1][f"teammate_{player}_ball_angle_scaled"] = np.array([(polar_ang(teammate_ball_pos) / (2 * np.pi))])

            for player in range(self.team_2):
                opponent_pos = -o[f"opponent_{player}_ego_position"][:, :2]
                opponent_vel = o[f"opponent_{player}_ego_linear_velocity"][:, :2]
                opponent_ball_pos = -opponent_pos + ball_pos

                cut_obs[-1][f"opponent_{player}_dist_scaled"] = np.array([(polar_mod(opponent_pos) / self.max_dist)])
                cut_obs[-1][f"opponent_{player}_angle_scaled"] = np.array([(polar_ang(opponent_pos) / (2 * np.pi))])
                cut_obs[-1][f"opponent_{player}_vel_scaled"] = np.array([(polar_mod(np.tanh(opponent_pos)) / sqrt_2)])
                cut_obs[-1][f"opponent_{player}_vel_angle_scaled"] = np.array([(polar_ang(opponent_pos) / (2 * np.pi))])
                cut_obs[-1][f"opponent_{player}_ball_dist_scaled"] = np.array([(polar_mod(opponent_ball_pos) / self.max_dist)])
                cut_obs[-1][f"opponent_{player}_ball_angle_scaled"] = np.array([(polar_ang(opponent_ball_pos) / (2 * np.pi))])

        return convertObservation(cut_obs)

    def calculate_rewards(self):

        if self.rew_type == "sparse":
            rewards = (100 * np.array(self.timestep.reward))  
    
            return rewards.tolist()

        elif self.rew_type == "simple":
            obs = self.timestep.observation
            ball_pos = [-o['ball_ego_position'][:, :2] for o in obs] 
            ball_op_goal_pos = [ball_pos[i] - obs[i]["opponent_goal_mid"][:, :2] for i in range(self.num_players)]
            ball_team_goal_pos = [ball_pos[i] - obs[i]["team_goal_mid"][:, :2] for i in range(self.num_players)]

            ball_dist = np.array([polar_mod(ball_pos[i]) for i in range(self.num_players)]) / self.max_dist
            ball_op_goal_dist = np.array([polar_mod(ball_op_goal_pos[i]) for i in range(self.num_players)]) / self.max_dist
            ball_team_goal_dist = np.array([polar_mod(ball_team_goal_pos[i]) for i in range(self.num_players)]) / self.max_dist
            kickable = ball_dist < self.dist_thresh

            rewards = (100 * np.array(self.timestep.reward))
            if not(np.any(rewards)):
                rewards += ((self.old_ball_dist - ball_dist) - 1.5)
                rewards += 10 * (kickable * (1 - self.got_kickable_rew))
                if np.any(kickable | self.got_kickable_rew):
                    rewards += 1.2 * (self.old_ball_op_goal_dist - ball_op_goal_dist)
                    rewards -= 1.2 * (self.old_ball_team_goal_dist - ball_team_goal_dist) 
            
            self.old_ball_dist = ball_dist
            self.old_ball_op_goal_dist = ball_op_goal_dist
            self.old_ball_team_goal_dist = ball_team_goal_dist
            self.got_kickable_rew = kickable | self.got_kickable_rew      

            return rewards.tolist()

        else:
            raise ValueError("Invalid reward type")
