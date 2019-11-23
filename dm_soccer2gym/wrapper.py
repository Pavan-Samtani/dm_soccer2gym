from gym import core, spaces
from dm_soccer2gym.loader import single_team_load, load
from dm_env import specs
from gym.utils import seeding
import gym
from dm_soccer2gym.viewer import DmControlViewer
import numpy as np
import sys


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
        elif not isinstance(odict, list):
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


class DmSoccerWrapper(core.Env):

    def __init__(self, team_1, team_2, task_kwargs={}, render_mode_list=None):
        
        
        self.team_1 = team_1
        self.team_2 = team_2
        self.num_players = team_1 + team_2
        time_limit = task_kwargs.get("time_limit", 45.)
        random_state = task_kwargs.get("random_state", None)
        disable_walker_contacts = task_kwargs.get("disable_walker_contacts", True)
            
        if team_2 == 0:
            self.dmcenv = single_team_load(team_size=team_1, time_limit=time_limit, 
                                           random_state=random_state,
                                           disable_walker_contacts=disable_walker_contacts)
        else:
            self.dmcenv = load(team_size=team_1, time_limit=time_limit, 
                               random_state=random_state,
                               disable_walker_contacts=disable_walker_contacts)

        # convert spec to space
        self.action_space = convertSpec2Space(self.dmcenv.action_spec(), clip_inf=True)
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
        return self.getObservation()

    def step(self, a):

        if type(self.action_space) == DmcDiscrete:
            a += self.action_space.offset
        self.timestep = self.dmcenv.step(a)

        return self.getObservation(), self.timestep.reward, self.timestep.last(), {}

    def render(self, mode='human', close=False):

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
