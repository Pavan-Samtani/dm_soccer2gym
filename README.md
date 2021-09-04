# dm_control2gym

dm_control2gym is a small wrapper to make [DeepMind Control Suite](https://github.com/deepmind/dm_control) environments available for [OpenAI Gym](https://github.com/openai/gym). https://github.com/martinseilair/dm_control2gym/ is the original repo, this fork has a focus of wrapping dm_control soccer environments in gym environment.

## Installation

```shell
$ git clone https://github.com/pavan-samtani/dm_soccer2gym/
$ cd dm_soccer2gym
$ pip install .
```

Tested with
- Python 3.5.2 and Ubuntu 16.04.
- Python 3.6.8 and Ubuntu 18.04


```python
import dm_soccer2gym

# make the 1vs0 dm_control soccer environment with a dense reward
env = dm_soccer2gym.make('1vs0', task_kwargs={"rew_type": "dense", "time_limit": 45.,
"disable_jump": True, "dist_thresh": 0.03,  'control_timestep': 0.05})

# use same syntax as in gym
env.reset()
for t in range(10):
    observation, reward, done, info = env.step([env.action_space.sample() for _ in range(env.num_players)]) # take a random action
    env.render()

```

To obtain the dm_control's soccer environment original observation dictionary:

```python
import dm_soccer2gym

# make the 1vs0 dm_control soccer environment with a dense reward
env = dm_soccer2gym.make('1vs0', task_kwargs={"rew_type": "dense", "time_limit": 45.,
"disable_jump": True, "dist_thresh": 0.03,  'control_timestep': 0.05})

# use same syntax as in gym
env.reset()
for t in range(10):
    observation, reward, done, info = env.step([env.action_space.sample() for _ in range(env.num_players)]) # take a random action
    env.render()

    # list of dicstionary with env's original observation
    obs_dict = env.timestep.observation
```

## Short documentation (from martinseilair/dm_control2gym)

### Spaces and Specs

The dm_control specs are converted to spaces. If there is only one entity in the observation dict, the original shape is used for the corresponding space. Otherwise, the observations are converted to a vector and concatenated.

### Rendering
Three rendering modes are available by default:

* `human`: Render scene and show it
* `rgb_array`: Render scene and return it as rgb array
* `human_rgb_array`: Render scene, show and return it


__Example__

```python
env = dm_soccer2gym.make('2vs2', task_kwargs={"rew_type": "dense",
"time_limit": 45., "disable_jump": True, "dist_thresh": 0.03,  
'control_timestep': 0.05, 'observables': 'all'})
```

## Known Error
As of 25/9/2019, with the current dm_soccer package, you might be getting the error below.

```shell
Traceback (most recent call last):
  File "/dm_control2gym/tests/sample.py", line 12, in <module>
    env.render()
  File "/dm_control2gym/dm_control2gym/wrapper.py", line 116, in render
    self._get_viewer(mode).update(self.pixels)
  File "/dm_control2gym/dm_control2gym/viewer.py", line 20, in update
    self.window.clear()
  File "/python3.6/site-packages/pyglet/window/__init__.py", line 1228, in clear
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
  File "/python3.6/site-packages/pyglet/gl/lib.py", line 105, in errcheck
    raise GLException(msg)
pyglet.gl.lib.GLException: b'invalid operation'
```
### To solve

Run:
```shell
export MUJOCO_GL = "osmesa"
```
