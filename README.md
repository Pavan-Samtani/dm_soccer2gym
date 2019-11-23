# dm_control2gym

dm_control2gym is a small wrapper to make [DeepMind Control Suite](https://github.com/deepmind/dm_control) environments available for [OpenAI Gym](https://github.com/openai/gym).
https://github.com/martinseilair/dm_control2gym/ is the original repo, this fork has a focus of wrapping dm_control soccer environments in gym environment.

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
import gym
import dm_soccer2gym

# make the 1vs0 dm_control soccerenvironment
env = dm_soccer2gym.make(domain_name="dm_soccer", task_name="1vs0")

# use same syntax as in gym
env.reset()
for t in range(10):
    observation, reward, done, info = env.step([env.action_space.sample() for _ in range(env.num_players)]) # take a random action
    env.render()

```

## Short documentation

### Spaces and Specs

The dm_control specs are converted to spaces. If there is only one entity in the observation dict, the original shape is used for the corresponding space. Otherwise, the observations are vectorized and concatenated.

### Rendering
Three rendering modes are available by default:

* `human`: Render scene and show it
* `rgb_array`: Render scene and return it as rgb array
* `human_rgb_array`: Render scene, show and return it

You can create your own rendering modes before making the environment by:

```python
dm_soccer2gym.create_render_mode(name, show=True, return_pixel=False, height=240, width=320, camera_id=-1, overlays=(),
             depth=False, scene_option=None)
```

* `name`: name of rendering mode
* `show`: rendered image is shown
* `return_pixel`: return the rendered image

It is possible to render in different render modes subsequently. Output of several render modes can be visualized at the same time.


__Example__

```python
env = dm_soccer2gym.make(domain_name="dm_soccer", task_name="2vs2", task_kwargs={'time_limit': 10.})
```

## What's new

- 2018-01-25: Optimized registering process (thanks to [rejuvyesh](https://github.com/rejuvyesh)), added access to procedurally generated environments, added render mode functionality
- 2019-09-25: Being compatible with MuJoCo200/gym=1.14.0 or later/dm_control=0.0.0
- 2019-11-23: DM Control Soccer Environment Support


## Known Error
As of 25/9/2019, with the current dm_control package(ver=0.0.0), you might be getting the error below.

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
## To solve

Run:
```shell
export MUJOCO_GL = "osmesa"
```
