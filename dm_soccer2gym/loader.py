
"""Multi-agent Single-Team MuJoCo soccer environment."""
"""File corresponds to dm_control/locomotion/soccer/__init__.py 
with about 4-5 lines modified to allow single team environments."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dm_control import composer
from dm_control.locomotion.socer import load
from dm_control.locomotion.soccer import _make_walker
from dm_control.locomotion.soccer.boxhead import BoxHead
from dm_control.locomotion.soccer.initializers import Initializer
from dm_control.locomotion.soccer.initializers import UniformInitializer
from dm_control.locomotion.soccer.observables import CoreObservablesAdder
from dm_control.locomotion.soccer.observables import InterceptionObservablesAdder
from dm_control.locomotion.soccer.observables import MultiObservablesAdder
from dm_control.locomotion.soccer.observables import ObservablesAdder
from dm_control.locomotion.soccer.pitch import Pitch
from dm_control.locomotion.soccer.pitch import RandomizedPitch
from dm_control.locomotion.soccer.soccer_ball import SoccerBall
from dm_control.locomotion.soccer.task import Task
from dm_control.locomotion.soccer.team import Player
from dm_control.locomotion.soccer.team import Team
from six.moves import range

_RGBA_BLUE = [.1, .1, .8, 1.]


def _make_players_single(team_size):
    """Construct home team of `team_size` players."""
    home_players = []
    for i in range(team_size):
        home_players.append(
                Player(Team.HOME, _make_walker("home%d" % i, i, _RGBA_BLUE)))
    return home_players


def single_team_load(team_size,
                     time_limit=45.,
                     random_state=None,
                     disable_walker_contacts=False):
    """Construct `team_size soccer environment.
    Args:
        team_size: Integer, the number of players in team. Must be between 1 and
            11.
        time_limit: Float, the maximum duration of each episode in seconds.
        random_state: (optional) an int seed or `np.random.RandomState` instance.
        disable_walker_contacts: (optional) if `True`, disable physical contacts
            between walkers.
    Returns:
        A `composer.Environment` instance.
    Raises:
        ValueError: If `team_size` is not between 1 and 11.
    """
    if team_size < 0 or team_size > 11:
        raise ValueError(
                "Team size must be between 1 and 11 (received %d)." % team_size)

    return composer.Environment(
            task=Task(
                    players=_make_players_single(team_size),
                    arena=RandomizedPitch(
                            min_size=(32, 24), max_size=(48, 36), keep_aspect_ratio=True),
                    disable_walker_contacts=disable_walker_contacts),
            time_limit=time_limit,
            random_state=random_state)