
from dm_control.composer.observation import observable as base_observable
from dm_control.locomotion.soccer.observables import InterceptionObservablesAdder


class InterceptionObservablesAdderv2(InterceptionObservablesAdder):
    
    def __call__(self, task, player):
    
        super().__call__(task, player)
        
        def _stats_i_received_pass(unused_physics):
          if (task.ball.hit and task.ball.repossessed and
              task.ball.last_hit is player and 
              not(task.ball.intercepted)):
            return 1.
          return 0.

        player.walker.observables.add_observable(
            'stats_i_received_pass',
            base_observable.Generic(_stats_i_received_pass))

        for dist in [5, 10, 15]:

          def _stats_i_received_pass_dist(physics, dist=dist):
            if (_stats_i_received_pass(physics) and
                task.ball.dist_between_last_hits is not None and
                task.ball.dist_between_last_hits > dist):
              return 1.
            return 0.

          player.walker.observables.add_observable(
              'stats_i_received_pass_%dm' % dist,
              base_observable.Generic(_stats_i_received_pass_dist))

