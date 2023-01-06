import numpy as np
from pysocialforce.scene import PedState
from pysocialforce.utils import DefaultConfig
from pysocialforce import forces

from pysocialforce.simulator import Simulator

class SocialPlanner(Simulator):

    def __init__(self, state, groups, obstacles, config_file):
        super().__init__(state, groups, obstacles, config_file)
        self._number_agents = state.shape[0]


    def convert_argument_to_state(self, **kwargs):
        state = np.zeros((self._number_agents, 7))
        for i in range(self._number_agents):
            state[i, :] = np.concatenate((
            kwargs['ob'][f'robot_{i}']['joint_state']['position'][0:2],
            kwargs['ob'][f'robot_{i}']['joint_state']['velocity'][0:2],
            kwargs[f'x_goal_{i}'],
            np.array([0.5]),
        ))
        self.peds.state = state


    def compute_action(self, **kwargs):
        self.convert_argument_to_state(**kwargs)
        action = self.compute_forces()
        return action
