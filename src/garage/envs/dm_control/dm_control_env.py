# pylint: skip-file
# flake8: noqa

import akro
from dm_control import suite
from dm_control.rl.control import flatten_observation
import numpy as np

from garage import Environment, StepType, TimeStep
from garage.envs import Step
from garage.envs.dm_control.dm_control_viewer import DmControlViewer
from garage.envs.env_spec import EnvSpec


class DmControlEnv(Environment):
    """
    Binding for `dm_control <https://arxiv.org/pdf/1801.00690.pdf>`_
    """

    def __init__(self, env, name=None, max_n_steps=500):
        self._name = name or type(env.task).__name__
        self._env = env
        self._viewer = None
        self._last_observation = None

        # action space
        action_spec = self._env.action_spec()
        if (len(action_spec.shape) == 1) and (-np.inf in action_spec.minimum or
                                              np.inf in action_spec.maximum):
            self._action_space = akro.Discrete(np.prod(action_spec.shape))
        else:
            self._action_space = akro.Box(low=action_spec.minimum,
                                          high=action_spec.maximum,
                                          dtype=np.float32)

        # observation_space
        flat_dim = self._flat_shape(self._env.observation_spec())
        self._observation_space = akro.Box(low=-np.inf,
                                           high=np.inf,
                                           shape=[flat_dim],
                                           dtype=np.float32)

        # spec
        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space)

        # max_n_steps
        self._max_n_steps = max_n_steps

        self._step_cnt = 0

    @classmethod
    def from_suite(cls, domain_name, task_name):
        return cls(suite.load(domain_name, task_name),
                   name='{}.{}'.format(domain_name, task_name))

    def step(self, action):
        # record last observation
        time_step = self._env.step(action)
        self._step_cnt += 1
        last_obs = self._last_observation

        next_obs = flatten_observation(time_step.observation)['observations']

        step_type = None
        if time_step.first():
            step_type = StepType.FIRST
        elif time_step.mid():
            if self._step_cnt == self.max_n_steps:
                step_type = StepType.TIMEOUT
            else:
                step_type = StepType.MID
        elif time_step.last():
            step_type = StepType.TERMINAL

        self._viewer.render()
        return TimeStep(env_spec=self.spec,
                        observation=last_obs,
                        action=action,
                        reward=time_step.reward,
                        next_observation=next_obs,
                        env_info=None,
                        agent_info=None,
                        step_type=step_type)

    def reset(self):
        time_step = self._env.reset()
        obs = flatten_observation(time_step.observation)['observations']
        self._last_observation = obs
        self._step_cnt = 0
        return TimeStep(env_spec=self.spec,
                        observation=None,
                        action=None,
                        reward=None,
                        next_observation=obs,
                        env_info=None,
                        agent_info=None,
                        step_type=StepType.FIRST)

    def render(self, mode='human'):
        # pylint: disable=inconsistent-return-statements
        if mode == 'rgb_array':
            return self._env.physics.render()
        else:
            raise NotImplementedError

    def visualize(self):
        if not self._viewer:
            title = 'dm_control {}'.format(self._name)
            self._viewer = DmControlViewer(title=title)
            self._viewer.launch(self._env)

    def close(self):
        if self._viewer:
            self._viewer.close()
        self._env.close()
        self._viewer = None
        self._env = None

    def _flat_shape(self, observation):
        return np.sum(int(np.prod(v.shape)) for k, v in observation.items())

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def spec(self):
        return self._spec

    @property
    def max_n_steps(self):
        return self.max_n_steps

    @property
    def render_modes(self):
        return ['rgb_array']

    def __getstate__(self):
        d = self.__dict__.copy()
        d['_viewer'] = None
        return d
