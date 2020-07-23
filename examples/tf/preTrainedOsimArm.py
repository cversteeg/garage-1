#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm.

Here it runs CartPole-v1 environment with 100 iterations.

Results:
    AverageReturn: 100
    RiseTime: itr 13
"""
import gym
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GarageEnv
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import AddOrnsteinUhlenbeckNoise
from garage.tf.algos import DDPG
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction

from osim.env.arm import Arm2DVecEnv

from garage.np.exploration_policies import AddOrnsteinUhlenbeckNoise
from garage.replay_buffer import PathBuffer

@wrap_experiment
def preTrainedOsimArm(ctxt, seed=1, snapshot_dir = 'data/local/experiment/osimArm_69'):
    """Train TRPO with CartPole-v1 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with LocalTFRunner(snapshot_config = ctxt) as runner:
        runner.restore(snapshot_dir)
        runner.resume(n_epochs = 30, batch_size = 1000)
        
        
preTrainedOsimArm(seed = 100)
