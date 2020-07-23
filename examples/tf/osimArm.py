#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:54:20 2020

@author: chris
"""

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
from garage.replay_buffer import HERReplayBuffer

from osim.env.arm import Arm2DVecEnv

from garage.np.exploration_policies import AddOrnsteinUhlenbeckNoise
from garage.replay_buffer import PathBuffer

@wrap_experiment
def osimArm(ctxt, seed=1):
    """Train TRPO with CartPole-v1 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with LocalTFRunner(ctxt) as runner:
        
        env = GarageEnv(Arm2DVecEnv(visualize=False))
        env.reset()
        
        policy = ContinuousMLPPolicy(env_spec=env.spec,
                                     hidden_sizes=[64, 64],
                                     hidden_nonlinearity=tf.nn.relu,
                                     output_nonlinearity=tf.nn.tanh)

        exploration_policy = AddOrnsteinUhlenbeckNoise(env.spec,
                                                       policy,
                                                       sigma=0.2)

        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=[64, 64],
                                    hidden_nonlinearity=tf.nn.relu)

        # replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

        

        replay_buffer = HERReplayBuffer(capacity_in_transitions=int(1e6),
                                        replay_k=4,
                                        reward_fn=env.compute_reward,
                                        env_spec=env.spec)

        ddpg = DDPG(env_spec=env.spec,
                    policy=policy,
                    policy_lr=1e-4,
                    qf_lr=1e-3,
                    qf=qf,
                    replay_buffer=replay_buffer,
                    max_path_length=100,
                    steps_per_epoch=20,
                    target_update_tau=1e-2,
                    n_train_steps=50,
                    discount=0.9,
                    min_buffer_size=int(1e4),
                    exploration_policy=exploration_policy,
                    policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                    qf_optimizer=tf.compat.v1.train.AdamOptimizer)
        
        # env.render()
        # obs = env.step(env.action_space.sample())
        # steps = 0
        # n_steps = 1000
        
        # while True:
        #     if steps == n_steps:
        #         env.close()
        #         break
        #     temp = policy.get_action(obs[0])
        #     obs = env.step(temp[0])
        #     env.render()
        #     steps += 1
            
        

        runner.setup(algo=ddpg, env=env)
        runner.train(n_epochs = 100,batch_size = 10)
        
        
osimArm(seed =100)
