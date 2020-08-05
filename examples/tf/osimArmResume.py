"""
Created on Mon Jun  1 16:54:20 2020

@author: chris
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
def osimArmResume(ctxt = None, snapshot_dir='data/local/experiment/osimArm_153',seed=1):
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        runner.restore(snapshot_dir)
        ddpg = runner._algo
        
        env = GarageEnv(Arm2DVecEnv(visualize=True))
        env.reset()
        
        policy = ddpg.policy
    
        
        env.render()
        obs = env.step(env.action_space.sample())
        steps = 0
        n_steps = 100
        
        while True:
            if steps == n_steps:
                env.close()
                break
            temp = policy.get_action(obs[0])
            obs = env.step(temp[0])
            env.render()
            steps += 1
            
        
        
osimArmResume(seed =100)
