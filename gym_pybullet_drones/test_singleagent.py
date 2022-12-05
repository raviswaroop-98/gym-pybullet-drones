"""Test script for single agent problems.

This scripts runs the best model found by one of the executions of `singleagent.py`

Example
-------
To run the script, type in a terminal:

    $ python test_singleagent.py --exp ./results/save-<env>-<algo>-<obs>-<act>-<time_date>

"""
import os
import time
from datetime import datetime
import argparse
import re
import numpy as np
import gym
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy as a2cppoCnnPolicy
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from stable_baselines3.sac import CnnPolicy as sacCnnPolicy
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from stable_baselines3.td3 import CnnPolicy as td3ddpgCnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.envs.single_agent_rl.TuneAviary import TuneAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync, str2bool

import shared_constants

DEFAULT_GUI = True
DEFAULT_PLOT = True
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_ALGO = 'ddpg'

def run(exp, gui=DEFAULT_GUI, plot=DEFAULT_PLOT, output_folder=DEFAULT_OUTPUT_FOLDER):
    #### Load the model from file ##############################
    default = "/Users/raviswarooprayavarapu/Documents/gym-pybullet-drones/gym_pybullet_drones/examples/results"
    if DEFAULT_ALGO == 'ppo':
        path = default+"/save-hover-ppo-kin-rpm-11.23.2022_17.14.52"
        mod = path + "/success_model.zip"
        model = PPO.load(mod)
    elif DEFAULT_ALGO == 'ddpg':
        path = default+"/save-hover-ddpg-kin-rpm-11.23.2022_23.49.26"
        mod = path + "/success_model.zip"
        model = DDPG.load(mod)

    #### Parameters to recreate the environment ################
    env_name = "hover-aviary-v0"
    OBS = ObservationType.KIN
    # Parse ActionType instance from file name
    ACT = ActionType.RPM
    init = np.array([[0,0,1]])
    prop_loss = 0
    #### Evaluate the model ####################################
    eval_env = gym.make(env_name,
                        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                        obs=OBS,
                        act=ACT,
                        initial_xyzs=init,
                        prop_fault = prop_loss
                        )
    mean_reward, std_reward = evaluate_policy(model,
                                              eval_env,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    #### Show, record a video, and log the model's performance #
    test_env = gym.make(env_name,
                        gui=gui,
                        record=False,
                        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                        obs=OBS,
                        act=ACT,
                        initial_xyzs=init,
                        prop_fault = prop_loss
                        )
    logger = Logger(logging_freq_hz=int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS),
                    num_drones=1,
                    output_folder=output_folder
                    )
    #import pdb;pdb.set_trace()
    obs = test_env.reset()
    start = time.time()
    t_reward = 0
    done = False
    for i in range(6*int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS)): # Up to 6''
        action, _states = model.predict(obs,
                                        deterministic=True # OPTIONAL 'deterministic=False'
                                        )
        #action = np.array([1,0.5])
        obs, reward, done, info = test_env.step(action)
        t_reward += reward
        test_env.render()
        if OBS==ObservationType.KIN:
            if test_env.prop_fault==1:
                action = np.append(action,[0])
            if test_env.prop_fault==2:
                action = np.array([action[0],0,action[1],0])
            logger.log(drone=0,
                       timestamp=i/test_env.SIM_FREQ,
                       state= np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
                       control=np.zeros(12)
                       )
        sync(np.floor(i*test_env.AGGR_PHY_STEPS), start, test_env.TIMESTEP)
        # if done: obs = test_env.reset() # OPTIONAL EPISODE HALT
    test_env.close()
    #logger.save_as_csv("sa") # Optional CSV save
    print("Total Reward : ",t_reward)
    # logger.save_as_csv()
    if plot:
        logger.plot()

    # with np.load(path+'/evaluations.npz') as data:
    #     print(data.files)
    #     print(data['timesteps'])
    #     print(data['results'])
    #     print(data['ep_lengths'])

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using TakeoffAviary')
    parser.add_argument('--exp',                           type=str,            help='The experiment folder written as ./results/save-<env>-<algo>-<obs>-<act>-<time_date>', metavar='')
    parser.add_argument('--gui',            default=DEFAULT_GUI,               type=str2bool,      help='Whether to use PyBullet GUI (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))