import numpy as np
import pandas as pd
import random, os, datetime, math


np.random.seed(88)

total_reward = 0
bandit_dict = {}

def get_next_bandit():
    best_bandit = 0
    best_bandit_expected = 0
    for bnd in bandit_dict:
        expect = (bandit_dict[bnd]['win'] - bandit_dict[bnd]['loss'] + bandit_dict[bnd]['opp'] - (bandit_dict[bnd]['opp']>0)*1.5) / (bandit_dict[bnd]['win'] + bandit_dict[bnd]['loss'] + bandit_dict[bnd]['opp']) * math.pow(0.97, bandit_dict[bnd]['win'] + bandit_dict[bnd]['loss'] + bandit_dict[bnd]['opp'])
        if expect > best_bandit_expected:
            best_bandit_expected = expect
            best_bandit = bnd
    return best_bandit

my_action_list = []
op_action_list = []

def multi_armed_probabilities(observation, configuration):
    global total_reward, bandit_dict

    if observation['step']==0:
        total_reward = 0
        bandit_dict = {}
        for i in range(configuration['banditCount']):
            bandit_dict[i] = {'win': 1, 'loss': 0, 'opp': 0}
        return np.random.randint(configuration['banditCount'])

    last_reward = observation['reward'] - total_reward
    total_reward = observation['reward']

    my_idx = observation['agentIndex']
    my_last_action = observation['lastActions'][my_idx]
    op_last_action = observation['lastActions'][1-my_idx]

    my_action_list.append(my_last_action)
    op_action_list.append(op_last_action)

    bandit_dict[my_last_action]['win'] = bandit_dict[my_last_action]['win'] + last_reward
    bandit_dict[my_last_action]['loss'] = bandit_dict[my_last_action]['loss'] + 1 - last_reward
    bandit_dict[op_last_action]['opp'] = bandit_dict[op_last_action]['opp'] + 1
    
    if last_reward:
        return my_last_action

    # if the last 3 actions were the same, do the same again 50%
    if observation['step'] >= 4 and (my_action_list[-1] == my_action_list[-2] == my_action_list[-1] == my_action_list[-3]) and np.random.rand() < 0.5:
            return my_action_list[-1]
    return get_next_bandit()
