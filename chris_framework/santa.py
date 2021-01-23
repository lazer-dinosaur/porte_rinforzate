from joblib import Parallel, delayed
from collections import defaultdict
import itertools as it
import pandas as pd
import numpy as np
import timing
import datetime
import time
# import fastcore
from scipy.stats import norm, sem, beta
from scipy.special import erf
import functools
from matplotlib import colors
import pylab as p
import analysis
from print_override import *
import utilities as ut

p.style.use('ggplot')

pd.set_option('display.max_rows', 80)
pd.set_option('display.max_columns', 400)
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None


class Agent:
    '''
    essential stuff is in `update` and `take_action`
    allows access to the bandit and other agent internals, for direct cheating and comparisons, which is quite a nice thing to have.

    '''
    def __init__(self,num_ban,label=None,rand_state=None,active=True,params={}):
        '''
        just initialise fucking everything for the moment, cos its probably quite cheap
        '''
        self.num_ban = num_ban
        self.rand_state = rand_state
        if label:
            self.label = label
        else:
            self.label = __class__.__name__

        self.active = True
        if rand_state:
            self.random = np.random.RandomState(rand_state)
        else:
            self.random = np.random.RandomState()
        self.params = params

        self.reward_sequence = [[] for i in range(num_ban)]
        self.reward_sequence_no = [[] for i in range(num_ban)]
        self.reward_sequence_entry_cycle = [[] for i in range(num_ban)]
        self.mean_reward = np.zeros(num_ban) + 0.5
        self.score = 0
        self.scores = np.zeros(num_ban)
        self.num_pulls = np.zeros(num_ban)
        self.num_pulls_opp = np.zeros(num_ban)
        self.num_pulls_total = np.zeros(num_ban)
        self.num_pulls_total_old = np.zeros(num_ban)
        self.last_reward = np.ones(num_ban)
        # origin threshold estimates:
        self.ote = np.full(num_ban,0.5)
        self.current_vals = np.full(num_ban,0.5)
        self.current_vals_alt = np.full(num_ban,0.5)
        self.current_vals_no = np.full(num_ban,0.5)
        self.current_vals_alt_no = np.full(num_ban,0.5)
        # self.current_vals = np.zeros(num_ban)
        # self.payout_history[choice,time_step] = reward
        self.post_a = np.ones(num_ban)
        self.post_b = np.ones(num_ban)
        self.post_a_strict = np.ones(num_ban)
        self.post_b_strict = np.ones(num_ban)

        self.action_history = []
        self.reward_history = []
        sss = 1
        self.α = np.ones((sss,num_ban))*1
        self.β = np.ones((sss,num_ban))*1
        self.w = np.ones((sss))

        self.x_ = np.tile(np.linspace(0,1,101),(num_ban,1))
        self.pmf_ = np.full((self.num_ban,101),1/101)
        self.x__ = np.tile(np.linspace(0,1,101),(num_ban,1))
        self.pmf__ = np.full((self.num_ban,101),1/101)
        self.last_choice_opp = np.nan
        self.cheat_state = np.zeros((10,10))

        self.μ_hist = []
        self.σ_hist = []

        depth = 200
        self.r = np.full((self.num_ban,depth),np.nan)
        self.moves = np.full((self.num_ban,depth),np.nan)
        self.moves_opp = np.full((self.num_ban,depth),np.nan)
        self.μ = np.ones(num_ban)

        self.geom_a = np.zeros(num_ban)
        self.geom_b = np.zeros(num_ban)
        self.geom_pulls_opp = np.zeros(num_ban)

    # @timing.timing()
    def update(self,opponent,bandits):
        self.score += self.reward
        self.scores[self.choice] += self.reward
        self.action_history.append(self.choice)
        self.reward_history.append(self.reward)

        self.non_essential_update(opponent,bandits)

        ################# THIS MUST BE LAST #################
        self.agent_update(opponent,bandits)

    # @timing.timing()
    def non_essential_update(self,opponent,bandits):
        '''
        override to specify whatever custom metrics you want to retain or calculate
        '''
        self.num_pulls[self.choice] += 1
        self.num_pulls_opp[opponent.choice] += 1
        self.num_pulls_total[self.choice] += 1
        self.num_pulls_total[opponent.choice] += 1

        # if self.time_step==1:
        #     self.last_reward = np.zseros(self.num_ban)
        self.last_reward[self.choice] = self.reward
        self.incremental_mean(self.choice,self.reward)

        self.post_a_strict[self.choice] += self.reward #* 0.97
        self.post_b_strict[self.choice] += (1 - self.reward)

        # self.r[self.choice] = np.roll(self.r[self.choice],-1)
        self.r[self.choice,:-1] = self.r[self.choice][1:]
        self.r[self.choice,-1] = self.reward

        # self.moves[self.choice] = np.roll(self.moves[self.choice],-1)
        self.moves[self.choice][:-1] = self.moves[self.choice][1:]
        self.moves[self.choice,-1] = self.time_step

        self.last_choice_opp = opponent.choice
        # self.moves_opp[opponent.choice] = np.roll(self.moves_opp[opponent.choice],-1)
        self.moves_opp[opponent.choice][:-1] = self.moves_opp[opponent.choice][1:]
        self.moves_opp[opponent.choice,-1] = self.time_step

        self.reward_sequence[self.choice].append(int(self.reward))
        self.reward_sequence[opponent.choice].append(np.nan)
        self.current_vals[self.choice] = self.current_p_estimate(self.reward_sequence[self.choice])
        self.current_vals_alt[self.choice] = self.current_p_estimate_alt(self.reward_sequence[self.choice])


        self.reward_sequence_no[self.choice].append(int(self.reward))
        self.current_vals_no[self.choice] = self.current_p_estimate(self.reward_sequence_no[self.choice])

        self.current_vals_alt_no[self.choice] = self.current_p_estimate_alt(self.reward_sequence_no[self.choice])

        # if self.__class__.__name__!='UCB':
        #     A1_score_est = self.OTE_score_predictor(predict_A1=True)
        #     A2_score_est = self.OTE_score_predictor(predict_A1=False)
        #     print(f'A1: real: {opponent.score:.2f}, est: {A1_score_est:.2f}, A2 real: {self.score:.2f}: Est: {A2_score_est:.2f}')

    def agent_update(self,opponent,bandits):
        pass

    # from math import * # MATH ALSO CONTAINS AN ERF
    def norm_cdf(self,x):
        ''' Cumulative distribution function for the standard normal distribution '''
        return (1.0 + erf(x / 2.0 ** 0.5)) / 2.0

    def norm_(self,a):
        return a / np.sum(a)

    def get_means(self):
        return self.post_a / (self.post_a + self.post_b)

    def get_vars(self):
        return self.post_a * self.post_b / ((self.post_a + self.post_b)**2 * (self.post_a + self.post_b + 1))

    def incremental_mean(self,choice,reward):
        # old_reward = np.nan_to_num(self.mean_reward[choice])
        old_reward = self.mean_reward[choice]
        self.mean_reward[choice] = old_reward  + (reward - old_reward) / self.num_pulls[choice]

    def argmax_tie_rand(self,a):
        return self.random.choice(np.flatnonzero(a==np.max(a)))

    def nan_argmax_tie_rand(self,a):
        return self.random.choice(np.flatnonzero(a==np.nanmax(a)))

    def argmax_tie_rand_axis(self,a):
        z = []
        for row in a:
            z.append(self.random.choice(np.flatnonzero(row==np.max(row))))
        return z

    def determine_ote_pmf(self,seq):
        '''
        1,1,1,0,1
        p*p*DECAY**1*p*DECAY**2*(1-p*DECAY**3)*p*DECAY**4
        '''
        seq = np.array(seq)
        a = np.arange(0,1.01,0.01)
        decays = DECAY**np.arange(seq.shape[0])
        a_ = a[:,None] * decays
        a_ = a_*seq + (1-a_)*(1-seq)
        prod = np.nanprod(a_,axis=1)
        return prod / np.sum(prod)

    # @timing.timing_p
    @ut.cache(1)
    def current_p_estimate(self,seq):
        seq = np.array(seq)
        a = np.arange(0,1.01,0.01)
        decays = DECAY**np.arange(seq.shape[0])
        a_ = a[:,None] * decays
        a_ = a_*seq + (1-a_)*(1-seq)
        prod = np.nanprod(a_,axis=1)
        return np.mean(a[np.flatnonzero(prod==np.max(prod))]) * DECAY ** seq.shape[0]

    @ut.cache(2)
    def current_p_estimate_alt(self,seq):
        seq = np.array(seq)
        a = np.arange(0,1.01,0.01)
        decays = DECAY**np.arange(seq.shape[0])
        a_ = a[:,None] * decays
        a_ = a_*seq + (1-a_)*(1-seq)
        prod = np.nanprod(a_,axis=1)
        prod = prod / np.sum(prod)
        return np.sum(a*prod) * DECAY ** seq.shape[0]

    # @ut.cache(3)
    def initial_p_estimate(self,seq):
        seq = np.array(seq)
        a = np.arange(0,1.01,0.01)
        decays = DECAY**np.arange(seq.shape[0])
        a_ = a[:,None] * decays
        a_ = a_*seq + (1-a_)*(1-seq)
        prod = np.nanprod(a_,axis=1)
        return np.mean(a[np.flatnonzero(prod==np.max(prod))])

    def points_from_OTE(self,A2_seq,predict_A1):
        p = self.initial_p_estimate(A2_seq)
        if predict_A1:
            return np.nansum(p * np.isnan(A2_seq).astype(int) * DECAY ** np.arange(0,len(A2_seq)))
        else:
            return np.nansum(p * (~np.isnan(A2_seq)).astype(int) * DECAY ** np.arange(0,len(A2_seq)))

    def OTE_score_predictor(self,predict_A1=True):
        ''' predicts A1 - replace A1 with A2 OTE, to predict A2 use predict_A1=False'''
        points = []
        for A2_seq in self.reward_sequence:
            points.append(self.points_from_OTE(A2_seq,predict_A1))
        return np.sum(points)

    # @timing.timing()
    def sample_mean_and_std(self,a,b):
        # μ, σ = beta.mean(a,b), beta.std(a,b) # THESE ARE SLOWER
        a_b = a + b
        μ = a / a_b
        σ = (a * b / (a_b**2 * (a_b + 1))) ** 0.5
        return μ, σ

    def sample_std(self,a,b):
        a_b = a + b
        σ = (a * b / (a_b**2 * (a_b + 1))) ** 0.5
        return σ

    # @timing.timing()
    def sample_window(self,depth):
        a = np.nansum(self.r[:,-depth:],axis=1) + 1
        b = np.clip(self.num_pulls,0,depth) - a + 2
        return a, b


class Interface(Agent):
    ''' way of interfacing to competition API '''
    def __init__(self,num_ban,label=None,rand_state=None,active=True,params={}):
        self.A1 = params['A1']
        self.configuration = {'banditCount':num_ban}
        self.observation = {
                            'step':0,
                            'reward':0,
                            'agentIndex':0,
                            'lastActions':None
                            }
        super().__init__(num_ban,label,rand_state)

    def take_action(self,opponent=None,bandits=None):
        return self.A1(self.observation,self.configuration)

    def non_essential_update(self,opponent,bandits):
        self.observation['step'] = self.time_step + 1
        self.observation['reward'] = self.score
        self.observation['lastActions'] = [self.choice,opponent.choice]


class Cheat(Agent):
    ''' returns the best current bandit '''
    def take_action(self,opponent=None,bandits=None):
        return self.argmax_tie_rand(bandits.history[:,self.time_step])

    def non_essential_update(self,opponent,bandits):
        pass

class RandomAgent(Agent):
    def take_action(self,opponent=None,bandits=None):
        return self.random.randint(self.num_ban)

    def non_essential_update(self,opponent,bandits):
        pass

class EpsilonGreedy(Agent):
    def take_action(self,opponent=None,bandits=None):
        if self.random.rand()>self.params.get('epsilon',-.1):
            return self.argmax_tie_rand(self.mean_reward)
        return self.random.choice(self.num_ban)

class EpsilonGreedyDecay(Agent):
    def take_action(self,opponent=None,bandits=None):
        if self.random.rand()>self.params.get('epsilon',-.1):
            return np.argmax(self.mean_reward*DECAY**(self.num_pulls_total))
        return self.random.choice(self.num_ban)

class Thompson(Agent):
    def take_action(self,opponent=None,bandits=None):
        return int(self.argmax_tie_rand(self.random.beta(self.post_a,self.post_b)))
        # return int(np.argmax(self.random.beta(self.post_a,self.post_b)))

    def agent_update(self,opponent,bandits):
        self.post_a[self.choice] += self.reward
        self.post_b[self.choice] += (1 - self.reward)

class UCB(Agent):
    def take_action(self,opponent=None,bandits=None):
        μ, σ = self.sample_mean_and_std(self.post_a,self.post_b)
        return self.argmax_tie_rand(μ + self.params.get('std_val',3)*σ)

    def agent_update(self,opponent,bandits):
        self.post_a[self.choice] += self.reward
        self.post_b[self.choice] += (1 - self.reward)

class UCB_decay_no(Agent):
    '''
    this was to try things out in a non-dual situation
    must set `no_opponent=True`
    '''
    def take_action(self,opponent=None,bandits=None):
        μ, σ = self.sample_mean_and_std(self.post_a,self.post_b)
        # if self.time_step==1999:
        #     print(np.corrcoef(bandits.history[:,self.time_step],μ)[0,1])
        #     print(np.corrcoef(bandits.history[:,self.time_step],self.current_vals_no)[0,1])
        #     ix = np.argsort(bandits.history[:,self.time_step])[::-1]
        #     print(bandits.history[:,0][ix],a=1)
        #     print(bandits.history[:,self.time_step][ix],a=1)
        #     print(μ[ix],a=1)
        #     print(self.current_vals_no[ix],a=1,n='current_vals_no')
        #     print(self.current_vals_alt_no[ix],a=1,n='current_vals_alt_no')
        #     print(self.num_pulls[ix],a=1)
        #     print()
        #     print(ix)
        #     print(self.reward_sequence_no[ix[0]])
        #     print(self.reward_sequence_no[94])
        #     print()

        # 876.97 45.34808816256756 4.557654345308903
        # 0.7835051546391752 % wins for UCB_decay_no
        # UCB wins: 21, UCB_decay_no wins: 76, draws: 3
        σ_factor = self.params.get('σ_factor',1)
        time_strength = self.params.get('time_strength',0.5)
        time_delay = self.params.get('time_delay',0)
        time_denom = self.params.get('time_denom',2000)
        return self.argmax_tie_rand(self.current_vals_alt_no + σ_factor / (np.clip(self.num_pulls+1 - time_strength*(np.clip(self.time_step-time_delay,0,np.inf)/time_denom),1,np.inf))**2)


    def agent_update(self,opponent,bandits):
        self.post_a[self.choice] += self.reward
        self.post_b[self.choice] += (1 - self.reward)


class UCB_discount(Agent):
    def take_action(self,opponent=None,bandits=None):
        ''' doesnt work good '''
        μ_ = np.sum(self.x_ * self.pmf_,axis=1)
        # μ_ = self.x_[range(self.x_.shape[0]),np.argmax(self.pmf_,axis=1)]
        σ_ = np.sum((self.x_ - μ_[:,None]) ** 2 * self.pmf_,axis=1) ** 0.5

        μ, σ = self.sample_mean_and_std(self.post_a,self.post_b)
        # self.μ_hist.append(μ_)
        # self.σ_hist.append(σ_)
        # self.plot(μ_,σ_,μ,σ)
        return self.argmax_tie_rand(μ_ + 0.1*(np.log(self.time_step)/self.num_pulls)**0.5)

    def agent_update(self,opponent,bandits):
        self.post_a[self.choice] += self.reward
        self.post_b[self.choice] += (1 - self.reward)

        self.reward_sequence[self.choice].append(int(self.reward))
        self.reward_sequence[opponent.choice].append(np.nan)

        x = np.linspace(0,1,101)
        self.x_[self.choice] = x * DECAY ** self.num_pulls_total[self.choice]
        self.x_[opponent.choice] = x * DECAY ** self.num_pulls_total[opponent.choice]

        self.pmf_[self.choice] = self.determine_ote_pmf(self.reward_sequence[self.choice])

class UCB_current(UCB):
    def take_action(self,opponent=None,bandits=None):
        μ, σ =  self.sample_mean_and_std(self.post_a,self.post_b)
        μ = self.current_vals
        return self.argmax_tie_rand(μ + 3*σ)

class UCB_selfish(UCB):
    '''UCB wins: 43, UCB_selfish wins: 55, draws: 2'''
    def take_action(self,opponent=None,bandits=None):
        ix = np.logical_and(self.last_reward,self.num_pulls_opp==0)
        if np.sum(ix):
            return self.argmax_tie_rand(ix)
        μ, σ =  self.sample_mean_and_std(self.post_a,self.post_b)
        return self.argmax_tie_rand(μ + 3*σ)

class UCB_window(UCB):
    '''UCB wins: 44, UCB_window wins: 53, draws: 2'''
    def take_action(self,opponent=None,bandits=None):
        a,b = self.sample_window(20)
        μ, σ =  self.sample_mean_and_std(a,b)
        return self.argmax_tie_rand(μ + 3*σ)

class UCB_window_long_std(UCB_window):
    ''' UCB_window wins: 49, UCB_window_long_std wins: 49, draws: 2 ??????? '''
    def take_action(self,opponent=None,bandits=None):
        a,b = self.sample_window(20)
        μ, σ =  self.sample_mean_and_std(a,b)
        μ_, σ_ =  self.sample_mean_and_std(self.post_a,self.post_b)
        return self.argmax_tie_rand(μ + 3*σ_)

class UCB_window_selfish(UCB_window):
    ''' UCB wins: 43, UCB_window_selfish wins: 57, draws: 0 '''
    def take_action(self,opponent=None,bandits=None):
        ix = np.logical_and(self.last_reward,self.num_pulls_opp==0)
        if np.sum(ix):
            return self.argmax_tie_rand(ix)
        a,b = self.sample_window(20)
        μ,σ =  self.sample_mean_and_std(a,b)
        return self.argmax_tie_rand(μ + 3*σ)

class UCB_window3(UCB_window):
    '''UCB wins: 22, UCB_window3 wins: 77, draws: 1'''
    def take_action(self,opponent=None,bandits=None):
        ix = np.logical_and(self.last_reward,self.num_pulls_opp==0)
        if np.sum(ix):
            return self.argmax_tie_rand(ix)
        w = np.sum(self.moves_opp>self.time_step-100,axis=1)
        a,b = self.sample_window(20)
        a_b = a + b - w
        μ = a / a_b
        # σ = (a * b / (a_b**2 * (a_b + 1))) ** 0.5
        σ = sample_std(self.post_a,self.post_b)
        return self.argmax_tie_rand(μ + 3*σ)

class UCB_window00(UCB_window):
    '''
    0.83 % wins for UCB_window00
    UCB wins: 17, UCB_window00 wins: 83, draws: 0
    '''
    def take_action(self,opponent=None,bandits=None):
        # random first door
        if self.time_step==0:
            return self.random.randint(0,100)
        # if it won and opponent didnt pick it, do it again ### really???
        if self.num_pulls[self.choice]==self.scores[self.choice] and self.num_pulls_opp[self.choice]==0:
            return self.choice
        # explore a totally new door to all players:
        ix = self.num_pulls_total==0
        if np.sum(ix):
            return self.argmax_tie_rand(ix)

        # ix = np.logical_and(self.last_reward,self.num_pulls_opp==0)
        # if np.sum(ix):
        #     return self.argmax_tie_rand(ix)
        #     # return self.argmax_tie_rand(ix)

        # w = np.sum(self.moves_opp>self.time_step-100,axis=1)
        w = np.sum(self.moves_opp>self.time_step-200,axis=1)
        # a,b = self.sample_window(20)
        a,b = self.sample_window(37)
        a_b = a + b - w + 4
        μ = a / a_b

        # MAX_BOUNDS = DECAY ** self.num_pulls_total
        # μ = np.clip(μ,0,MAX_BOUNDS)
        # σ = (a * b / (a_b**2 * (a_b + 1))) ** 0.5

        a_b = self.post_a + self.post_b
        σ = (self.post_a * self.post_b / (a_b**2 * (a_b + 1))) ** 0.5
        return self.argmax_tie_rand(μ + 3*σ)
        # return self.argmax_tie_rand(np.clip(μ + 3*σ,0,MAX_BOUNDS))


class Enemy(Agent):
    ''' faster re-write of the somewhat mad, but rather good public agent '''
    def take_action(self,opponent=None,bandits=None):
        if self.time_step==0:
            self.post_b -= 1
            return self.random.randint(self.num_ban)
        if self.reward:
            return self.choice
        if self.time_step>3 and np.all(self.choice==self.action_history[-3:]) and self.random.rand() < 0.5:
            return self.choice
        expect = (self.post_a - self.post_b + self.num_pulls_opp-(self.num_pulls_opp>0)*1.5) / (self.num_pulls_total + 1) * DECAY**(self.num_pulls_total + 1)
        return self.nan_argmax_tie_rand(expect)

    def agent_update(self,opponent,bandits):
        self.post_a[self.choice] += self.reward
        self.post_b[self.choice] += 1 - self.reward

    def non_essential_update(self,opponent,bandits):
        self.num_pulls[self.choice] += 1
        self.num_pulls_opp[opponent.choice] += 1
        self.num_pulls_total[self.choice] += 1
        self.num_pulls_total[opponent.choice] += 1


class Enemy_hat(Agent):
    '''
    Apparently a marginal improvement of `Enemy`
    0.547550432276657 % wins for Enemy_hat
    Enemy wins: 314, Enemy_hat wins: 380, draws: 13
    '''
    def take_action(self,opponent=None,bandits=None):
        if self.time_step==0:
            self.post_b -= 1
            return self.random.randint(self.num_ban)
        if self.reward and self.time_step<350:
            return self.choice
        expect = (self.post_a - self.post_b + self.num_pulls_opp-(self.num_pulls_opp>0)*1.5) / (self.num_pulls_total + 1) * DECAY**(self.num_pulls_total + 1)
        return self.nan_argmax_tie_rand(expect)

    def agent_update(self,opponent,bandits):
        self.post_a[self.choice] += self.reward
        self.post_b[self.choice] += 1 - self.reward

class Enemy8(Agent):
    '''
    0.5302663438256658 % wins for Enemy8
    Enemy wins: 194, Enemy8 wins: 219, draws: 9
    '''
    def take_action(self,opponent=None,bandits=None):
        if self.time_step==0:
            self.post_b -= 1
            return self.random.randint(self.num_ban)
        if self.reward and self.time_step<500:
            return self.choice
        expect = (self.post_a - self.post_b + self.num_pulls_opp-(self.num_pulls_opp>0)*1.5) / (self.num_pulls_total + 1) * DECAY**(self.num_pulls_total + 1)
        return self.nan_argmax_tie_rand(expect)

    def agent_update(self,opponent,bandits):
        self.post_a[self.choice] += self.reward
        self.post_b[self.choice] += 1 - self.reward


class RL(Agent):
    ''' 
    https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/ 
    Q-learning, basic implementation.. essentially useless..
    '''
    def __init__(self,num_ban,label=None,rand_state=None,active=True,params={}):
        super().__init__(num_ban,label,rand_state,active,params)

        self.trial_num = 0

        self.state = tuple(np.zeros(3*num_ban))
        self.states = []
        self.ix = np.arange(self.num_ban)

        self.zeros = np.zeros(self.num_ban)

        self.Q = {self.state:np.zeros(num_ban)}#np.zeros((num_ban,num_ban))
        self.Q_count = defaultdict(int)
        # learning rate
        self.α = 0.1
        # discounting rate = importance of future rewards
        self.γ = 0.7
        # explore:
        self.ε = 0.1

    def _missing_entry(self,value):
        return lambda: value

    @timing.timing_p
    def take_action(self,opponent=None,bandits=None):
        if self.random.rand()<self.ε and self.params.get('explore_until',np.inf)>self.trial_num:
            return self.random.choice(self.num_ban)
        return self.argmax_tie_rand(self.Q.get(self.state,self.zeros))

    @timing.timing_p
    def non_essential_update(self,opponent,bandits):
        self.num_pulls[self.choice] += 1
        self.num_pulls_opp[opponent.choice] += 1
        self.num_pulls_total[self.choice] += 1
        self.num_pulls_total[opponent.choice] += 1

        a = self.scores.tolist()
        b = self.num_pulls.tolist()
        c = self.num_pulls_opp.tolist()
        next_state = tuple(a+b+c)
        self.states.append([self.state,next_state,self.choice])
        self.state = next_state

    @timing.timing_p
    def reset(self,opponent):
        self.trial_num += 1

        if self.score>opponent.score:
            reward = 1
        elif self.score<opponent.score:
            reward = -1
        else:
            reward = 0

        for state,next_state,choice in self.states:
            update = (1-self.α)*self.Q.setdefault(state,np.zeros(self.num_ban))[choice] + self.α*(reward+self.γ*np.max(self.Q.setdefault(next_state,np.zeros(self.num_ban))))

            self.Q[state][choice] = update
            self.Q_count[state] += 1

        self.score = 0
        self.scores = np.zeros(self.num_ban)
        self.num_pulls = np.zeros(self.num_ban)
        self.num_pulls_opp = np.zeros(self.num_ban)
        self.state = (0,)*3*self.num_ban
        self.action_history = []
        self.reward_history = []
        self.states = []


####################################################################################
####################################################################################
####################################################################################

class Bandits():
    def __init__(self,num_ban,num_cycles,rand_state=None,verbose=False,parallel=False):
        self.parallel = parallel
        self.rand_state = rand_state
        if rand_state:
            self.random = np.random.RandomState(rand_state)
        else:
            self.random = np.random.RandomState()
        # assume uniform:
        self.history = np.zeros((num_ban,num_cycles+1))
        # self.history[:,0] = self.random.rand(num_ban)
        self.history[:,0] = self.random.randint(1,101,num_ban,) / 100
        if verbose:
            print(self.history[:,0],'truth')
        # self.history[:,0] = [0,0.1,0.9,0.6]

        # self.history = random.randint(1,81,(num_ban,num_cycles+1)) / 100
        # self.history = np.full((num_ban,num_cycles+1),0.5)
        # self.history = np.zeros((num_ban,num_cycles+1))
        # self.history[0,0] = 0.7# 1 # 0.7
        self.history[:,1:] = 0
        self.reward_sequence = [[] for i in range(num_ban)]

    def get_reward(self,agent):
        return int(self.random.rand()<self.history[agent.choice,self.time_step])

    # @timing.timing_p
    def update(self,A1,A2=None):
        # copy hack for parallelisation:
        if self.parallel:
            self.history = np.array(self.history)
        self.history[:,self.time_step+1] = self.history[:,self.time_step]

        self.history[A1.choice,self.time_step+1] = self.history[A1.choice,self.time_step+1] * DECAY
        if A2:
            self.history[A2.choice,self.time_step+1] = self.history[A2.choice,self.time_step+1] * DECAY



if __name__ == '__main__':

    DECAY = 0.97
    PRINT = 1
    ENUMER = True
    TIMER = 1

    benchmark_models = [
                        # RandomAgent,
                        # EpsilonGreedy,
                        # EpsilonGreedyDecay,
                        # Thompson,
                        UCB,
                        Enemy,
                        # Enemy,
                        # Enemy22,
                        # Enemy7,
                        Enemy8,
                        # Enemy_hat,
                        # UCB_selfish,
                        # UCB_window,
                        UCB_window33,
                        RL
                        ]

    trial_models = [
                        UCB,
                        # RandomAgent,
                        # Thompson,
                        # Cheat,
                        # UCB,
                        # Interface,
                        Enemy,
                        # Enemy22,
                        # Enemy7,
                        # Enemy8,
                        # Enemy_hat,
                        # UCB_window33,
                        # UCB_window00,
                        # RL
                        ]

    params = {
                # 'opponent_history':np.arange(100,400),
                # 'sample_window_depth':np.arange(25,60),
                # 'constant':np.arange(0,6),
                # 'flip':[1],
                # 'avoid_opponent':[1,0],
                # 'mult':[1,0],
                # 'std_val':np.arange(3.1,3.9,0.1),
                # 'std_val':np.arange(2,5,0.1),
                # 'σ_factor':np.arange(5,9,1),
                # 'σ_factor':np.arange(1,7,1),
                'σ_factor':np.arange(0,3,0.1),
                'σ_power':[1,2],
                'cv_choice':[0,1],
                # 'σ_power':[1,2],
                # 'time_strength':[0.5,1,1.5,2,2.5],
                # 'time_denom':[500,1000,2000,3000,4000],
                # 'time_delay':[-500,0,500,1000,1500],
                }

    import opponent_agent
    A1_params = {'A1':opponent_agent.multi_armed_probabilities}
    best_params = {
                'opponent_history':250,
                'sample_window_depth':35,
                'constant':3,
                'flip':1,
                'avoid_opponent':1,
                'mult':0,
                'std_val':3.0,
                }


    # RL_params = {
    #             'explore_until':50000
    #             }
    RL_params = {}
    # best_params = A1_params
    # # A1_params = {}
    best_params = {}

    # HEAD TO HEAD BATTLE:
    analysis.trial(Bandits,trial_models,num_ban=100,num_cycles=2000,N_trials=100,parallel=False,verbose=0,verbose_summary=1,seed=878,plot=0,A1_params=A1_params,A2_params=best_params,no_opponent=False)

    # RUN THE EXTREMELY SIMPLISTIC AND BASICALLY USELESS IN THIS CONTEXT Q-LEARNING MODEL.. BUT THE FRAME WORK IS THERE TO PUT IN OPPONENTS:
    # analysis.trial_RL(Bandits,trial_models,num_ban=3,num_cycles=100,N_trials=100,parallel=False,verbose=0,verbose_summary=1,seed=878,plot=1,A1_params=A1_params,A2_params=RL_params,no_opponent=False)

    # PARAMETER SEARCH USING RANDOM-SEARCH APPROACH:
    # analysis.search(Bandits,trial_models,num_ban=100,num_cycles=2000,N_trials=100,seed=17,params=params,num_params=1000,no_opponent=False)

    # RUN SEVERAL MODELS AGAINST EACH OTHER:
    # analysis.benchmark(Bandits,benchmark_models,num_ban=100,num_cycles=2000,N_trials=100,seed=77,parallel=1)

    # PLOT A HEATMAP FROM THE BENCHMARK RESULTS:
    # analysis.heatmaps('data/2021-01-06_10:49:04.csv')
    raise

