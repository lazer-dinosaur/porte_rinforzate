from torch import nn
import numpy as np
import torch
import tqdm


class Model(nn.Module):
    def __init__(self, n_doors: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_doors * 3, 200),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.1),
                                    nn.Linear(200, n_doors),
                                    nn.Sigmoid())
        # self.distribution = nn.Sequential(nn.Linear(1024, n_doors),
        #                             nn.Softmax())
        # self.predict = nn.Sequential(nn.Linear(1024, n_doors),
        #                             nn.Sigmoid())
    
    def _build_input_(self, hero_state, enemy_state, hero_rewards):
        t = torch.tensor(np.concatenate([hero_state, enemy_state, hero_rewards]).astype(np.float32))[None, :]
        return t
    
    def forward(self, x):
        x = self.layers(x)
        return x
        # door = self.distribution(x)
        # reward = self.predict(x)
        # return door, reward
    
    def call(self, env_state, hero_rewards):
        hero_state, enemy_state = env_state
        t = self._build_input_(hero_state, enemy_state, hero_rewards)
        x = self.forward(t)
        # door, reward = self.forward(t)
        
        # return door, reward
        return x
        # return torch.argmax(x, dim=1) , torch.max(x, dim=1)


class TheDoors:
    def __init__(self, n_doors=100, decay=0.97, seed=42):
        np.random.seed(seed)
        self._set_initial_state(n_doors)
        self.decay = decay
        self.n_doors = n_doors
    
    def open_doors(self, choices):
        rewards = []
        for player_id, door in enumerate(choices):
            state = np.random.uniform(0, 1)
            reward = self.thresholds[door] >= state
            rewards.append(reward.type(torch.float))
            self.doors_states[player_id][door] += 1
        for player_id, door in enumerate(choices):
            self.thresholds[door] *= self.decay
        return rewards
    
    def get_state(self, player_id):
        players_state = self.doors_states[player_id]
        other = self.doors_states[1 - player_id]
        return players_state, other
    
    def reset(self):
        self._set_initial_state(self.n_doors)
    
    def _set_initial_state(self, n_doors):
        self.thresholds = torch.tensor(np.random.uniform(0, 1, n_doors))
        self.doors_states = np.zeros((2, n_doors))


class Player:
    def __init__(self, player_id: int, n_doors, is_random: bool, selection_function=None):
        if is_random:
            self.selection_function = lambda x, y: None
        else:
            self.selection_function = selection_function
        self.player_id = player_id
        self.n_doors = n_doors
        self._set_initial_state(n_doors)
        self.start_eps = 1.
        self.end_eps = 1.e-3
        self.eps_decay = .999
        self.is_random = is_random
    
    def choose(self, environment):
        x = self.selection_function(environment.get_state(self.player_id), self.rewards)
        sample = np.random.uniform(0, 1)
        eps_threshold = self.get_eps_threshold()
        
        if sample > eps_threshold:
            t = torch.max(x, dim=1)
            door = t.indices
        else:
            door = np.random.choice(self.n_doors)
            door = torch.tensor([[door]], dtype=torch.long)
        return door, x
    
    def get_eps_threshold(self):
        self.steps += 1
        if self.is_random:
            return 1.
        else:
            return piecewise_linear(self.steps, (0, 50, 200), (1., 1e-2, 1e-2))
            # return self.end_eps + (self.start_eps - self.end_eps) * math.exp(-1. * self.steps / self.eps_decay)
    
    def _set_initial_state(self, n_doors):
        self.rewards = np.zeros(n_doors)
        self.steps = 0
    
    def reset(self):
        self._set_initial_state(self.n_doors)


def linear_function(x, x0, x1, y0, y1):
    m = (y1 - y0) / (x1 - x0)
    b = y0 - m * x0
    return m * x + b


def piecewise_linear(step, X, Y):
    """
    Piecewise linear function.

    :param step: current step.
    :param X: list of breakpoints
    :param Y: list of values at breakpoints
    :return: value of piecewise linear function with values Y_i at step X_i
    """
    assert len(X) == len(Y)
    X = np.array(X)
    if step < X[0]:
        return Y[0]
    idx = np.where(step >= X)[0][-1]
    if idx == (len(Y) - 1):
        return Y[-1]
    else:
        return linear_function(step, X[idx], X[idx + 1], Y[idx], Y[idx + 1])

if __name__ == '__main__':
    n_doors = 10
    env = TheDoors(n_doors=n_doors)
    model = Model(n_doors)
    RL_player = Player(selection_function=model.call, player_id=0, n_doors=n_doors, is_random=False)
    random_player = Player(player_id=1, n_doors=n_doors, is_random=True)
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=0.00001, betas=(0.999, .999), weight_decay=0.01)
    n_games = 10000
    wins = 0
    diffs = []
    diff = 0
    iterator = tqdm.tqdm(range(n_games))
    for _ in iterator:
        iterator.set_description(f'{diff}')
        env.reset()
        RL_player.reset()
        random_player.reset()
        for _ in range(200):
            optimizer.zero_grad()
            rl_choice, rl_estimate = RL_player.choose(env)
            random_choice, _ = random_player.choose(env)
            rl_reward, random_reward = env.open_doors(
                (rl_choice, random_choice))  # TODO: order depends on player_id, risks to be switched around
            RL_player.rewards[rl_choice] += int(rl_reward)
            random_player.rewards[random_choice] += int(random_reward)
            log_probs = torch.log(rl_estimate)
            supervised_loss = torch.mean(torch.abs(rl_estimate * rl_reward))
            # supervised_loss = torch.mean(torch.abs(rl_estimate - env.thresholds))
            # choice_loss = torch.mean(1. - rl_reward)
            # stato, azione, stato_risultante, reward
            # loss = torch.mean(torch.tensor([1. - float(rl_reward)]))  # ISSUE: rl_reward not a tensor
            supervised_loss.backward()
            # choice_loss.backward()
            optimizer.step()
        
        random_player_tot_reward = np.sum(random_player.rewards)
        RL_player_tot_reward = np.sum(RL_player.rewards)
        diff = RL_player_tot_reward - random_player_tot_reward
        if diff > 0:
            wins += 1
        diffs.append(diff)
    print(diffs)
    print(f'wins: {wins}')
