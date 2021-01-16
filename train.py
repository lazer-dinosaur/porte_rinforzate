from pathlib import Path

from torch import nn
from torch import optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml


class PolicyEstimator(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_units):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_units = hidden_units
        # # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.n_outputs),
            nn.Sigmoid(),
            nn.Softmax(dim=-1)).to(device)
    
    def forward(self, state):
        x = torch.FloatTensor(state).to(device)
        return self.network(x)


class TheDoors:
    def __init__(self, n_doors, turns, decay, seed=42):
        np.random.seed(seed)
        self._set_initial_state(n_doors)
        self.decay = decay
        self.n_doors = n_doors
        self.turns = turns
    
    def step(self, choices):
        rewards = []
        for player_id, door in enumerate(choices):
            state = np.random.uniform(0, 1)
            reward = self.thresholds[door] >= state
            rewards.append(float(reward))
            self.doors_states[player_id][door] += 1
            if reward:
                self.players_rewards[player_id][door] += 1
        for player_id, door in enumerate(choices):
            self.thresholds[door] *= self.decay
        
        self.steps += 1
        done = self.steps >= self.turns
        return rewards, done
    
    def get_state(self, player_id):
        player_state = self.doors_states[player_id]
        other = self.doors_states[1 - player_id]
        return np.concatenate([player_state, other, self.players_rewards[player_id]])
    
    def reset(self):
        self._set_initial_state(self.n_doors)
    
    def _set_initial_state(self, n_doors):
        self.thresholds = torch.tensor(np.random.uniform(0, 1, n_doors))
        self.doors_states = np.zeros((2, n_doors))
        self.players_rewards = np.zeros((2, n_doors))
        self.steps = 0


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma ** i * rewards[i]
                  for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]  # TODO: perche' invertito?
    return r - r.mean()
    # return r


def normalize(x):
    x = (x + abs(np.min(x))) / (np.min(x) + np.max(x))
    return x / np.sum(x)


def reinforce(env, policy_estimator, optimizer, num_episodes=2000, batch_size=10, gamma=0.99,
              epsilon_decay=.99, starting_ep=0, starting_epsilon=1.):  # Set up lists to hold results
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1
    
    epsilon = starting_epsilon
    action_space = np.arange(env.n_doors)
    ep = starting_ep
    wins = 0
    losses = 0
    while ep < num_episodes:
        ep += 1
        env.reset()
        states = []
        rewards = []
        actions = []
        rewards_random = []
        if ep % 1000 == 0:
            torch.save({'network': policy_estimator.network.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'episode': ep,
                        'epsilon': epsilon, },
                       f'logs/{run_name}/latest.pt')
        done = False
        epsilon *= epsilon_decay  # TODO: epsilon decade ad episodio o dentro ogni episodio?
        epsilon = max(1e-3, epsilon)
        writer.add_scalar('Meta/epsilon', epsilon, ep)
        while not done:
            # Get actions and convert to numpy array
            s_0 = env.get_state(player_id=0)
            action_probs = policy_estimator.forward(s_0).cpu().detach().numpy()
            if torch.rand(1) > epsilon:
                action = np.random.choice(action_space, p=action_probs)
            else:
                action = np.random.choice(action_space)
            random_action = np.random.choice(action_space)
            r, done = env.step((action, random_action))
            
            states.append(s_0)
            actions.append(action)
            rewards.append(r[0])
            rewards_random.append(r[1])
            # If done, batch data
            if done:  # fine episodio
                sum_rewards = np.sum(rewards)
                sum_rewards_random = np.sum(rewards_random)
                wins += int(sum_rewards > sum_rewards_random)  # TODO: add additional reward for wins
                losses += int(sum_rewards < sum_rewards_random)
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                writer.add_scalar('Wins-Losses', wins - losses, ep)
                print(f"\rEp: {ep + 1}", end="")
                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(batch_rewards).to(device)
                    # Actions are used as indices, must be LongTensor
                    action_tensor = torch.LongTensor(batch_actions).to(device)
                    
                    # Calculate loss
                    logprob = torch.log(policy_estimator.forward(state_tensor))
                    selected_logprobs = reward_tensor * torch.gather(logprob, 1, action_tensor[None, :]).squeeze()
                    loss = -selected_logprobs.mean()
                    writer.add_scalar('Loss', loss, ep)
                    
                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()
                    
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1


if __name__ == '__main__':
    # get hyperparameters
    config_name = '3'
    version = '10'
    run_name = f'{config_name}.{version}'
    with open(f'configs/{config_name}.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        print(data)
    n_doors = data['n_doors']
    turns = data['turns']
    hidden_units = data['hidden_units']
    lr = data['lr']
    batch_size = data['batch_size']
    num_episodes = data['num_episodes']
    door_decay = data['door_decay']
    epsilon_decay = data['epsilon_decay']

    # CPU vs GPU
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Computing on {device} device')

    # Define environment
    env = TheDoors(n_doors=n_doors, turns=turns, decay=door_decay)

    # Define network
    policy_estimator = PolicyEstimator(n_inputs=3 * n_doors,
                                       n_outputs=n_doors,
                                       hidden_units=hidden_units).to(device)

    # Define optimizer
    optimizer = optim.AdamW(policy_estimator.parameters(), lr=lr, )

    log_dir = f'logs/{run_name}'
    ckpt_path = Path(f'{log_dir}/latest.pt')
    if ckpt_path.exists():
        checkpoint = torch.load(f'{log_dir}/latest.pt', map_location=device)
        policy_estimator.network.load_state_dict(checkpoint['network'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        starting_step = checkpoint['episode']
        starting_epsilon = checkpoint['epsilon']
    else:
        starting_step = -1
        starting_epsilon = 1.
    
    writer = SummaryWriter(log_dir=log_dir)
    reinforce(env, policy_estimator,
                    optimizer=optimizer,
                    num_episodes=num_episodes,
                    batch_size=batch_size,
                    gamma=0.99,
                    epsilon_decay=epsilon_decay,
                    starting_ep=starting_step,
                    starting_epsilon=starting_epsilon)
