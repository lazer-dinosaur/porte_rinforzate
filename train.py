from torch import nn
from torch import optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


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
            nn.Softmax(dim=-1)).to(device)
        
        # self.network = nn.Sequential(nn.Linear(n_inputs, 200),
        #                              nn.ReLU(),
        #                              # nn.Dropout(p=0.1),
        #                              nn.Linear(200, n_outputs),
        #                              nn.Softmax(dim=-1))
        #                              # nn.Sigmoid())
    
    def forward(self, state):
        action_probs = self.network(torch.FloatTensor(state).to(device))
        return action_probs


class TheDoors:
    def __init__(self, n_doors, turns, decay=0.97, seed=42):
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
            # rewards.append(reward.type(torch.float))
            rewards.append(float(reward))
            self.doors_states[player_id][door] += 1
        for player_id, door in enumerate(choices):
            self.thresholds[door] *= self.decay
        self.steps += 1
        done = self.steps >= self.turns
        return rewards, done
    
    def get_state(self, player_id):
        player_state = self.doors_states[player_id]
        other = self.doors_states[1 - player_id]
        return np.concatenate([player_state, other, self.thresholds])
    
    def reset(self):
        self._set_initial_state(self.n_doors)
    
    def _set_initial_state(self, n_doors):
        self.thresholds = torch.tensor(np.random.uniform(0, 1, n_doors))
        self.doors_states = np.zeros((2, n_doors))
        self.steps = 0


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma ** i * rewards[i] for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def normalize(x):
    x = (x + abs(np.min(x))) / (np.min(x) + np.max(x))
    return x / np.sum(x)


def reinforce(env, policy_estimator, num_episodes=2000, batch_size=10, gamma=0.99):  # Set up lists to hold results
    total_rewards = []
    total_rewards_random = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    tot_diffs = []
    batch_counter = 1
    
    # Define optimizer
    optimizer = optim.AdamW(policy_estimator.network.parameters(), lr=0.001)
    # optimizer = torch.optim.AdamW(list(policy_estimator.network.parameters()), lr=0.00001, betas=(0.999, .999), weight_decay=0.01)
    action_space = np.arange(env.n_doors)
    ep = 0
    global_step = 0
    wins = 0
    epsilon = 1.
    while ep < num_episodes:
        # s_0 = env.reset()
        env.reset()
        states = []
        rewards = []
        actions = []
        rewards_random = []
        done = False
        while not done:
            global_step += 1
            # Get actions and convert to numpy array
            s_0 = env.get_state(player_id=0)
            action_probs = policy_estimator.forward(s_0).cpu().detach().numpy()
            deviation = np.random.normal(0, epsilon, n_doors)
            action_probs_dev = action_probs + deviation
            action_probs_dev = normalize(action_probs_dev)
            epsilon *= .99999
            epsilon = max(1e-3, epsilon)
            writer.add_scalar('Meta/epsilon', epsilon, global_step)
            action = np.random.choice(action_space, p=action_probs_dev)
            random_action = np.random.choice(action_space)
            # rl_choice, rl_estimate = policy_estimator.choose(env)
            # s_1, r, done, _ = env.step(action)
            r, done = env.step((action, random_action))
            
            states.append(s_0)
            rewards.append(r[0])
            writer.add_scalar('Reward/rl_agent', r[0], global_step)
            writer.add_scalar('Reward/random', r[1], global_step)
            
            rewards_random.append(r[1])
            actions.append(action)
            # s_0 = s_1
            
            # If done, batch data
            if done:  # fine episodio
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))
                total_rewards_random.append(np.sum(rewards_random))
                wins += int(np.sum(rewards) > np.sum(rewards_random))
                writer.add_scalar('Wins', wins, global_step)
                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(batch_rewards).to(device)
                    # Actions are used as indices, must be
                    # LongTensor
                    action_tensor = torch.LongTensor(batch_actions).to(device)
                    
                    # Calculate loss
                    logprob = torch.log(policy_estimator.forward(state_tensor))
                    selected_logprobs = reward_tensor * torch.gather(logprob, 1, action_tensor[None, :]).squeeze()
                    loss = -selected_logprobs.mean()
                    writer.add_scalar('Loss', loss, global_step)
                    
                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()
                    
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1
                
                # avg_rewards = np.mean(total_rewards[-100:])
                # avg_rewards_random = np.mean(total_rewards_random[-100:])
                # Print running average
                sort_probs = np.sort(action_probs)[::-1]
                diff = sort_probs[:-1][:4] - sort_probs[1:][4]
                tot_diffs.append(diff)
                for i, d in enumerate(diff):
                    writer.add_scalar(f'Diff/{i}', d, global_step)
                print(f"\rEp: {ep + 1}", end="")
                # print(f"\rEp: {ep + 1} Average of last 100: {avg_rewards} vs {avg_rewards_random} | {epsilon} | {diff}", end="")
                ep += 1
    
    return total_rewards, wins, tot_diffs, total_rewards, total_rewards_random


if __name__ == '__main__':
    n_doors = 10
    turns = 100
    hidden_units = 1024
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Computing on {device} device')
    
    env = TheDoors(n_doors=n_doors, turns=turns)
    policy_estimator = PolicyEstimator(n_inputs=3 * n_doors, n_outputs=n_doors, hidden_units=hidden_units).to(device)
    writer = SummaryWriter(log_dir=f'logs/IU{hidden_units}')
    out = reinforce(env, policy_estimator, num_episodes=10000, batch_size=10, gamma=0.99)
