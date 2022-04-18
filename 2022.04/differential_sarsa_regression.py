'''
The cart pole problem, continuous.
Solved via sarsa using neural network state-action value approximation.
'''
import random
import gym
import numpy as np
import torch

from tqdm import tqdm

# Params you can't f with:

# ZERO THROTTLE: 0
# FULL THROTTLE FORWARD: 1
_NUM_ACTIONS = 2
_STATE_SHAPE = (4, )

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(_STATE_SHAPE[0] << 1, 64),
            torch.nn.Linear(64, 1)])
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.leaky_relu(x)
        return x

    def __call__(self, s, a):
        sh = _STATE_SHAPE[0]
        x = np.zeros(sh<<1)
        x[a * sh: (a+1) * sh] = s
        x = torch.as_tensor(x, dtype=torch.float32)
        return super().__call__(x)


def best_action(s, nn):
    return max(range(2), key=lambda a: nn(s, a))


def fit(trials=100000, epsilon=0.1, alpha=0.2, lmbda=0.95, lr=0.0001):
    # R is estimate of the average reward
    R = 0
    env = gym.make('CartPole-v1')
    nn = NN().float()
    optimizer = torch.optim.Adam(nn.parameters(), lr=lr)
    loss = torch.nn.MSELoss()
    ep_lens = [0]
    for i in tqdm(range(trials)):
        s = env.reset()
        a = best_action(s, nn)
        if random.random() < epsilon:
            a = 1 - a
        while True:
            sprime, r, is_episode_done, _ = env.step(a)
            if is_episode_done:
                r = torch.tensor([-1], dtype=torch.float32)
                optimizer.zero_grad()
                loss(r, nn(s, a)).backward()
                optimizer.step()
                break
            else:
                r = torch.tensor([0], dtype=torch.float32)
                aprime = best_action(s, nn)
                if random.random() < epsilon:
                    aprime = 1 - aprime
                with torch.no_grad():
                    pred = lmbda * nn(sprime, aprime).item() + r
                optimizer.zero_grad()
                y = nn(s, a)
                # print(s, a, y)
                l = loss(pred, y)
                l.backward()
                optimizer.step()
            ep_lens[-1] += 1
        if i % 100 == 0:
            print(f'Avg episode length: {sum(ep_lens) / len(ep_lens)}')
            ep_lens = [0]
            print(nn(s, 0), nn(s, 1))
            # for param in nn.parameters():
                # print(param.data)
        else:
            ep_lens.append(0)
    return nn


def render_episodes(w, nepisodes=10):
    env = gym.make('CartPole-v1')
    for _ in range(nepisodes):
        s = env.reset()
        done = False
        while not done:
            a = best_action(s, w)
            s, r, done, d = env.step(a)
            env.render()
    env.close()


def main():
    w = fit()
    render_episodes(w)


if __name__ == '__main__':
    main()
