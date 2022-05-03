'''
The cart pole problem, continuous.
Solved via Actor-Critic Algorithm with Elegibility Traces (continuing)
The policy pi uses a neural-network function H. Thereafter applying softmax.
The state-value v function is simple linear regression using tiling.
'''
import random
import gym
import numpy as np
import torch

from tqdm import tqdm
from tile import tile_encode

_NUM_ACTIONS = 2
_STATE_SHAPE = (4, )


_BUCKETS_PER_FEAT = [
    2,
    3,
    4,
    4]

_FEAT_RANGES = [
    [-4.8, 4.8],
    [-3, 3],
    [-0.418, 0.418],
    [-3, 3]
N_TILES = 8



def clip_state(s, feat_ranges):
    for i, fr in enumerate(feat_ranges):
        s[i] = min(fr[1], max(fr[0], s[i]))
    return s


class H(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(_STATE_SHAPE[0] * _NUM_ACTIONS, 32),
            torch.nn.Linear(32, _NUM_ACTIONS)])
        self.activation_functions = torch.nn.ModuleList([
            torch.nn.LeakyReLU(),
            torch.nn.LeakyReLU()])
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            torch.nn.init.kaiming_uniform_(layer.weight)

    def forward(self, x):
        for layer, activation_function in zip(self.layers, self.activation_functions):
            x = activation_function(layer(x))
        return x

    def __call__(self, s, a):
        x = np.zeros(_STATE_SHAPE[0] * _NUM_ACTIONS)
        x[a * _STATE_SHAPE[0]: (a+1) * _STATE_SHAPE[0]] = s
        x = torch.as_tensor(s, dtype=torch.float32)
        res = super().__call__(x)

class Policy:
    def __init__(self, h_func):
        self.h = h_func

    def __call__(self, s):
        X = torch.as_tensor([[i] for i in range(_NUM_ACTIONS)], dtype=torch.float32)
        print(h(X))
        exit(0)

    def update_params(self, d, gradient):
        self.update_elegibility_trace_vector(gradient)


class V:
    def __init__(self, buckets_per_feat, n_tiles, lmbda, alpha):
        self.buckets_per_feat = buckets_per_feat
        self.n_tiles = n_tiles
        self.w = np.zeros((self.n_tiles * np.product(self.buckets_per_feat), ))
        self.lmbda = lmbda
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.z = 0

    def __call__(self, s):
        s = self.clip_state(s)
        te = tile_encode([s], _FEAT_RANGES, self.buckets_per_feat, self.n_tiles)
        return self.w.dot(te

    def update_params(self, d, gradient):
        self.update_elegibility_trace_vector(gradient)
        self.w += self.alpha * d * self.z

    def update_elegibility_trace_vector(self, gradient):
        self.z = self.lmbda * self.z + gradient


def fit(trials=50000, pi, v, alpha):
    '''
    Run the algorithm with:
        Policy pi
        State-value function v
    alpha: update rate for R
    '''
    # R is estimate of the average reward
    R = 0
    env = gym.make('CartPole-v1')
    env._max_episode_steps = np.inf
    s = env.reset()
    for i in tqdm(range(trials)):
        a = pi(s)
        sprime, r, is_episode_done, _ = env.step(a)
        r = -1 if is_episode_done else 0
        d = r - R + v(sprime) - v(s)
        R += alpha * d
        v.update_params(d, s)
                          with torch.no_grad():
            pred = r + nn(sprime, aprime).item()
        y = nn(s, a)
        d = (r - R + pred - y).item()
        # l = loss(pred, R + nn(s, a))
        # optimizer.zero_grad()
        # l.backward()
        # optimizer.step()
        y.backward()
        with torch.no_grad():
            for param in nn.parameters():
                b4 = param.data.detach()
                param.data = param.data + alpha * d * param.grad.detach()
                if torch.isnan(param).all():
                    print(b4)
                    print(alpha * d * param.grad)
                    print(' <| |> ')
                    print(param)
                    exit(0)
        R = R + beta * d
        if is_episode_done:
            s = env.reset()
            a = best_action(s, nn)
            if random.random() < epsilon:
                a = 1 - a
            if len(ep_lens) > 10:
                print(f'Avg episode length: {sum(ep_lens) / len(ep_lens)}')
                ep_lens = [0]
                with torch.no_grad():
                    print(nn(s, 0), nn(s, 1))
                    for param in nn.parameters():
                        print(param)
                i = 0
            else:
                ep_lens.append(0)
        else:
            s, a = sprime, aprime
            ep_lens[-1] += 1
    return nn


def render_episodes(w, nepisodes=3):
    env = gym.make('CartPole-v1')
    env._max_episode_steps = np.inf
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
