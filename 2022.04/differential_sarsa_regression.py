'''
The cart pole problem, continuous.
Solved via differential sarsa using linear regression with tiling.
'''
import random
import gym
import numpy as np

from tile import hash_encode
from tqdm import tqdm

# Params you can't f with:

# FULL THROTTLE REVERSE: -1
# ZERO THROTTLE: 0
# FULL THROTTLE FORWARD: 1
_ACTIONS = [0, 1]

_STATE_BOUNDS = [
    [-4.8, 4.8],     # Cart Position
    [-2, 2],         # Cart Velocity         - Note: actually [-inf, inf]
    [-0.418, 0.418], # Pole Angle
    [-3, 3]]         # Pole Angular Velocity - Note: actually [-inf, inf]

# None means -inf to inf, we'll deal with that later.
# Params you can f with:

_BUCKETS_PER_FEAT = [2, 2, 4, 4]

_N_TILINGS = 8
_HASH_TABLE_SIZE = 1024
_STATE_SHAPE = (2 * _HASH_TABLE_SIZE, )


def clip(a, a_min, a_max):
    '''More efficient than np.clip cuz doesn't need to be a ufunc'''
    return min(a_max, max(a_min, a))


def state_action_to_x(s, a):
    res = np.zeros(_STATE_SHAPE)
    for i in [1, 3]:
        s[i] = clip(s[i], *_STATE_BOUNDS[i])
    # print('-'*30, [s], _STATE_BOUNDS, _BUCKETS_PER_FEAT, _N_TILINGS, _HASH_TABLE_SIZE, sep='\n', end='\n' + '-'*30 + '\n')
    hash_idx = hash_encode(
        [s],
        _STATE_BOUNDS,
        _BUCKETS_PER_FEAT,
        _N_TILINGS,
        _HASH_TABLE_SIZE).item()
    res[a * _HASH_TABLE_SIZE + hash_idx] = 1
    return res


def state_action_value(x, w):
    return w.T.dot(x)


def best_action(s, w):
    return max(
        _ACTIONS,
        key=lambda a:
        state_action_value(
            state_action_to_x(s, a),
            w))


def fit(trials=100_000, alpha=0.5, beta=0.2, epsilon=0.1):
    w = np.zeros(_STATE_SHAPE)
    action_to_not = {a: [x for x in _ACTIONS if x != a] for a in _ACTIONS}
    # R is estimate of the average reward
    R = 0
    env = gym.make('CartPole-v1')
    s, a = env.reset(), random.choice(_ACTIONS)
    x = state_action_to_x(s, a)
    cv, pv = 0, 0
    for _ in tqdm(range(trials)):
        # print(w.min(), w.max())
        if s[1] > cv:
            print(f'{cv:.3f} {pv:.3f} {R}')
            cv = s[1]
        if s[3] > pv:
            print(f'{cv:.3f} {pv:.3f} {R}')
            pv = s[3]
        sprime, r, done, _ = env.step(a)
        aprime = best_action(sprime, w)
        if random.random() < epsilon:
            aprime = random.choice(action_to_not[aprime])
        xprime = state_action_to_x(sprime, aprime)
        d = r - R + state_action_value(xprime, w) - state_action_value(x, w)
        R += beta * d
        # print(f'{alpha:05f} {d:05f} {x}')
        w += alpha * d * x
        if done:
            s = env.reset()
            a = best_action(s, w)
            if random.random() < epsilon:
                a = random.choice(action_to_not[a])
            x = state_action_to_x(s, a)
        else:
            s, a, x = sprime, aprime, xprime
    return w


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
