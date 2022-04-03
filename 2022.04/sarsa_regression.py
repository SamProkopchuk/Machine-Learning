'''
The moutain car problem
Solved via sarsa linear regression
'''
import random

import numpy as np

from math import cos


# Params you can f with:

_EPSILON = 1e-10

_ACTIONS = [-1, 0, 1]

_POS_BOUNDS = (-1.2, 0.6)
_VEL_BOUNDS = (-0.7, 0.7)

_POS_BUCKETS = 8
_VEL_BUCKETS = 8

_AXIS_SHIFTS = 4

# Params you can't f with:

N_BUCKETS_PER_TILE = _POS_BUCKETS * _VEL_BUCKETS
N_TILES = 1 + (4 * _AXIS_SHIFTS)
N_BUCKETS = N_BUCKETS_PER_TILE * N_TILES
X_SIZE = len(_ACTIONS) * N_BUCKETS


def clip(a, a_min, a_max):
    '''More efficient than np.clip cuz doesn't need to be a ufunc'''
    return min(a_max, max(a_min, a))


def get_dimensional_tiling_index(x_min, x_max, axis_tiles, n_axis_tiles):
    pass


def state_action_to_x(pos, vel, a):
    '''
    Idea:
    We start at some center tile
    and stride in combinations of (-1, 1) x y directions
    such that all strides are evenly spaced out
    '''
    res = np.zeros((X_SIZE, 1))
    off = _ACTIONS.index(a) * N_BUCKETS
    pos_base_ubs = np.linspace(*_POS_BOUNDS, _POS_BUCKETS + 1)[1:]
    vel_base_ubs = np.linspace(*_VEL_BOUNDS, _VEL_BUCKETS + 1)[1:]

    i = 0
    for dx, dy in product((-1, 1), (-1, 1)):
        for axis_shift in range(N_BUCKETS):
            if axis_shift == 0 and not (dx == dy == -1):
                continue
            stride = axis_shift / (_AXIS_SHIFTS + 1)
            pos_idx = np.searchsorted(pos_ubs + dx * stride, pos)
            vel_idx = np.searchsorted(vel_ubs + dy * stride, vel)
            res[off + i * N_BUCKETS_PER_TILE + pos_idx + vel_idx * _POS_BUCKETS] = 1
            i += 1
    return res


def next_state(s, a):
    next_pos = clip(curr_pos + curr_vel, *_POS_BOUNDS)
    next_vel = 0 if next_pos = _POS_BOUNDS[0] else clip(curr_vel + 0.001 * a - 0.0025 * cos(3 * curr_pos), *_VEL_BOUNDS)
    return next_pos, next_vel


def environment(s, a):
    return -1, next_state(s, a)


def state_action_value(x, w):
    return w.T.dot(x)


def best_action(pos, vel, w):
    x = state_action_to_x(pos, vel, a)
    w = state_action_value(x, w)
    return max(_ACTIONS, key=lambda a: state_action_value(x, w))


def is_terminal_state(s):
    return s[0] == _POS_BOUNDS[1]


def main(trials=10000, lmda=1, alpha=1e-2, epsilon=1e-1):
    w = np.zeros((N_BUCKETS, 1))
    action_to_not = {a: [x for x in _ACTIONS if x != a] for a in _ACTIONS}

    for _ in range(trials):
        s = [random.uniform(*_POS_BOUNDS), random.uniform(*_VEL_BOUNDS)]
        a = best_action(*s, w)
        if random.random() < epsilon:
            a = random.choice(action_to_not[a])
        if not is_terminal_state(s): # Basically 0% chance of being false but whatever
            while True :
                r, sprime = environment(s, a)
                if is_terminal_state(sprime):
                    w += alpha * (...)
                    break
                aprime = best_action(*sprime, w)
                w += alpha * (...)
                s, a = sprime, aprime


if __name__ == '__main__':
    main()
