'''
sarsa_sav_function_approximation

=>

sarsa state-action value function approximation

'''
from math import cos
import numpy as np

_EPSILON = 1e-10

_ACTIONS = {-1, 0, 1}

_POS_BOUNDS = (-1.2, 0.6)
_VEL_BOUNDS = (-0.7, 0.7)

_POS_BUCKETS = 8
_VEL_BUCKETS = 8

_AXIS_SHIFTS = 4

######################

N_BUCKETS_PER_TILE = _POS_BUCKETS * _VEL_BUCKETS
N_TILES = 1 + (4 * _AXIS_SHIFTS)
N_BUCKETS = N_BUCKETS_PER_TILE * N_TILES


def clip(a, a_min, a_max):
    '''More efficient than np.clip cuz doesn't need to be a ufunc'''
    return min(a_max, max(a_min, a))


def state_to_X(pos, vel):
    '''
    Idea:
    We start at some center tile
    and stride in combinations of (-1, 1) x y directions
    such that all strides are evenly spaced out
    '''
    res = np.zeros(N_BUCKETS)
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
            res[i * N_BUCKETS_PER_TILE + pos_idx + vel_idx * _POS_BUCKETS] = 1
            i += 1
    return res


def next_state(curr_pos, curr_vel, a):
    return clip(curr_pos + curr_vel, _POS_BOUNDS)


def next_velocity(curr_pos, curr_vel, a):
    return clip(curr_vel + 0.001 * a - 0.0025 *
                cos(3 * curr_pos), *_VEL_BOUNDS)


def main():
    pass


if __name__ == '__main__':
    main()
