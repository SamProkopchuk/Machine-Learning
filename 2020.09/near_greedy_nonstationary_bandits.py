#!/usr/bin/env python

'''
An RL approach to a non-stationary k-bandits problem.
This example uses the near greedy approach with
a weighted average value-function.

The accumulated reward over time is then graphed,
depicting the effects of different epsilons.
'''

__author__ = 'Sam Prokopchuk'

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm

# Feel free to (conservatively) tinker with these constants:

# Bandit 'variables':
NUM_BANDITS = 10
MIN_MEAN = -10.
MAX_MEAN = 10.
# Note: A uniform dist from MIN_MEAN to MAX_MEAN
# will be used to determine the bandits' averages.
VARIANCE = 1
# Probability of each bandits mean changing each round:
WALK_PROB = 0.2
# Distribution from which that bandit will determine
# how far to walk from its current mean.
# Should be an instance of scipy.stats.rv_continuous:
WALK_DISTRIBUTION = norm(0, 0.4)

# Different probabilities of exploring every iteration
# (Instead of exploiting)
# (chosing current value with highest expectation)
EPSILONS = [0.01, 0.1, 0.5, 0.9, 0.99]
# The weight to give the most recent reward signal:
STEP_SIZE = 0.2
TIME_STEPS = 1000


def use_bandit(mean, variance=VARIANCE):
    return norm(mean, variance).rvs()


def main():
    n = NUM_BANDITS
    bandit_means = np.random.uniform(low=MIN_MEAN, high=MAX_MEAN, size=n)

    # Store the rewards over time for each epsilon in a single ndarray:
    reward_history = np.zeros((len(EPSILONS), TIME_STEPS))
    # Expected values of each bandit for each EPSILON:
    expected_values = np.zeros((len(EPSILONS), n))

    for t in range(TIME_STEPS):
        r = np.random.random()
        for e_idx, e in enumerate(EPSILONS):
            if r < e:
                # Explore
                bandit_idx = np.random.randint(n)
            else:
                # Exploit
                bandit_idx = np.argmax(expected_values[e_idx])

            sample = use_bandit(bandit_means[bandit_idx])
            # Simple weighted average:
            expected_values[e_idx][bandit_idx] += (
                STEP_SIZE * (sample - expected_values[e_idx][bandit_idx]))

            if t != 0:
                reward_history[e_idx][t] = reward_history[e_idx][t-1] + sample

        update_mean_idx = np.argwhere(np.random.random(n) < WALK_PROB)
        # No obvious way (to me) of vectorizing...
        for idx in update_mean_idx:
            bandit_means[idx] += WALK_DISTRIBUTION.rvs()

    for i, e in enumerate(EPSILONS):
        plt.plot(np.arange(TIME_STEPS), reward_history[i],
                 label=f'{chr(949)} = {e}')

    plt.xlabel('Time Step')
    plt.ylabel('Accumulated Reward')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
