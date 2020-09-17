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

# Bandit 'variables':
NUM_BANDITS = 10
MIN_MEAN = -10.
MAX_MEAN = 10.
# Note: A uniform dist from MIN_MEAN to MAX_MEAN
# will be used to determine the bandits' averages.
VARIANCE = 1
WALK_PROB = 0.1
# Distribution from which that bandit will determine
# how far to walk from its current mean:
WALK_DISTRIBUTION = norm(0, 0.01)


# Different probabilities of not chosing current value with highest expectation
EPSILONS = [0.01, 0.1, 0.5]
# The weight to give the most recent reward signal:
STEP_SIZE = 0.05
TIME_STEPS = 1000


def main():
    n = NUM_BANDITS
    bandit_means = np.random.uniform(low=MIN_MEAN, high=MAX_MEAN, size=n)

    bandits = [norm(m, VARIANCE) for m in bandit_means]

    # Store the rewards over time for each epsilon in a single ndarray:
    rewards_over_time = np.zeros((len(EPSILONS), TIME_STEPS))
    # Expected values of each bandit for each EPSILON:
    expected_values = np.zeros((len(EPSILONS), n))

    for t in TIME_STEPS:
        r = np.random.random()
        for idx, e in enumerate(EPSILONS):
            if e < r:
                # Explore
                use_bandit_idx = np.random.randint(n)
            else:
                # Exploit
                use_bandit_idx = np.argmax(expected_values)

            # Take a single sample from the 'action'th rv.
            sample = bandits[use_bandit_idx].rvs()

    # for i, e in enumerate(EPSILONS):
    #     plt.plot(np.arange(TIME_STEPS), rewards_over_time[i], label=f'{chr(949)} = {e}')
    # plt.xlabel('Time Step')
    # plt.ylabel('Accumulated Reward')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
