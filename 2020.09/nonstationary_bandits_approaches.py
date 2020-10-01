#!/usr/bin/env python

'''
Three RL approaches to a non-stationary k-bandits problem.

The accumulated reward for different approaches with
varying microparams over time is then graphed,
depicting the effects of different epsilons.
'''

__author__ = 'Sam Prokopchuk'

import numpy as np
import matplotlib.pyplot as plt


class Bandit(object):
    '''Contains all bandit properties and methods'''

    def __init__(self, mean: float, variance: float = 1., walk_prob: float = 0.1):
        self.mean = mean
        self.variance = variance
        self.walk_prob = walk_prob

    def update_params(self):
        '''
        This method updates the bandits' parameters by "walking" its mean
        under the offchance of its walk_prob probability.
        '''
        if np.random.random() < self.walk_prob:
            self.mean += np.random.normal()

    def get_reward(self):
        return np.random.normal(self.mean, self.variance)


class ActionSelectionRule(object):
    '''Base Class for action-selection rules'''

    def __init__(self):
        self.last_action_idx = None
        self.awaiting_feedback = False

    def get_action(self):
        raise NotImplementedError

    def feedback(self, reward):
        raise NotImplementedError


class EpsilonActionSelectionRule(ActionSelectionRule):
    def __init__(self, epsilon: float, num_actions: int):
        super().__init__()
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.expected_rewards = np.zeros(num_actions)

    def get_action(self):
        assert(not self.awaiting_feedback)
        if np.random.random() < self.epsilon:
            # Explore
            self.last_action_idx = np.random.randint(self.num_actions)
        else:
            # Exploit:
            self.last_action_idx = np.argmax(self.expected_rewards)
        self.action_selected_count[action_idx] += 1
        self.awaiting_feedback = True
        return action_idx


class EGreedy(EpsilonActionSelectionRule):
    '''Select object which has highest expectation'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_selected_count = np.zeros(self.num_actions)

    def feedback(self, reward):
        assert(self.awaiting_feedback)
        delta = reward - self.expected_rewards[self.last_action_idx]
        delta /= self.action_selected_count[self.last_action_idx]
        self.expected_rewards[self.last_action_idx] += delta
        self.awaiting_feedback = False


class WeightedAverageGreedy(EpsilonActionSelectionRule):
    '''Select object with highest weighted average expectation'''

    def __init__(self, alpha: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def feedback(self, reward):
        assert(self.awaiting_feedback)
        self.expected_rewards[last_action_idx] += self.alpha * \
            (reward - self.expected_rewards[self.last_action_idx])
        self.awaiting_feedback = False


class User(object):
    '''Acts as a user who uses bandits and accumulates reward over time'''

    def __init__(self, asr: ActionSelectionRule):
        self.iterations = 0
        self.reward_history = {0: 0}
        self.asr = asr

    def use(bandit: Bandit):
        self.iterations += 1
        reward = bandit.get_reward()
        total_reward = self.reward_history[self.iterations - 1] + reward
        self.reward_history[self.iterations] = total_reward


def main(num_bandits=10, min_mean=-10, max_mean=10, time_steps=1000):
    '''
    Args:
    num_bandits: Self Explanatory
    min_mean: minimum initial mean for each bandit
    max_mean: maximum initial mena for each bandit

    '''
    # Params to make things easier:
    num_bandits = 10
    min_mean = -10
    max_mean = 10

    bandit_means = np.random.uniform(min_mean, max_mean, num_bandits)
    bandits = [Bandit(mean) for mean in bandit_means]

    users = []

    epsilons = [0.01, 0.1, 0.5]
    users += [User(EGreedy(e, num_bandits)) for e in epsilons]

    for t in range(time_steps):
        exit(0)


if __name__ == '__main__':
    main()
