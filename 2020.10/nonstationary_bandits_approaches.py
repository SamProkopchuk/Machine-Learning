#!/usr/bin/env python

'''
Approaches to a non-stationary k-bandits problem.

Users (Objects that interact with the bandits):
 - Users choose bandits by a given action-selection function.
 - They accumulate reward over time by using bandits.

The Environment:

A number of bandits is given.
Each bandit will provide a reward upon usage.
The reward is taken from a gaussian distribution with:
 - Variance of 1. (Can be changed if desired)
 - Individually chosen means randomly taken from the same uniform range.

After each round (when each user has used a bandit),
the bandits' update_params() method should be called.
This will cause the bandits' means to "walk" will probability walk_prob.
The amount walked is determined by np.random.normal();
(aka the standard normal distribution).
'''

__author__ = 'Sam Prokopchuk'

import numpy as np
import matplotlib.pyplot as plt

from itertools import product


class Bandit(object):
    '''Contains all bandit properties and methods'''

    def __init__(self, mean: float, variance: float = 1.,
                 walk_prob: float = 0.1):
        self.mean = mean
        self.variance = variance
        self.walk_prob = walk_prob

    def update_params(self):
        '''
        This method updates the bandits' parameters by 'walking' its mean
        under the offchance of its walk_prob probability.
        '''
        if np.random.random() < self.walk_prob:
            self.mean += np.random.normal()

    def get_reward(self):
        return np.random.normal(self.mean, self.variance)

    def __repr__(self):
        return (f'{self.__class__.__name__} with' +
                f'mean {self.mean} and variance {self.variance}')


class ActionSelectionRule(object):
    '''Base class for action-selection rules'''

    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.action_idx = None
        self.awaiting_feedback = False
        self.action_selected_count = np.zeros(num_actions)
        self.action_value_estimates = np.zeros(num_actions)

    def get_action(self):
        raise NotImplementedError

    def feedback(self, reward):
        raise NotImplementedError


class AverageActionValueRule(ActionSelectionRule):
    '''Action values are determined by their average reward'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_selected_count = np.zeros(self.num_actions)

    def feedback(self, reward):
        assert(self.awaiting_feedback)
        delta = reward - self.action_value_estimates[self.action_idx]
        delta /= self.action_selected_count[self.action_idx]
        self.action_value_estimates[self.action_idx] += delta
        self.awaiting_feedback = False


class WeightedAverageActionValueRule(ActionSelectionRule):
    '''Action values are determined by a weighted average of their reward'''

    def __init__(self, alpha: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def feedback(self, reward):
        assert(self.awaiting_feedback)
        self.action_value_estimates[self.action_idx] += self.alpha * \
            (reward - self.action_value_estimates[self.action_idx])
        self.awaiting_feedback = False


class EpsilonActionSelectionRule(ActionSelectionRule):
    '''Base class for approaches that '''

    def __init__(self, epsilon: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def get_action(self):
        assert(not self.awaiting_feedback)
        if np.random.random() < self.epsilon:
            # Explore
            self.action_idx = np.random.randint(self.num_actions)
        else:
            # Exploit:
            self.action_idx = np.argmax(self.action_value_estimates)
        self.action_selected_count[self.action_idx] += 1
        self.awaiting_feedback = True
        return self.action_idx


class AverageEGreedy(EpsilonActionSelectionRule, AverageActionValueRule):
    '''
    Select object with highest average expectation
    Occasionally explore with some epsilon probability.
    '''

    def __repr__(self):
        return (f'{self.__class__.__name__} with ' +
                f'epsilon {self.epsilon}')


class WeightedAverageEGreedy(
        WeightedAverageActionValueRule, EpsilonActionSelectionRule):
    '''
    Select object with highest weighted average expectation
    Occasionally explore with some epsilon probability.
    '''

    def __repr__(self):
        return (f'{self.__class__.__name__} with ' +
                f'step size {self.alpha} and epsilon {self.epsilon}')


class UCB(ActionSelectionRule):
    '''Base class for Upper-Confidence-Bound action selection approaches.'''
    EPSILON = 1e-6

    def __init__(self, c, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def get_action(self):
        assert(not self.awaiting_feedback)
        t = self.action_selected_count.sum()
        self.action_idx = np.argmax(
            self.action_value_estimates +
            self.c * np.sqrt(np.log(t) / (
                self.action_selected_count + UCB.EPSILON)))
        self.action_selected_count[self.action_idx] += 1
        self.awaiting_feedback = True
        return self.action_idx


class AverageUCB(UCB, AverageActionValueRule):
    '''
    Upper-Confidence-Bonud Action Selection
    Action-value estimates are aquired by averaging previous rewards.
    '''

    def __repr__(self):
        return f'{self.__class__.__name__} with explore constant {self.c}'


class WeightedAverageUCB(UCB, WeightedAverageActionValueRule):
    '''
    Upper-Confidence-Bonud Action Selection
    Action-value estimates are aquired using a weighted average.
    '''

    def __repr__(self):
        return (f'{self.__class__.__name__} with ' +
                f'explore constant {self.c} and step size {self.alpha}')


class User(object):
    '''Acts as a user who uses bandits and accumulates reward over time'''

    def __init__(self, asr: ActionSelectionRule):
        self.iterations = 0
        self.reward_history = {0: 0}
        self.total_reward = 0
        self.asr = asr

    def select_action(self, bandits):
        bandit_idx = self.asr.get_action()
        self.iterations += 1
        reward = bandits[bandit_idx].get_reward()
        self.total_reward += reward
        self.reward_history[self.iterations] = self.total_reward
        self.asr.feedback(reward)

    def __repr__(self):
        return (('User with action selection rule:') +
                (f'\n{self.asr}\nTotal reward: {self.total_reward}'))


def main(num_bandits=10, min_mean=-10, max_mean=10, time_steps=1000):
    '''
    Args:
    num_bandits: Self Explanatory
    min_mean: minimum initial mean for each bandit
    max_mean: maximum initial mena for each bandit

    '''
    bandit_means = np.random.uniform(min_mean, max_mean, num_bandits)
    bandits = [Bandit(mean) for mean in bandit_means]

    epsilons = [0.01, 0.1, 0.5]
    step_sizes = [0.1, 0.2, 0.5]
    ess_product = list(product(epsilons, step_sizes))

    users = []
    eg_users = [User(AverageEGreedy(e, num_bandits)) for e in epsilons]
    wa_users = [User(WeightedAverageEGreedy(0.2, e, num_bandits))
                for e in epsilons]
    ucb_users = [User(AverageUCB(ss, num_bandits)) for ss in step_sizes]
    waucb_users = [User(WeightedAverageUCB(ss, e, num_bandits))
                   for ss, e in ess_product]

    users = eg_users + wa_users + ucb_users + waucb_users

    for t in range(time_steps):
        for user in users:
            user.select_action(bandits)
        for bandit in bandits:
            bandit.update_params()
    for user in users:
        print(user)


if __name__ == '__main__':
    main()
