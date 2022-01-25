'''
Tabular Dyna-Q

Gridworld where going to terminal state -> reward of 1
going to any other state -> reward of 0

Actions:

     ^
     |
     1
<-2     0->
     3
     |
     v

#######XG
##X####X#
S#X####X#
##X######
#####X###
#########
'''

# Use random module since it's like faster
# than numpy.random when vectorization isn't required
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
# Efficiency ez clap:
from bisect import bisect_left, insort

# Exclude the terminal state 8 and walls:
_STATES = list(set(range(6 * 9)) - {7, 8, 16, 25, 11, 20, 29, 41})
_ACTIONS = [*range(4)]


def environment(s, a) -> int:
    # Imagine making a readable function WeirdChamp
    if a == 0:
        return (
            s if s %
            9 == 8 or s in {
                10, 19, 28, 6, 15, 24, 40} else s + 1), 0
    elif a == 1:
        return (s if s < 9 or s in {34, 38, 50}
                else s - 9), (1 if s == 17 else 0)
    elif a == 2:
        return (
            s if s %
            9 == 0 or s in {
                8, 12, 17, 21, 26, 30, 42} else s - 1), 0
    else:  # a == 3
        return (s if s > 44 or s in {2, 32} else s + 9), 0


# def visualize_policy(pi):
#     state_to_action = np.zeros((2, 10, 10))
#     for b, p, d in product(range(2), range(12, 22), range(1, 11)):
#         state_to_action[b][p - 12][d - 1] = int(pi[bool(b), p, d])
#     for b in range(2):
#         plt.imshow(state_to_action[b])
#         plt.show()


def sorted_list_contains(l, x):
    i = bisect_left(l, x)
    return i != len(l) and l[i] == x


def insort_if_not_contains(l, x):
    if not sorted_list_contains(l, x):
        insort(l, x)


def main(trials=1000, lmbda=1, alpha=1e-1, epsilon=2e-1, n=50):
    action_to_not = {a: [x for x in _ACTIONS if x != a] for a in _ACTIONS}
    action_values = {(s, a): 1 for s, a in product(_STATES, _ACTIONS)} | {(8, a): 1 for a in _ACTIONS}
    model = {(s, a): None for s, a in product(_STATES, _ACTIONS)}
    seen_s = []
    seen_sa = []

    for _ in tqdm(range(trials)):
        s = random.choice(_STATES)
        a = max(_ACTIONS, key=lambda x: action_values[s, x])
        if random.random() < epsilon:
            a = random.choice(action_to_not[a])

        sprime, r = environment(s, a)
        action_values[s,
                      a] += alpha * (r + lmbda * max(action_values[sprime,
                                                                   _a] for _a in _ACTIONS) - action_values[s,
                                                                                                           a])
        model[s, a] = (sprime, r)
        insort_if_not_contains(seen_s, s)
        insort_if_not_contains(seen_sa, (s, a))

        for __ in range(n):
            s = random.choice(seen_s)
            a = random.choice([a for a in _ACTIONS if model[s, a] is not None])

            r, sprime = model[s, a]
            action_values[s,
                          a] += alpha * (r + lmbda * max(action_values[sprime,
                                                                       _a] for _a in _ACTIONS) - action_values[s,
                                                                                                               a])


if __name__ == '__main__':
    main()
