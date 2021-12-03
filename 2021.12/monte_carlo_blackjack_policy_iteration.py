'''
Monte carlo policy iteration for blackjack

Dealer:
 - While sum(cards) < 17: hit

Agent's policy:
 - If sum(cards) < 20: hit
   - Otherwise, stand.

States:
 - Whether the player has a usable ace (False, True)
 - The sum of the player's cards (12, 21)
  - This is because if sum(cards) < 12 then there's
    0 risk of hitting so we say the agent will hit.
 - The card the dealer shows (1 (Ace), 10)

Actions:
 - Stand: False
 - Hit: True
'''

import numpy as np
import matplotlib.pyplot as plt
from random import randint, choice
from itertools import product
from functools import lru_cache

_STATES = {(b, p, d) for b, p, d in product(
        (False, True),
        range(12, 22),
        range(1, 11))}
_ACTIONS = (False, True)

class Policy(object):
    def __init__(self, arg):
        self.policy = {s: choice(_ACTIONS) for s in _STATES}

    def update_policy(s, a):
        self.policy[s] = a

    def __call__(self, s):
        return self.policy[s]


def card() -> int:
    return np.clip(randint(1, 14), 1, 10)


def has_usable_ace(cards) -> bool:
    return 1 in cards and sum(cards) + 10 <= 21

@lru_cache(maxsize=1)
def cardsum(cards) -> int:
    return sum(cards) + 10 * has_usable_ace(cards)

@lru_cache(maxsize=1)
def cards_to_state(player_cards, dealer_cards) -> tuple:
    return (
        has_usable_ace(player_cards),
        cardsum(player_cards),
        dealer_cards[0])

def initial_state_to_cards(s) -> tuple:
    # Generate player and dealer cards that would produce an initial state s
    player_cards = (1, s[1]-11) if s[0] else (s[1] // 2, (s[1] + 1) // 2)
    dealer_cards = (s[2], card())
    return player_cards, dealer_cards


def generate_episode(s, a, pi) -> int:
    '''
    Generate an episode starting with state s and action a
    '''
    player_cards, dealer_cards = initial_state_to_cards(s)
    res = []
    # sar = "state" "action" "reward"
    sar = [s, a]
    while sar[1]:
        player_cards.append(card())
        sar.append(-1 if cardsum(player_cards) > 21 else 0)
        res.append(sar)
        if cardsum(player_cards) > 21:
            return res
        sar = [cards_to_state(player_cards, dealer_cards), Policy(cards_to_state(player_cards, dealer_cards))]
    while cardsum(dealer_cards) < 17:
        dealer_cards.append(card())
    if cardsum(dealer_cards) > 21:
        sar.append(1)
    else:
        ps, ds = cardsum(player_cards), cardsum(dealer_cards)
        sar.append(-1 if ps < ds else (0 if ps == ds else 1))
    res.append(sar)
    return res

# def visualize_action_values(action_values):
#     visual_action_values = np.zeros((2, 10, 10))
#     for b, p, d in product(range(2), range(12, 22), range(1, 11)):
#         visual_action_values[b][p - 12][d - 1] = action_values[(bool(b), p, d)]
#     for b in range(2):
#         plt.imshow(visual_action_values[b])
#         plt.show()


def main(trials=10000, lmbda=1):
    pi = Policy()
    action_values = {(s, a): 0 for s, a in product(_STATES, _ACTIONS)}

    for _ in range(trials):
        s = choice(_STATES)
        a = choice(_ACTIONS)
        player_cards = [card(), card()]
        dealer_cards = [card(), card()]
        while cardsum(player_cards) < 12:
            player_cards.append(card())
        state = cards_to_state(player_cards, dealer_cards)
        reward = simulate_and_get_reward(player_cards, dealer_cards)
        action_values[state] = (
            action_values[state] * state_trials[state] + reward) / (state_trials[state] + 1)
        state_trials[state] += 1

    visualize_action_values(action_values)


if __name__ == '__main__':
    main()
