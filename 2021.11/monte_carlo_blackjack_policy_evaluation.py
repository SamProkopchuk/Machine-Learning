'''
State-value monte carlo approximation for blackjack

Dealer:
 - While sum(cards) < 17: hit

Agent's policy:
 - W
'''

import numpy as np
from random import choice
from itertools import product

'''
States:
 - Whether the player has a usable ace (False, True)
 - The sum of the player's cards (12, 21)
 - The card the dealer shows (1 (Ace), 10)
'''


def card():
    return np.clip(choice(range(1, 14), 1, 10))


def cards_to_state(player_cards, dealer_cards):
    pass


def simulate_and_get_reward(player_cards, dealer_cards):
    pass


def main(trials=10000, lmbda=1):
    states = {(b, p, d) for b, p, d in product(
        (False, True),
        range(12, 22),
        range(1, 11))}
    state_values = {s: 0 for s in states}
    state_returns = state_values.copy()

    for _ in range(trials):
        player_cards = [card(), card()]
        dealer_cards = [card(), card()]
        state = cards_to_state(player_cards, dealer_cards)


if __name__ == '__main__':
    main()
