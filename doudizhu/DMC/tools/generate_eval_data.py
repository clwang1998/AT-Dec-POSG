"""Generate evaluation data for full-game Doudizhu evaluation."""

import argparse
import pickle
import numpy as np


deck = []
for i in range(3, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])


def get_parser():
    parser = argparse.ArgumentParser(description='DouZero: random data generator')
    parser.add_argument('--output', default='eval_data', type=str)
    parser.add_argument('--num_games', default=1000, type=int)
    parser.add_argument('--seed', default=2026, type=int)
    return parser


def generate(rng):
    shuffled = list(rng.permutation(deck))
    hands = [
        sorted(shuffled[:17]),
        sorted(shuffled[17:34]),
        sorted(shuffled[34:51]),
    ]
    landlord_cards = sorted(shuffled[51:54])
    landlord_player = int(rng.integers(0, 3))
    hands[landlord_player] = sorted(hands[landlord_player] + landlord_cards)
    card_play_data = {
        'landlord': hands[landlord_player],
        'landlord_up': hands[(landlord_player - 1) % 3],
        'landlord_down': hands[(landlord_player + 1) % 3],
        'three_landlord_cards': landlord_cards,
    }
    return card_play_data


def main() -> None:
    flags = get_parser().parse_args()
    rng = np.random.default_rng(flags.seed)
    output_pickle = flags.output + '.pkl'

    print('output_pickle:', output_pickle)
    print('generating data...')

    data = [generate(rng) for _ in range(flags.num_games)]

    print('saving pickle file...')
    with open(output_pickle, 'wb') as file_obj:
        pickle.dump(data, file_obj, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
