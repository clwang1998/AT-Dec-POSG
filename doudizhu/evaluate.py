import argparse
import os
import sys


def _extract_gpu_device(argv):
    for index, arg in enumerate(argv):
        if arg == '--gpu_device' and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith('--gpu_device='):
            return arg.split('=', 1)[1]
    return ''


def _seed_everything(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['CUDA_VISIBLE_DEVICES'] = _extract_gpu_device(sys.argv[1:])

    parser = argparse.ArgumentParser('DMC Full-Game Evaluation')
    parser.add_argument('--player_1_bid', type=str, default='auto')
    parser.add_argument('--player_2_bid', type=str, default='auto')
    parser.add_argument('--player_3_bid', type=str, default='auto')
    parser.add_argument(
        '--bidding',
        type=str,
        default='auto',
        help='Shared bidding checkpoint. Use `auto` to infer a sibling '
             '`general_bidding_<frames>.ckpt` file from the play checkpoints.',
    )
    play_help = (
        'Play-card agent: `random`, `rlcard`, `perfectdou`, '
        '`perfectdou:/path/to/PerfectDou`, '
        '`perfectdou_repro:/path/to/checkpoint.pt`, or a checkpoint path.'
    )
    parser.add_argument('--landlord', '--player_1_playcard', dest='landlord', type=str, default='random', help=play_help)
    parser.add_argument('--landlord_down', '--player_2_playcard', dest='landlord_down', type=str, default='random', help=play_help)
    parser.add_argument('--landlord_up', '--player_3_playcard', dest='landlord_up', type=str, default='random', help=play_help)
    parser.add_argument('--eval_data', type=str, default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--gpu_device', type=str, default='')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
    _seed_everything(args.seed)

    from DMC.evaluation.full_game import evaluate_full_game, resolve_model_paths

    model_path_dict = resolve_model_paths(
        args.player_1_bid,
        args.player_2_bid,
        args.player_3_bid,
        args.landlord,
        args.landlord_down,
        args.landlord_up,
        shared_bidding=args.bidding,
    )
    evaluate_full_game(model_path_dict, args.eval_data, args.num_workers)
