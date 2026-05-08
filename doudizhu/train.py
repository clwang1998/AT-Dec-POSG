import os
import sys


def _extract_gpu_devices(argv):
    for index, arg in enumerate(argv):
        if arg == '--gpu_devices' and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith('--gpu_devices='):
            return arg.split('=', 1)[1]
    return '0'


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
    # Set the visible-device mask before importing the training package so
    # CUDA initialization follows the same contract as DouZero-style launches.
    os.environ["CUDA_VISIBLE_DEVICES"] = _extract_gpu_devices(sys.argv[1:])

    from DMC.dmc import parser, train

    flags = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    _seed_everything(flags.seed)
    train(flags)
