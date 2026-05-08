"""Training CLI for the modified DouZero pipeline.

Compared with the original DouZero setup, this project adds flags for:
- prioritized replay
- paper Module A / B / C toggles for ablations
- optional bidding and multiply-side training hooks
"""

import argparse
import os
from pathlib import Path


def parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'invalid boolean value: {value}')


parser = argparse.ArgumentParser(description='DouZero: PyTorch DouDizhu AI')

# General Settings
parser.add_argument('--xpid', default='dmc',
                    help='Experiment id (default: dmc)')
parser.add_argument('--seed', default=2026, type=int,
                    help='Base random seed for reproducible training runs')
parser.add_argument('--save_interval', default=10, type=int,
                    help='Time interval (in minutes) at which to save the model')    
parser.add_argument('--objective', default='adp', type=str, choices=['adp', 'wp', 'logadp'],
                    help='Use ADP or WP as reward (default: ADP)')    

# Training settings
parser.add_argument('--actor_device_cpu', action='store_true',
                    help='Use CPU as actor device')
parser.add_argument('--gpu_devices', default='0', type=str,
                    help='Comma-separated visible GPU list for this run')
parser.add_argument('--num_actor_devices', default=1, type=int,
                    help='The number of devices used for simulation')
parser.add_argument('--num_actors', default=2, type=int,
                    help='The number of actors for each simulation device')
parser.add_argument('--training_device', default='0', type=str,
                    help='Learner device index within the visible GPU list. `cpu` means using cpu')
parser.add_argument('--load_model', action='store_true',
                    help='Load an existing model')
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='Disable saving checkpoint')
parser.add_argument('--savedir', default='dmc_checkpoints',
                    help='Root dir where experiment data will be saved')

# Hyperparameters
parser.add_argument('--total_frames', default=100000000000, type=int,
                    help='Total environment frames to train for')
parser.add_argument('--exp_epsilon', default=0.01, type=float,
                    help='The probability for exploration')
parser.add_argument('--replay_buffer_size', default=64, type=int,
                    help='Number of unrolls stored in prioritized replay for each position')
parser.add_argument('--replay_warmup_size', default=16, type=int,
                    help='Minimum number of unrolls before prioritized replay starts sampling')
parser.add_argument('--priority_alpha', default=0.6, type=float,
                    help='Exponent used to turn priorities into sampling probabilities')
parser.add_argument('--priority_beta', default=0.4, type=float,
                    help='Importance-sampling correction strength for prioritized replay')
parser.add_argument('--priority_epsilon', default=1e-6, type=float,
                    help='Small constant added to TD errors when updating priorities')
parser.add_argument('--enable_module_a', type=parse_bool_arg, nargs='?', const=True, default=True,
                    help='Whether to enable Module A. Default: true. For ablations pass `--enable_module_a false`.')
parser.add_argument('--enable_module_b', type=parse_bool_arg, nargs='?', const=True, default=True,
                    help='Whether to enable Module B. Default: true. For ablations pass `--enable_module_b false`.')
parser.add_argument('--enable_module_c', type=parse_bool_arg, nargs='?', const=True, default=True,
                    help='Whether to enable Module C. Default: true. For ablations pass `--enable_module_c false`.')
parser.add_argument('--separate_farmer_seats', type=parse_bool_arg, nargs='?', const=True, default=False,
                    help='Train landlord_up and landlord_down as separate main play learners instead of a shared farmer model. Default: false.')
parser.add_argument('--train_bidding', type=parse_bool_arg, nargs='?', const=True, default=True,
                    help='Whether to train the explicit bidding head. Default: true. For ablations pass `--train_bidding false`.')
parser.add_argument('--train_multiply', action='store_true',
                    help='Train multiply-stage play records as extra play transitions')
parser.add_argument('--belief_coef', default=0.05, type=float,
                    help='Weight of the hidden-hand belief prediction loss for play positions')
parser.add_argument('--coord_sender_coef', default=0.05, type=float,
                    help='Weight of supervising farmer coordination embeddings with sender private hand targets')
parser.add_argument('--coord_receiver_coef', default=0.05, type=float,
                    help='Weight of teammate belief consistency loss driven by coordination embeddings')
parser.add_argument('--opponent_pool_size', default=6, type=int,
                    help='Legacy compatibility: number of historical checkpoints kept for the old opponent pool')
parser.add_argument('--league_snapshot_size', default=6, type=int,
                    help='Number of recent policy snapshots kept in the league pool for each group')
parser.add_argument('--league_exploiter_size', default=4, type=int,
                    help='Number of top-scoring exploiter checkpoints kept in the league pool for each group')
parser.add_argument('--league_main_prob', default=0.2, type=float,
                    help='Probability of sampling the live main policy as the opponent source')
parser.add_argument('--league_snapshot_prob', default=0.5, type=float,
                    help='Probability of sampling a recent snapshot from the league pool')
parser.add_argument('--league_exploiter_prob', default=0.3, type=float,
                    help='Probability of sampling a score-ranked exploiter from the league pool')
parser.add_argument('--external_opponent', default='', type=str,
                    help='Optional external sparring opponent added to Module C sampling. Supported: perfectdou')
parser.add_argument('--external_opponent_prob', default=0.0, type=float,
                    help='Probability mass assigned to the external sparring opponent when Module C sampling is enabled')
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REPO_PERFECTDOU_ROOT = _PROJECT_ROOT.parent / "PerfectDou-official"
_DEFAULT_OFFICIAL_PERFECTDOU_ROOT = Path(
    os.environ.get("PERFECTDOU_REPO_ROOT", str(_DEFAULT_REPO_PERFECTDOU_ROOT))
)
_DEFAULT_OFFICIAL_PERFECTDOU_DIR = Path(
    os.environ.get(
        "PERFECTDOU_DIR",
        str(_DEFAULT_OFFICIAL_PERFECTDOU_ROOT / "perfectdou" / "model" / "perfectdou"),
    )
)
parser.add_argument('--perfectdou_repo_root', default=str(_DEFAULT_OFFICIAL_PERFECTDOU_ROOT), type=str,
                    help='PerfectDou repo root used when `--external_opponent perfectdou` is enabled')
parser.add_argument('--perfectdou_dir', default=str(_DEFAULT_OFFICIAL_PERFECTDOU_DIR), type=str,
                    help='Directory containing PerfectDou ONNX checkpoints for external-opponent rollouts')
parser.add_argument('--league_exploiter_train_prob', default=0.33, type=float,
                    help='Probability that a self-play episode trains an exploiter instead of a main policy')
parser.add_argument('--league_exploiter_reset_interval', default=20000, type=int,
                    help='Frames between resetting each exploiter from its corresponding live main policy; 0 disables resets')
parser.add_argument('--minimax_exploiter_alpha', default=0.12, type=float,
                    help='Two-sided exploiter shaping strength for landlord/farmer exploiters.')
parser.add_argument('--minimax_exploiter_gamma', default=1.0, type=float,
                    help='Discount factor used in minimax exploiter shaping (clipped to [0, 1]).')
parser.add_argument('--minimax_value_floor', default=0.0, type=float,
                    help='Lower bound used for shifted opponent values in minimax shaping; bonus uses `-alpha*gamma*max(value-floor, 0)` to keep the shaping term non-positive.')
parser.add_argument('--minimax_exploiter_enabled', type=parse_bool_arg, nargs='?', const=True, default=True,
                    help='Enable two-sided minimax exploiter shaping during actor rollout. Pass `--minimax_exploiter_enabled false` to disable.')
parser.add_argument('--opponent_refresh_episodes', default=20, type=int,
                    help='How often actors refresh the opponent pool metadata')
parser.add_argument('--sa_initial_temperature', default=1.0, type=float,
                    help='Initial temperature for simulated annealing opponent selection')
parser.add_argument('--sa_final_temperature', default=0.1, type=float,
                    help='Final temperature for simulated annealing opponent selection')
parser.add_argument('--sa_decay_episodes', default=2000, type=int,
                    help='Number of episodes used to anneal opponent sampling temperature')
parser.add_argument('--batch_size', default=16, type=int,
                    help='Learner batch size')
parser.add_argument('--unroll_length', default=100, type=int,
                    help='The unroll length (time dimension)')
parser.add_argument('--num_buffers', default=50, type=int,
                    help='Number of shared-memory buffers')
parser.add_argument('--num_threads', default=1, type=int,
                    help='Number learner threads')
parser.add_argument('--max_grad_norm', default=40., type=float,
                    help='Max norm of gradients')

# Optimizer settings
parser.add_argument('--learning_rate', default=0.0001, type=float,
                    help='Learning rate')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant')
parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum')
parser.add_argument('--epsilon', default=1e-8, type=float,
                    help='RMSProp epsilon')
