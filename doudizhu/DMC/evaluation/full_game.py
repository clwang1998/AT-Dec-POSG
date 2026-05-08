import copy
import multiprocessing as mp
import os
import pickle
import re
import sys
from contextlib import contextmanager

from .game_eval import GameEnv

from .deep_agent import DeepAgent
from .random_agent import RandomAgent


FULL_GAME_POSITIONS = (
    'first',
    'second',
    'third',
    'landlord',
    'landlord_down',
    'landlord_up',
)

PLAYCARD_POSITIONS = (
    'landlord',
    'landlord_down',
    'landlord_up',
)


@contextmanager
def _temporary_cwd(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _perfectdou_infoset_adapter(infoset):
    adapted = copy.deepcopy(infoset)
    action_seq = getattr(adapted, 'card_play_action_seq', None)
    if action_seq:
        normalized = []
        for item in action_seq:
            if isinstance(item, tuple) and len(item) == 2:
                normalized.append(list(item[1]))
            else:
                normalized.append(list(item))
        adapted.card_play_action_seq = normalized
    return adapted


class _DMCCompatiblePerfectDouAgent:

    def __init__(self, base_agent):
        self._base_agent = base_agent

    def act(self, infoset):
        return self._base_agent.act(_perfectdou_infoset_adapter(infoset))


def _normalize_card_play_data(card_play_data):
    if 'first' in card_play_data:
        return card_play_data
    if 'landlord' not in card_play_data:
        raise KeyError('Unsupported eval_data sample format.')

    landlord_cards = list(card_play_data['three_landlord_cards'])
    first_hand = list(card_play_data['landlord'])
    for card in landlord_cards:
        first_hand.remove(card)

    return {
        'first': sorted(first_hand),
        'second': sorted(card_play_data['landlord_down']),
        'third': sorted(card_play_data['landlord_up']),
        'three_landlord_cards': sorted(landlord_cards),
    }


def infer_bidding_model_path(play_model_paths):
    frame_pattern = re.compile(
        r'^general_(?:landlord|landlord_down|landlord_up|farmer)_(\d+)\.ckpt$')
    for model_path in play_model_paths:
        if not model_path or model_path == 'random':
            continue
        directory = os.path.dirname(model_path)
        basename = os.path.basename(model_path)
        frame_match = frame_pattern.match(basename)
        if frame_match is not None:
            candidate = os.path.join(
                directory, f'general_bidding_{frame_match.group(1)}.ckpt')
            if os.path.exists(candidate):
                return candidate
        for token in ('landlord_down', 'landlord_up', 'landlord', 'farmer'):
            if token in basename:
                candidate = os.path.join(
                    directory, basename.replace(token, 'bidding', 1))
                if os.path.exists(candidate):
                    return candidate
    return None


def resolve_model_paths(
    first_bid,
    second_bid,
    third_bid,
    landlord,
    landlord_down,
    landlord_up,
    shared_bidding='auto',
):
    play_model_paths = {
        'landlord': landlord,
        'landlord_down': landlord_down,
        'landlord_up': landlord_up,
    }
    explicit_bid_paths = [first_bid, second_bid, third_bid]
    if any(path not in (None, '', 'auto') for path in explicit_bid_paths):
        bid_model_paths = {
            'first': first_bid if first_bid not in (None, '', 'auto') else 'random',
            'second': second_bid if second_bid not in (None, '', 'auto') else 'random',
            'third': third_bid if third_bid not in (None, '', 'auto') else 'random',
        }
    else:
        shared_bid_path = shared_bidding
        if shared_bid_path in (None, '', 'auto'):
            shared_bid_path = infer_bidding_model_path(play_model_paths.values()) or 'random'
        bid_model_paths = {
            'first': shared_bid_path,
            'second': shared_bid_path,
            'third': shared_bid_path,
        }
    return {
        **bid_model_paths,
        **play_model_paths,
    }


def _assert_playcard_position(position, agent_name):
    if position in PLAYCARD_POSITIONS:
        return
    raise ValueError(
        f'`{agent_name}` is only supported for play-card positions, '
        f'not bidding position `{position}`.'
    )


def _load_rlcard_agent(position):
    _assert_playcard_position(position, 'rlcard')
    from .rlcard_agent import RLCardAgent
    return RLCardAgent(position)


def _import_perfectdou_agent(repo_root=None):
    import_error = None
    try:
        from perfectdou.evaluation.perfectdou_agent import PerfectDouAgent
        return PerfectDouAgent
    except Exception as exc:  # pragma: no cover - exercised by integration
        import_error = exc

    repo_candidates = []
    if repo_root:
        repo_candidates.append(repo_root)
    env_root = os.environ.get('PERFECTDOU_REPO')
    if env_root:
        repo_candidates.append(env_root)

    for candidate in repo_candidates:
        candidate = os.path.abspath(candidate)
        if not os.path.isdir(candidate):
            continue
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        try:
            with _temporary_cwd(candidate):
                from perfectdou.evaluation.perfectdou_agent import PerfectDouAgent
            return PerfectDouAgent
        except Exception as exc:  # pragma: no cover - exercised by integration
            import_error = exc

    message = (
        'PerfectDou support requires the official PerfectDou repo and its '
        'dependencies (including onnxruntime). Either install it into the '
        'current environment, or set PERFECTDOU_REPO=/path/to/PerfectDou, '
        'or pass a play model like `perfectdou:/path/to/PerfectDou`.'
    )
    if import_error is not None:
        raise ImportError(message) from import_error
    raise ImportError(message)


def _load_perfectdou_agent(position, model_path):
    _assert_playcard_position(position, 'perfectdou')
    repo_root = None
    if model_path.startswith('perfectdou:'):
        _, _, repo_root = model_path.partition(':')
        repo_root = repo_root or None
    PerfectDouAgent = _import_perfectdou_agent(repo_root=repo_root)
    if repo_root:
        with _temporary_cwd(os.path.abspath(repo_root)):
            return _DMCCompatiblePerfectDouAgent(PerfectDouAgent(position))
    return _DMCCompatiblePerfectDouAgent(PerfectDouAgent(position))


def _infer_perfectdou_repo_root_from_checkpoint(checkpoint_path):
    if not checkpoint_path:
        return None
    candidate = os.path.abspath(os.path.dirname(checkpoint_path))
    while True:
        if os.path.isdir(os.path.join(candidate, 'perfectdou', 'evaluation')):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            return None
        candidate = parent


def _import_perfectdou_repro_agent(checkpoint_path=None, repo_root=None):
    import_error = None
    try:
        from perfectdou.evaluation.repro_agent import ReproAgent
        return ReproAgent
    except Exception as exc:  # pragma: no cover - exercised by integration
        import_error = exc

    repo_candidates = []
    if repo_root:
        repo_candidates.append(repo_root)
    inferred_root = _infer_perfectdou_repo_root_from_checkpoint(checkpoint_path)
    if inferred_root:
        repo_candidates.append(inferred_root)
    for env_name in ('PERFECTDOU_REPRO_REPO', 'PERFECTDOU_REPO'):
        env_root = os.environ.get(env_name)
        if env_root:
            repo_candidates.append(env_root)

    for candidate in repo_candidates:
        candidate = os.path.abspath(candidate)
        if not os.path.isdir(candidate):
            continue
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        try:
            from perfectdou.evaluation.repro_agent import ReproAgent
            return ReproAgent
        except Exception as exc:  # pragma: no cover - exercised by integration
            import_error = exc

    message = (
        'PerfectDou reproduction checkpoints require the local PerfectDou-test '
        'repo with `perfectdou/evaluation/repro_agent.py`. Pass a play model like '
        '`perfectdou_repro:/path/to/checkpoint.pt`, or set '
        '`PERFECTDOU_REPRO_REPO=/path/to/PerfectDou-test`.'
    )
    if import_error is not None:
        raise ImportError(message) from import_error
    raise ImportError(message)


def _load_perfectdou_repro_agent(position, model_path):
    _assert_playcard_position(position, 'perfectdou_repro')
    checkpoint_path = model_path
    if model_path.startswith('perfectdou_repro:'):
        _, _, checkpoint_path = model_path.partition(':')
    ReproAgent = _import_perfectdou_repro_agent(checkpoint_path=checkpoint_path)
    return ReproAgent(position, checkpoint_path)


def load_full_game_models(model_path_dict):
    players = {}
    for position in FULL_GAME_POSITIONS:
        model_path = model_path_dict[position]
        if model_path == 'random':
            players[position] = RandomAgent()
        elif model_path == 'rlcard':
            players[position] = _load_rlcard_agent(position)
        elif model_path == 'perfectdou' or model_path.startswith('perfectdou:'):
            players[position] = _load_perfectdou_agent(position, model_path)
        elif position in PLAYCARD_POSITIONS and (
            model_path.startswith('perfectdou_repro:')
            or model_path.endswith('.pt')
            or model_path.endswith('.pth')
        ):
            players[position] = _load_perfectdou_repro_agent(position, model_path)
        else:
            players[position] = DeepAgent(position, model_path)
    return players


def _worker(card_play_data_list, model_path_dict, q):
    env = GameEnv(load_full_game_models(model_path_dict))
    bid_count_hist = [0, 0, 0, 0]

    for card_play_data in card_play_data_list:
        env.bid_init(copy.deepcopy(_normalize_card_play_data(card_play_data)))
        while not env.bid_over:
            env.step()
        if not env.draw:
            if env.bid_count > 0:
                bid_count_hist[env.bid_count - 1] += 1
            while not env.game_over:
                env.step()
        env.reset()

    q.put((
        env.num_wins['landlord'],
        env.num_wins['farmer'],
        env.num_wins['first'],
        env.num_wins['second'],
        env.num_wins['third'],
        env.num_wins['draw'],
        env.num_scores['landlord'],
        env.num_scores['farmer'],
        env.num_scores['first'],
        env.num_scores['second'],
        env.num_scores['third'],
        bid_count_hist,
    ))


def data_allocation_per_worker(card_play_data_list, num_workers):
    worker_data = [[] for _ in range(num_workers)]
    for index, data in enumerate(card_play_data_list):
        worker_data[index % num_workers].append(data)
    return worker_data


def evaluate_full_game(model_path_dict, eval_data, num_workers):
    with open(eval_data, 'rb') as file_obj:
        card_play_data_list = pickle.load(file_obj)

    worker_inputs = data_allocation_per_worker(card_play_data_list, num_workers)
    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []

    for worker_data in worker_inputs:
        process = ctx.Process(target=_worker, args=(worker_data, model_path_dict, q))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    totals = {
        'landlord_wins': 0,
        'farmer_wins': 0,
        'first_wins': 0,
        'second_wins': 0,
        'third_wins': 0,
        'draws': 0,
        'landlord_scores': 0,
        'farmer_scores': 0,
        'first_scores': 0,
        'second_scores': 0,
        'third_scores': 0,
        'bid_count_hist': [0, 0, 0, 0],
    }

    for _ in range(num_workers):
        result = q.get()
        totals['landlord_wins'] += result[0]
        totals['farmer_wins'] += result[1]
        totals['first_wins'] += result[2]
        totals['second_wins'] += result[3]
        totals['third_wins'] += result[4]
        totals['draws'] += result[5]
        totals['landlord_scores'] += result[6]
        totals['farmer_scores'] += result[7]
        totals['first_scores'] += result[8]
        totals['second_scores'] += result[9]
        totals['third_scores'] += result[10]
        for index, count in enumerate(result[11]):
            totals['bid_count_hist'][index] += count

    num_total_wins = totals['landlord_wins'] + totals['farmer_wins']
    if num_total_wins == 0:
        raise RuntimeError('No completed non-draw games were evaluated.')

    print('Resolved models:')
    for position in FULL_GAME_POSITIONS:
        print(f'  {position}: {model_path_dict[position]}')
    print('WP results:')
    print(
        'First : Second : Third - {} : {} : {}'.format(
            totals['first_wins'] / num_total_wins,
            totals['second_wins'] / num_total_wins,
            totals['third_wins'] / num_total_wins,
        )
    )
    print(
        'landlord : Farmers - {} : {}'.format(
            totals['landlord_wins'] / num_total_wins,
            totals['farmer_wins'] / num_total_wins,
        )
    )
    print('ADP results:')
    print(
        'First : Second : Third - {} : {} : {}'.format(
            totals['first_scores'] / num_total_wins,
            totals['second_scores'] / num_total_wins,
            totals['third_scores'] / num_total_wins,
        )
    )
    print(
        'landlord : Farmers - {} : {}'.format(
            totals['landlord_scores'] / num_total_wins,
            totals['farmer_scores'] / num_total_wins,
        )
    )
    print(f"number of draw: - {totals['draws']}")
    print(f"bid count histogram: {totals['bid_count_hist']}")
