"""League-style self-play utilities.

Original DouZero mainly trains against the current self-play population.
This project extends that with:
- snapshot checkpoints for diversity
- exploiter policies as approximate best responses
- annealed opponent sampling over the league pool
"""

import json
import os

import numpy as np
import torch

from .perfectdou_external import PerfectDouExternalOpponent
from .models import (
    canonical_model_dict,
    canonical_position,
    play_group_for_training_position,
    is_exploiter_position,
    get_main_training_positions,
    get_exploiter_training_positions,
    module_c_enabled,
    separate_farmer_seats_enabled,
)


PLAY_GROUPS = ('landlord', 'farmer')
LEAGUE_ENTRY_TYPES = ('snapshot', 'exploiter')


def _to_device(device):
    if device == 'cpu':
        return torch.device('cpu')
    return torch.device('cuda:' + str(device))


def load_matching_weights(model, pretrained_state):
    model_state = model.state_dict()
    filtered_state = {
        key: value for key, value in pretrained_state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model_state.update(filtered_state)
    model.load_state_dict(model_state)


def _pool_root(flags):
    return os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'league_pool')))


def _metadata_path(flags, group, entry_type):
    return os.path.join(_pool_root(flags), f'{group}_{entry_type}.json')


def _resolve_entry_path(flags, path):
    if not path:
        return path
    resolved = os.path.expandvars(os.path.expanduser(path))
    if os.path.exists(resolved):
        return resolved
    # Keep league metadata usable after repository moves by rebasing stale
    # absolute paths to the current pool root when a matching checkpoint file
    # exists there.
    rebased = os.path.join(_pool_root(flags), os.path.basename(resolved))
    if os.path.exists(rebased):
        return rebased
    if '/AT-Dec-POS/' in resolved:
        migrated = resolved.replace('/AT-Dec-POS/', '/AT-Dec-POSG/')
        if os.path.exists(migrated):
            return migrated
    return resolved


def _load_metadata(flags, group, entry_type):
    path = _metadata_path(flags, group, entry_type)
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as file_obj:
        metadata = json.load(file_obj)
    valid_entries = []
    for raw_entry in metadata:
        entry = dict(raw_entry)
        entry.setdefault('type', entry_type)
        entry.setdefault('group', group)
        entry['path'] = _resolve_entry_path(flags, entry.get('path'))
        if not entry.get('path') or not os.path.exists(entry['path']):
            continue
        valid_entries.append(entry)
    return valid_entries


def _save_metadata(flags, group, entry_type, metadata):
    os.makedirs(_pool_root(flags), exist_ok=True)
    with open(_metadata_path(flags, group, entry_type), 'w', encoding='utf-8') as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=True, indent=2)


def _checkpoint_path(flags, group, entry_type, frame):
    return os.path.join(_pool_root(flags), f'{group}_{entry_type}_{frame}.ckpt')


def _keep_limit(flags, entry_type):
    if entry_type == 'snapshot':
        return flags.league_snapshot_size
    return flags.league_exploiter_size


def _sort_metadata(metadata, entry_type):
    if entry_type == 'snapshot':
        return sorted(metadata, key=lambda entry: entry.get('frame', 0), reverse=True)
    return sorted(
        metadata,
        key=lambda entry: (entry.get('score', 0.0), entry.get('frame', 0)),
        reverse=True)


def _prune_metadata(flags, group, entry_type, metadata):
    metadata = _sort_metadata(metadata, entry_type)
    keep = metadata[:_keep_limit(flags, entry_type)]
    keep_paths = {entry['path'] for entry in keep}
    for entry in metadata[_keep_limit(flags, entry_type):]:
        stale_path = entry.get('path')
        if stale_path and stale_path not in keep_paths and os.path.exists(stale_path):
            os.remove(stale_path)
    return keep


def _score_map(stats):
    farmer_score = stats.get('mean_episode_return_farmer')
    if farmer_score is None:
        farmer_score = np.mean([
            float(stats.get('mean_episode_return_landlord_up', 0.0)),
            float(stats.get('mean_episode_return_landlord_down', 0.0)),
        ])
    return {
        'landlord': float(stats.get('mean_episode_return_landlord', 0.0)),
        'farmer': float(farmer_score),
    }


def _save_entry(flags, frame, learner_model, group, entry_type, score):
    path = _checkpoint_path(flags, group, entry_type, frame)
    if entry_type == 'snapshot':
        model_key = group
        if separate_farmer_seats_enabled(flags) and group == 'farmer':
            # Separate-seat training has no shared `farmer` main model; use
            # the canonical landlord_up weights for the league snapshot, which
            # matches the existing checkpoint restore fallback.
            model_key = 'landlord_up'
    else:
        model_key = f'{group}_exploiter'
    torch.save(learner_model.get_model(model_key).state_dict(), path)
    metadata = _load_metadata(flags, group, entry_type)
    metadata = [entry for entry in metadata if entry.get('frame') != frame]
    metadata.append({
        'frame': int(frame),
        'path': path,
        'score': float(score),
        'type': entry_type,
        'group': group,
    })
    metadata = _prune_metadata(flags, group, entry_type, metadata)
    _save_metadata(flags, group, entry_type, metadata)


def save_opponent_pool_snapshot(flags, frame, learner_model, stats):
    if not module_c_enabled(flags):
        return
    os.makedirs(_pool_root(flags), exist_ok=True)
    score_map = _score_map(stats)
    for group in PLAY_GROUPS:
        _save_entry(flags, frame, learner_model, group, 'snapshot', score_map[group])
        _save_entry(flags, frame, learner_model, group, 'exploiter', score_map[group])


class LeaguePolicyPool:
    def __init__(self, flags, device):
        self.flags = flags
        self.device = device
        self.metadata = {
            group: {entry_type: [] for entry_type in LEAGUE_ENTRY_TYPES}
            for group in PLAY_GROUPS
        }
        self.cache = {}
        self.external_cache = {}
        self.episode_index = 0
        self.last_refresh_episode = -1
        self.main_index = 0
        self.exploiter_index = 0

    def _refresh(self, force=False):
        if not force and self.last_refresh_episode >= 0:
            if self.episode_index - self.last_refresh_episode < self.flags.opponent_refresh_episodes:
                return
        for group in PLAY_GROUPS:
            for entry_type in LEAGUE_ENTRY_TYPES:
                self.metadata[group][entry_type] = _load_metadata(
                    self.flags, group, entry_type)
        self.last_refresh_episode = self.episode_index

    def _temperature(self):
        decay_episodes = max(1, self.flags.sa_decay_episodes)
        progress = min(1.0, self.episode_index / decay_episodes)
        start = self.flags.sa_initial_temperature
        end = self.flags.sa_final_temperature
        return max(end, start + (end - start) * progress)

    def _load_model(self, group, path):
        cache_key = (group, path)
        model = self.cache.get(cache_key)
        if model is None:
            model = canonical_model_dict[group]().to(_to_device(self.device))
            if torch.cuda.is_available() and self.device != 'cpu':
                pretrained = torch.load(path, map_location=_to_device(self.device))
            else:
                pretrained = torch.load(path, map_location='cpu')
            load_matching_weights(model, pretrained)
            model.eval()
            self.cache[cache_key] = model
        return model

    def sample_focus_group(self):
        main_positions = get_main_training_positions(self.flags)
        exploiter_positions = get_exploiter_training_positions(self.flags)
        if not module_c_enabled(self.flags) or len(exploiter_positions) == 0:
            focus_group = main_positions[self.main_index % len(main_positions)]
            self.main_index += 1
            self.episode_index += 1
            return focus_group
        # Each episode updates either a main policy or an exploiter. This
        # turns one self-play stream into a lightweight league schedule.
        use_exploiter = np.random.rand() < self.flags.league_exploiter_train_prob
        if use_exploiter:
            focus_group = exploiter_positions[
                self.exploiter_index % len(exploiter_positions)]
            self.exploiter_index += 1
        else:
            focus_group = main_positions[
                self.main_index % len(main_positions)]
            self.main_index += 1
        self.episode_index += 1
        return focus_group

    def _entry_probabilities(self, entries, entry_type):
        if len(entries) == 0:
            return None
        if entry_type == 'snapshot':
            energies = np.array([entry.get('frame', 0) for entry in entries], dtype=np.float32)
        else:
            energies = np.array([entry.get('score', 0.0) for entry in entries], dtype=np.float32)
        energies = energies - float(np.max(energies))
        probabilities = np.exp(energies / max(self._temperature(), 1e-6))
        probability_sum = probabilities.sum()
        if not np.isfinite(probability_sum) or probability_sum <= 0:
            return np.full(len(entries), 1.0 / len(entries), dtype=np.float32)
        return probabilities / probability_sum

    def _external_opponent_name(self):
        return str(getattr(self.flags, 'external_opponent', '') or '').strip().lower()

    def _external_opponent_enabled(self):
        return (
            self._external_opponent_name() == 'perfectdou'
            and float(getattr(self.flags, 'external_opponent_prob', 0.0)) > 0.0
        )

    def _sample_external_opponent(self, group):
        if not self._external_opponent_enabled():
            return None, None
        opponent_name = self._external_opponent_name()
        model = self.external_cache.get(opponent_name)
        if model is None:
            if opponent_name != 'perfectdou':
                return None, None
            model = PerfectDouExternalOpponent(
                perfectdou_dir=getattr(self.flags, 'perfectdou_dir'),
                repo_root=getattr(self.flags, 'perfectdou_repo_root'),
            )
            self.external_cache[opponent_name] = model
        return model, {'group': group, 'type': 'external', 'name': opponent_name}

    def _sample_from_entries(self, group, entry_type):
        entries = self.metadata.get(group, {}).get(entry_type, [])
        if len(entries) == 0:
            return None, None
        probabilities = self._entry_probabilities(entries, entry_type)
        choice = int(np.random.choice(len(entries), p=probabilities))
        selected = dict(entries[choice])
        return self._load_model(group, selected['path']), selected

    def _normalized_role_probs(self):
        raw = [
            ('main', self.flags.league_main_prob),
            ('snapshot', self.flags.league_snapshot_prob),
            ('exploiter', self.flags.league_exploiter_prob),
        ]
        if self._external_opponent_enabled():
            raw.append(('external', self.flags.external_opponent_prob))
        role_names = [name for name, _ in raw]
        role_weights = np.array([weight for _, weight in raw], dtype=np.float32)
        role_weights = np.maximum(role_weights, 0.0)
        if role_weights.sum() <= 0:
            fallback = np.zeros(len(role_names), dtype=np.float32)
            fallback[role_names.index('snapshot') if 'snapshot' in role_names else 0] = 1.0
            return role_names, fallback
        return role_names, role_weights / role_weights.sum()

    def sample_opponent(self, target_group):
        if not module_c_enabled(self.flags):
            opponent_group = 'farmer' if play_group_for_training_position(target_group) == 'landlord' else 'landlord'
            return None, {'group': opponent_group, 'type': 'main'}
        if is_exploiter_position(target_group):
            # Exploiters are trained as best responses against the current
            # live opponent rather than against historical policies.
            opponent_group = 'farmer' if play_group_for_training_position(target_group) == 'landlord' else 'landlord'
            return None, {'group': opponent_group, 'type': 'main'}
        self._refresh()
        opponent_group = 'farmer' if target_group == 'landlord' else 'landlord'
        role_names, role_probs = self._normalized_role_probs()
        role_order = list(np.random.choice(
            role_names, size=len(role_names), replace=False, p=role_probs))
        for role in role_order:
            if role == 'main':
                return None, {'group': opponent_group, 'type': 'main'}
            if role == 'external':
                model, info = self._sample_external_opponent(opponent_group)
                if model is not None:
                    return model, info
                continue
            model, info = self._sample_from_entries(opponent_group, role)
            if model is not None:
                return model, info
        return None, {'group': opponent_group, 'type': 'main'}


AnnealedOpponentPool = LeaguePolicyPool


class EpisodePolicyModel:
    def __init__(self, current_model, opponent_pool):
        self.current_model = current_model
        self.opponent_pool = opponent_pool
        self.training_position = 'landlord'
        self.training_play_group = 'landlord'
        self.opponent_group = 'farmer'
        self.opponent_info = None
        self.role_models = {}

    def prepare_next_episode(self):
        # Build one episode-specific policy view: one side may be the live
        # training policy while the opponent may come from the league pool.
        self.training_position = self.opponent_pool.sample_focus_group()
        self.training_play_group = play_group_for_training_position(self.training_position)
        self.opponent_group = 'farmer' if self.training_play_group == 'landlord' else 'landlord'
        opponent_model, opponent_info = self.opponent_pool.sample_opponent(self.training_position)
        self.opponent_info = opponent_info
        separate_farmer_seats = separate_farmer_seats_enabled(self.opponent_pool.flags)
        self.role_models = {
            'landlord': self.current_model.get_model('landlord'),
            'bidding': self.current_model.get_model('bidding'),
        }
        if separate_farmer_seats:
            self.role_models['landlord_up'] = self.current_model.get_model('landlord_up')
            self.role_models['landlord_down'] = self.current_model.get_model('landlord_down')
        else:
            self.role_models['farmer'] = self.current_model.get_model('farmer')
        if separate_farmer_seats and self.training_position in ('landlord_up', 'landlord_down'):
            self.role_models[self.training_position] = self.current_model.get_model(self.training_position)
        elif separate_farmer_seats and self.training_position == 'farmer_exploiter':
            # The shared farmer exploiter should control both farmer seats
            # during separate-seat training so its trajectories enter the
            # dedicated exploiter replay buffer.
            exploiter_model = self.current_model.get_model('farmer_exploiter')
            self.role_models['landlord_up'] = exploiter_model
            self.role_models['landlord_down'] = exploiter_model
        else:
            self.role_models[self.training_play_group] = self.current_model.get_model(self.training_position)
        if opponent_model is not None:
            if separate_farmer_seats and self.opponent_group == 'farmer':
                self.role_models['landlord_up'] = opponent_model
                self.role_models['landlord_down'] = opponent_model
            else:
                self.role_models[self.opponent_group] = opponent_model
        return self

    def forward(self, position, z, x, training=False, flags=None, debug=False,
                coord_input=None, infoset=None):
        if separate_farmer_seats_enabled(flags) and position in ('landlord_up', 'landlord_down'):
            role_key = position
        else:
            role_key = canonical_position(position)
        model = self.role_models[role_key]
        if hasattr(model, 'act_from_infoset'):
            return model.act_from_infoset(position, infoset, flags=flags)
        return model.forward(
            z, x, training, flags, debug,
            coord_input=coord_input,
        )
