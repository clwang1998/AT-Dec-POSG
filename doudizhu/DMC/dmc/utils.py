"""Actor-side rollout, replay, and batching utilities.

This file diverges from the original DouZero data path in three important
ways:
- trajectories are staged through per-position queues plus prioritized replay
- rollout records extra privileged targets for belief / coordination losses
- self-play can sample league opponents and exploiter policies
"""

import os
import typing
import logging
import threading
import traceback
import random
import numpy as np
from collections import Counter
import time
from DMC.radam.radam import RAdam

import torch
from torch import multiprocessing as mp

from .adversarial import AnnealedOpponentPool, EpisodePolicyModel
from .env_utils import Environment
from .models import (
    canonical_position,
    get_training_positions,
    module_a_enabled,
    module_b_enabled,
    multiply_training_enabled,
    play_group_for_training_position,
    separate_farmer_seats_enabled,
)
from DMC.env import Env

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('dmc')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
Buffers = typing.Dict[str, typing.List[torch.Tensor]]

PLAY_POSITION_INDEX = {"landlord": 31, "landlord_up": 32, "landlord_down": 33}
BID_TYPE_INDEX = {"landlord": 41, "landlord_up": 42, "landlord_down": 43}
BID_TYPE_MAP = {41: "landlord", 42: "landlord_up", 43: "landlord_down"}


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha, beta, epsilon):
        self.capacity = max(1, int(capacity))
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.storage = [None for _ in range(self.capacity)]
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.sample_ids = np.full(self.capacity, -1, dtype=np.int64)
        self.size = 0
        self.next_idx = 0
        self.next_sample_id = 0
        self.max_priority = 1.0
        self.lock = threading.Lock()

    def __len__(self):
        with self.lock:
            return self.size

    def add(self, sample, priority=None):
        with self.lock:
            idx = self.next_idx
            if priority is None:
                priority = self.max_priority
            priority = max(float(priority), self.epsilon)
            self.storage[idx] = sample
            self.priorities[idx] = priority
            self.sample_ids[idx] = self.next_sample_id
            self.next_sample_id += 1
            self.next_idx = (self.next_idx + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size):
        with self.lock:
            if self.size == 0:
                raise RuntimeError('Replay buffer is empty')
            valid_priorities = np.maximum(self.priorities[:self.size], self.epsilon)
            scaled_priorities = valid_priorities ** self.alpha
            priority_sum = scaled_priorities.sum()
            if not np.isfinite(priority_sum) or priority_sum <= 0:
                probabilities = np.full(self.size, 1.0 / self.size, dtype=np.float32)
            else:
                probabilities = scaled_priorities / priority_sum
            # Small buffers still need to provide a learner batch, so sampling
            # falls back to replacement during warmup.
            replace = self.size < batch_size
            indices = np.random.choice(
                self.size, size=batch_size, replace=replace, p=probabilities)
            weights = (self.size * probabilities[indices]) ** (-self.beta)
            weights /= max(weights.max(), 1.0)
            samples = [self.storage[idx] for idx in indices]
            sample_ids = self.sample_ids[indices].copy()
            return samples, sample_ids, weights.astype(np.float32)

    def update_priorities(self, sample_ids, priorities):
        with self.lock:
            if self.size == 0:
                return
            current_ids = self.sample_ids[:self.size]
            for sample_id, priority in zip(sample_ids, priorities):
                matches = np.where(current_ids == int(sample_id))[0]
                if len(matches) == 0:
                    continue
                idx = int(matches[0])
                new_priority = max(float(priority), self.epsilon)
                self.priorities[idx] = new_priority
                self.max_priority = max(self.max_priority, new_priority)


_replay_buffers = {}
_replay_buffers_lock = threading.Lock()


def _as_cpu_tensor(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return torch.tensor(value, device="cpu")


def _get_replay_buffer(position, flags):
    with _replay_buffers_lock:
        replay_buffer = _replay_buffers.get(position)
        if replay_buffer is None:
            replay_buffer = PrioritizedReplayBuffer(
                flags.replay_buffer_size,
                flags.priority_alpha,
                flags.priority_beta,
                flags.priority_epsilon,
            )
            _replay_buffers[position] = replay_buffer
        return replay_buffer

def create_env(flags):
    return Env(flags.objective)

def get_batch(b_queues, position, flags, lock, shutdown_event=None):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with lock:
        b_queue = b_queues[position]
        reader = getattr(b_queue, "_reader", None)
        replay_buffer = _get_replay_buffer(position, flags)
        warmup_size = max(flags.batch_size, flags.replay_warmup_size)

        def next_item():
            while True:
                if shutdown_event is not None and shutdown_event.is_set():
                    return None
                if reader is None:
                    return b_queue.get()
                if reader.poll(0.5):
                    return b_queue.get()

        # Before training a position, first collect enough unrolls from actors to
        # build a minimally stable replay distribution.
        while len(replay_buffer) < warmup_size:
            item = next_item()
            if item is None:
                return None
            replay_buffer.add(item)
        for _ in range(flags.batch_size):
            item = next_item()
            if item is None:
                return None
            replay_buffer.add(item)
        buffer, replay_sample_ids, sampling_weights = replay_buffer.sample(flags.batch_size)
        batch = {
            key: torch.stack([m[key] for m in buffer], dim=1)
            for key in ["done", "episode_return", "target", "obs_z", "obs_x_batch", "obs_type"]
        }
        for key in (
            "belief_primary_target",
            "belief_secondary_target",
            "coord_target",
        ):
            if key in buffer[0]:
                batch[key] = torch.stack([m[key] for m in buffer], dim=1)
        batch["replay_sample_ids"] = torch.from_numpy(replay_sample_ids)
        batch["sampling_weights"] = torch.from_numpy(sampling_weights)
        del buffer
        return batch


def update_priorities(position, sample_ids, priorities):
    replay_buffer = _replay_buffers.get(position)
    if replay_buffer is None:
        return
    replay_buffer.update_priorities(sample_ids, priorities)


def reset_replay_buffer(position, flags):
    with _replay_buffers_lock:
        _replay_buffers[position] = PrioritizedReplayBuffer(
            flags.replay_buffer_size,
            flags.priority_alpha,
            flags.priority_beta,
            flags.priority_epsilon,
        )

def create_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    """
    optimizers = {}
    for position in get_training_positions(flags):
        optimizer = RAdam(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            eps=flags.epsilon)
        optimizers[position] = optimizer
    return optimizers


def _play_target_for_group(group, obs_type, episode_return):
    if group in ('landlord_up', 'landlord_down'):
        return episode_return['play'][group]
    play_group = play_group_for_training_position(group)
    if play_group == 'landlord':
        return episode_return['play']['landlord']
    original_position = 'landlord_up' if int(obs_type) == 32 else 'landlord_down'
    return episode_return['play'][original_position]


def _rollout_record_key(position, flags):
    if separate_farmer_seats_enabled(flags) and position in ('landlord_up', 'landlord_down'):
        return position
    return canonical_position(position)


def _records_training_position(training_position, acting_position, flags):
    if training_position == 'landlord_exploiter':
        return acting_position == 'landlord'
    if training_position == 'farmer_exploiter':
        return acting_position in ('landlord_up', 'landlord_down')
    return _rollout_record_key(training_position, flags) == _rollout_record_key(acting_position, flags)


def _bid_target_for_type(obs_type, episode_return):
    if int(obs_type) == 41:
        return episode_return['bid']['landlord']
    return -episode_return['bid'][BID_TYPE_MAP[int(obs_type)]]


def _compute_farmer_joint_value(farmer_joint_obs, farmer_up_model, farmer_down_model, flags):
    with torch.no_grad():
        up_values = farmer_up_model(
            farmer_joint_obs['landlord_up']['z_batch'],
            farmer_joint_obs['landlord_up']['x_batch'],
            return_value=True,
            flags=flags)['values'].squeeze(-1)
        down_values = farmer_down_model(
            farmer_joint_obs['landlord_down']['z_batch'],
            farmer_joint_obs['landlord_down']['x_batch'],
            return_value=True,
            flags=flags)['values'].squeeze(-1)
    return 0.5 * (torch.max(up_values).item() + torch.max(down_values).item())


def _compute_landlord_value(landlord_obs, landlord_main_model, flags):
    with torch.no_grad():
        landlord_values = landlord_main_model(
            landlord_obs['z_batch'],
            landlord_obs['x_batch'],
            return_value=True,
            flags=flags)['values'].squeeze(-1)
    return torch.max(landlord_values).item()


def _compute_exploiter_bonus(training_position, done, episode_model, env, flags):
    if training_position not in ('landlord_exploiter', 'farmer_exploiter'):
        return 0.0
    if done or not getattr(flags, 'minimax_exploiter_enabled', False):
        return 0.0
    alpha = float(getattr(flags, 'minimax_exploiter_alpha', 0.0))
    if alpha <= 0:
        return 0.0
    gamma = float(getattr(flags, 'minimax_exploiter_gamma', 1.0))
    gamma = min(max(gamma, 0.0), 1.0)
    value_floor = float(getattr(flags, 'minimax_value_floor', 0.0))
    if training_position == 'landlord_exploiter':
        farmer_up_model = episode_model.role_models.get('landlord_up')
        farmer_down_model = episode_model.role_models.get('landlord_down')
        if farmer_up_model is None or farmer_down_model is None:
            return 0.0
        farmer_joint_obs = env.get_joint_farmer_observations()
        farmer_joint_value = _compute_farmer_joint_value(
            farmer_joint_obs, farmer_up_model, farmer_down_model, flags)
        shifted_farmer_value = max(farmer_joint_value - value_floor, 0.0)
        return -alpha * gamma * shifted_farmer_value
    landlord_main_model = episode_model.role_models.get('landlord')
    if landlord_main_model is None:
        return 0.0
    landlord_obs = env.get_value_observation('landlord')
    landlord_value = _compute_landlord_value(
        landlord_obs, landlord_main_model, flags)
    shifted_landlord_value = max(landlord_value - value_floor, 0.0)
    return -alpha * gamma * shifted_landlord_value


def act(i, device, batch_queues, model, flags):
    try:
        base_seed = int(getattr(flags, 'seed', 2026))
        device_offset = 0 if device == 'cpu' else int(device)
        actor_seed = base_seed + device_offset * 1000 + int(i)
        random.seed(actor_seed)
        np.random.seed(actor_seed)
        torch.manual_seed(actor_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(actor_seed)

        T = flags.unroll_length
        training_positions = get_training_positions(flags)
        use_module_a = module_a_enabled(flags)
        use_module_b = module_b_enabled(flags)
        train_bidding = 'bidding' in training_positions
        train_multiply = multiply_training_enabled(flags)
        log.info('Device %s Actor %i started.', str(device), i)

        env = create_env(flags)
        env = Environment(env, device)
        opponent_pool = AnnealedOpponentPool(flags, device)
        episode_model = EpisodePolicyModel(model, opponent_pool)

        # Buffers are organized by training position. By default the two
        # farmer seats share one canonical learner, while the separate-seat
        # ablation records landlord_up / landlord_down independently.
        done_buf = {p: [] for p in training_positions}
        episode_return_buf = {p: [] for p in training_positions}
        target_buf = {p: [] for p in training_positions}
        obs_z_buf = {p: [] for p in training_positions}
        size = {p: 0 for p in training_positions}
        type_buf = {p: [] for p in training_positions}
        obs_x_batch_buf = {p: [] for p in training_positions}
        belief_primary_buf = {p: [] for p in training_positions}
        belief_secondary_buf = {p: [] for p in training_positions}
        coord_target_buf = {p: [] for p in training_positions}
        minimax_bonus_buf = {p: [] for p in training_positions}

        position, obs, env_output = env.initial(episode_model.prepare_next_episode, device, flags=flags)
        bid_obs_buffer = env_output["begin_buf"]["bid_obs_buffer"]
        multiply_obs_buffer = env_output["begin_buf"]["multiply_obs_buffer"]
        while True:
            active_training_position = episode_model.training_position
            # Bidding and play trajectories are generated in the same episode
            # but stored under different learner positions.
            for bid_obs in bid_obs_buffer:
                if not train_bidding:
                    continue
                obs_z_buf["bidding"].append(bid_obs['z_batch'])
                obs_x_batch_buf["bidding"].append(bid_obs["x_batch"])
                type_buf["bidding"].append(BID_TYPE_INDEX[bid_obs["position"]])
                size["bidding"] += 1
            for mul_obs in multiply_obs_buffer:
                if not train_multiply:
                    continue
                if not _records_training_position(
                    active_training_position, mul_obs["position"], flags
                ):
                    continue
                obs_z_buf[active_training_position].append(mul_obs['z_batch'])
                obs_x_batch_buf[active_training_position].append(mul_obs["x_batch"])
                type_buf[active_training_position].append(PLAY_POSITION_INDEX[mul_obs["position"]])
                minimax_bonus_buf[active_training_position].append(0.0)
                size[active_training_position] += 1
            while True:
                with torch.no_grad():
                    agent_output = episode_model.forward(
                        position,
                        obs['z_batch'],
                        obs['x_batch'],
                        flags=flags,
                        infoset=env.get_infoset())
                _action_idx = int(agent_output['action'].cpu().detach().numpy())
                action = obs['legal_actions'][_action_idx]
                record_transition = _records_training_position(
                    active_training_position, position, flags
                )
                if record_transition:
                    # `obs_z` stores the chosen action in the first slice plus
                    # public history after it, matching the learner's value
                    # network input format.
                    obs_z_buf[active_training_position].append(
                        torch.vstack((_cards2tensor(action).unsqueeze(0), env_output['obs_z'])).float())
                    obs_x_batch_buf[active_training_position].append(env_output['obs_x_no_action'].float())
                    if use_module_a:
                        belief_primary_buf[active_training_position].append(env_output['belief_primary_target'].float())
                        belief_secondary_buf[active_training_position].append(env_output['belief_secondary_target'].float())
                    if use_module_b and play_group_for_training_position(active_training_position) == 'farmer':
                        coord_target_buf[active_training_position].append(env_output['coord_target'].float())
                    type_buf[active_training_position].append(PLAY_POSITION_INDEX[position])
                    size[active_training_position] += 1
                completed_training_position = active_training_position
                position, obs, env_output = env.step(
                    action, episode_model.prepare_next_episode, device, flags=flags)
                if record_transition:
                    minimax_bonus_buf[active_training_position].append(
                        _compute_exploiter_bonus(
                            completed_training_position,
                            bool(env_output['done'].item()),
                            episode_model,
                            env,
                            flags))
                if env_output['done']:
                    bid_obs_buffer = env_output["begin_buf"]["bid_obs_buffer"]
                    multiply_obs_buffer = env_output["begin_buf"]["multiply_obs_buffer"]
                    for p in training_positions:
                        diff = size[p] - len(target_buf[p])
                        if diff > 0 and p not in ("bidding", completed_training_position):
                            continue
                        if diff > 0:
                            done_buf[p].extend([False for _ in range(diff-1)])
                            done_buf[p].append(True)
                            offset = len(target_buf[p])
                            step_bonuses = minimax_bonus_buf[p][offset:offset + diff]
                            cumulative_bonus = [0.0 for _ in range(diff)]
                            if (
                                p == completed_training_position
                                and p in ('landlord_exploiter', 'farmer_exploiter')
                            ):
                                # DMC targets are episode-level returns, so we
                                # fold the per-step minimax penalties back into
                                # the remaining undiscounted return prefix.
                                running_bonus = 0.0
                                for bonus_index in range(diff - 1, -1, -1):
                                    running_bonus += step_bonuses[bonus_index]
                                    cumulative_bonus[bonus_index] = running_bonus
                            for index in range(diff):
                                obs_type = type_buf[p][index + offset]
                                if p == "bidding":
                                    episode_return = _bid_target_for_type(
                                        obs_type, env_output['episode_return'])
                                elif p == completed_training_position:
                                    episode_return = _play_target_for_group(
                                        p, obs_type, env_output['episode_return'])
                                else:
                                    continue
                                episode_return_buf[p].append(episode_return)
                                target_buf[p].append(episode_return + cumulative_bonus[index])
                    break
            for p in training_positions:
                # Flush every ready chunk, not just one, so actor throughput is
                # bounded by fixed-length unrolls rather than episode endings.
                while size[p] >= T:
                    batch = {
                        "done": torch.stack([_as_cpu_tensor(ndarr) for ndarr in done_buf[p][:T]]),
                        "episode_return": torch.stack([_as_cpu_tensor(ndarr) for ndarr in episode_return_buf[p][:T]]),
                        "target": torch.stack([_as_cpu_tensor(ndarr) for ndarr in target_buf[p][:T]]),
                        "obs_z": torch.stack([_as_cpu_tensor(ndarr) for ndarr in obs_z_buf[p][:T]]),
                        "obs_x_batch": torch.stack([_as_cpu_tensor(ndarr) for ndarr in obs_x_batch_buf[p][:T]]),
                        "obs_type": torch.stack([_as_cpu_tensor(ndarr) for ndarr in type_buf[p][:T]])
                    }
                    if p != "bidding":
                        if use_module_a:
                            batch["belief_primary_target"] = torch.stack(
                                [_as_cpu_tensor(ndarr) for ndarr in belief_primary_buf[p][:T]])
                            batch["belief_secondary_target"] = torch.stack(
                                [_as_cpu_tensor(ndarr) for ndarr in belief_secondary_buf[p][:T]])
                        if use_module_b and play_group_for_training_position(p) == 'farmer':
                            batch["coord_target"] = torch.stack(
                                [_as_cpu_tensor(ndarr) for ndarr in coord_target_buf[p][:T]])
                    # Actors emit fixed-length unrolls; the learner later
                    # re-samples them with PER rather than consuming them FIFO.
                    batch_queues[p].put(batch)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    obs_x_batch_buf[p] = obs_x_batch_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    type_buf[p] = type_buf[p][T:]
                    belief_primary_buf[p] = belief_primary_buf[p][T:]
                    belief_secondary_buf[p] = belief_secondary_buf[p][T:]
                    coord_target_buf[p] = coord_target_buf[p][T:]
                    minimax_bonus_buf[p] = minimax_bonus_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e

def _cards2tensor(list_cards):
    """
    Convert a list of integers to the tensor
    representation
    See Figure 2 in https://arxiv.org/pdf/2106.06135.pdf
    """
    if len(list_cards) == 0:
        return torch.zeros(54, dtype=torch.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    matrix = np.concatenate((matrix.flatten('F'), jokers))
    matrix = torch.from_numpy(matrix)
    return matrix
