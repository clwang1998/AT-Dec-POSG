import os
import threading
import time
import timeit
import pprint
from collections import deque
import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn
import torch.nn.functional as F

import DMC.dmc.models
import DMC.env.env
from .adversarial import load_matching_weights, save_opponent_pool_snapshot
from .file_writer import FileWriter
from .models import (
    Model,
    get_training_positions,
    module_a_enabled,
    module_b_enabled,
    module_c_enabled,
    play_group_for_training_position,
    is_exploiter_position,
    main_position_for_training_position,
    separate_farmer_seats_enabled,
)
from .utils import (
    get_batch,
    log,
    create_env,
    create_optimizers,
    act,
    update_priorities,
    reset_replay_buffer,
)

ALL_TRACKED_POSITIONS = (
    'landlord',
    'farmer',
    'landlord_up',
    'landlord_down',
    'landlord_exploiter',
    'farmer_exploiter',
    'bidding',
)

# Track a short moving window per learner position so checkpoint logs stay
# interpretable even though different positions update asynchronously.
mean_episode_return_buf = {p: deque(maxlen=100) for p in ALL_TRACKED_POSITIONS}


def _stat_keys(training_positions):
    stat_keys = [
        'mean_episode_return_landlord',
        'loss_landlord',
    ]
    if 'farmer' in training_positions:
        stat_keys.extend([
            'mean_episode_return_farmer',
            'loss_farmer',
        ])
    else:
        for position in ('landlord_up', 'landlord_down'):
            if position in training_positions:
                stat_keys.extend([
                    f'mean_episode_return_{position}',
                    f'loss_{position}',
                ])
    if 'landlord_exploiter' in training_positions:
        stat_keys.extend([
            'mean_episode_return_landlord_exploiter',
            'loss_landlord_exploiter',
        ])
    if 'farmer_exploiter' in training_positions:
        stat_keys.extend([
            'mean_episode_return_farmer_exploiter',
            'loss_farmer_exploiter',
        ])
    if 'bidding' in training_positions:
        stat_keys.extend([
            'mean_episode_return_bidding',
            'loss_bidding',
        ])
    return stat_keys

def compute_loss(logits, targets, weights=None):
    squared_error = (logits.squeeze(-1) - targets) ** 2
    if weights is not None:
        loss = (squared_error * weights).mean()
    else:
        loss = squared_error.mean()
    return loss

def compute_loss_for_bid(logits, targets, weights=None):
    return compute_loss(logits, targets, weights)


def compute_belief_loss(logits, targets, weights=None):
    per_dim_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none')
    per_sample_loss = per_dim_loss.mean(dim=-1)
    if weights is not None:
        return (per_sample_loss * weights).mean()
    return per_sample_loss.mean()


def _sender_remaining_hand(obs_z):
    current_hand = obs_z[:, 2, :]
    played_action = obs_z[:, 0, :]
    return torch.clamp(current_hand - played_action, min=0.0, max=1.0)


def _episode_return_mask(position, obs_type):
    if position == 'landlord_up':
        return obs_type == 32
    if position == 'landlord_down':
        return obs_type == 33
    play_group = play_group_for_training_position(position)
    if play_group == 'landlord':
        return obs_type == 31
    if play_group == 'farmer':
        return (obs_type == 32) | (obs_type == 33)
    return (obs_type == 41) | (obs_type == 42) | (obs_type == 43)


def _restore_stats(checkpoint_stats, training_positions):
    restored = {
        'mean_episode_return_landlord': checkpoint_stats.get('mean_episode_return_landlord', 0.0),
        'loss_landlord': checkpoint_stats.get('loss_landlord', 0.0),
        'mean_episode_return_landlord_exploiter': checkpoint_stats.get('mean_episode_return_landlord_exploiter', 0.0),
        'loss_landlord_exploiter': checkpoint_stats.get('loss_landlord_exploiter', 0.0),
        'mean_episode_return_farmer_exploiter': checkpoint_stats.get('mean_episode_return_farmer_exploiter', 0.0),
        'loss_farmer_exploiter': checkpoint_stats.get('loss_farmer_exploiter', 0.0),
        'mean_episode_return_bidding': checkpoint_stats.get('mean_episode_return_bidding', 0.0),
        'loss_bidding': checkpoint_stats.get('loss_bidding', 0.0),
    }
    if 'farmer' in training_positions:
        restored['mean_episode_return_farmer'] = checkpoint_stats.get(
            'mean_episode_return_farmer',
            np.mean([checkpoint_stats.get('mean_episode_return_landlord_up', 0.0),
                     checkpoint_stats.get('mean_episode_return_landlord_down', 0.0)]))
        restored['loss_farmer'] = checkpoint_stats.get(
            'loss_farmer',
            np.mean([checkpoint_stats.get('loss_landlord_up', 0.0),
                     checkpoint_stats.get('loss_landlord_down', 0.0)]))
    else:
        for position in ('landlord_up', 'landlord_down'):
            restored[f'mean_episode_return_{position}'] = checkpoint_stats.get(
                f'mean_episode_return_{position}',
                checkpoint_stats.get('mean_episode_return_farmer', 0.0))
            restored[f'loss_{position}'] = checkpoint_stats.get(
                f'loss_{position}',
                checkpoint_stats.get('loss_farmer', 0.0))
    return restored


def _restore_position_frames(checkpoint_frames, training_positions):
    restored = {
        'landlord': checkpoint_frames.get('landlord', 0),
        'landlord_exploiter': checkpoint_frames.get('landlord_exploiter', 0),
        'farmer_exploiter': checkpoint_frames.get('farmer_exploiter', 0),
        'bidding': checkpoint_frames.get('bidding', 0),
    }
    if 'farmer' in training_positions:
        restored['farmer'] = checkpoint_frames.get(
            'farmer',
            checkpoint_frames.get('landlord_up', 0) + checkpoint_frames.get('landlord_down', 0))
    else:
        restored['landlord_up'] = checkpoint_frames.get(
            'landlord_up',
            checkpoint_frames.get('farmer', 0) // 2)
        restored['landlord_down'] = checkpoint_frames.get(
            'landlord_down',
            checkpoint_frames.get('farmer', 0) // 2)
    return restored


def _checkpoint_state_key(position, checkpoint_model_states):
    if position in checkpoint_model_states:
        return position
    if position == 'landlord_exploiter' and 'landlord' in checkpoint_model_states:
        return 'landlord'
    if position in ('landlord_up', 'landlord_down') and 'farmer' in checkpoint_model_states:
        return 'farmer'
    if position == 'farmer_exploiter':
        if 'farmer' in checkpoint_model_states:
            return 'farmer'
        if 'landlord_up' in checkpoint_model_states:
            return 'landlord_up'
        if 'landlord_down' in checkpoint_model_states:
            return 'landlord_down'
    if position == 'farmer':
        if 'landlord_up' in checkpoint_model_states:
            return 'landlord_up'
        if 'landlord_down' in checkpoint_model_states:
            return 'landlord_down'
    return None


def _create_optimizer_for_position(flags, learner_model, position):
    return type(create_optimizers(flags, learner_model)[position])(
        learner_model.parameters(position),
        lr=flags.learning_rate,
        eps=flags.epsilon)


def _refresh_exploiter(position, learner_model, actor_models, optimizers, flags, position_locks):
    if not is_exploiter_position(position):
        return
    # Exploiters are periodically reset from the corresponding main policy.
    # This mirrors league-training best-response refresh instead of letting
    # exploiters drift forever.
    main_position = main_position_for_training_position(position)
    if position == 'farmer_exploiter' and separate_farmer_seats_enabled(flags):
        # Separate-seat training has no shared `farmer` learner. Re-use the
        # canonical landlord_up farmer seat as the reset anchor, matching the
        # checkpoint restore and league snapshot compatibility path.
        main_position = 'landlord_up'
    lock_order = [main_position]
    if position != main_position:
        lock_order.append(position)
    lock_order.sort()
    acquired_locks = []
    try:
        for lock_position in lock_order:
            position_locks[lock_position].acquire()
            acquired_locks.append(position_locks[lock_position])
        learner_model.get_model(position).load_state_dict(
            learner_model.get_model(main_position).state_dict())
        optimizers[position] = _create_optimizer_for_position(flags, learner_model, position)
        reset_replay_buffer(position, flags)
        mean_episode_return_buf[position].clear()
        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(
                learner_model.get_model(position).state_dict())
    finally:
        for lock in reversed(acquired_locks):
            lock.release()

def learn(position, actor_models, model, batch, optimizer, flags, lock):
    """Performs a learning (optimization) step."""
    if flags.training_device != "cpu":
        device = torch.device('cuda:'+str(flags.training_device))
    else:
        device = torch.device('cpu')
    obs_x = batch["obs_x_batch"]
    obs_x = torch.flatten(obs_x, 0, 1).to(device).float()
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    target_unroll = batch['target'].to(device).float()
    target = torch.flatten(target_unroll, 0, 1)
    sampling_weights = batch['sampling_weights'].to(device).float()
    flat_sampling_weights = torch.flatten(
        sampling_weights.unsqueeze(0).expand(target_unroll.shape[0], -1), 0, 1)
    belief_primary_target = None
    belief_secondary_target = None
    coord_target = None
    use_module_a = module_a_enabled(flags) and position != "bidding"
    use_module_b = (
        module_b_enabled(flags)
        and play_group_for_training_position(position) == "farmer"
        and position != "bidding"
    )
    # Play positions can carry extra privileged labels; bidding stays a simple
    # value regression task.
    if use_module_a and 'belief_primary_target' in batch:
        belief_primary_target = torch.flatten(batch['belief_primary_target'].to(device), 0, 1).float()
        belief_secondary_target = torch.flatten(batch['belief_secondary_target'].to(device), 0, 1).float()
    if use_module_b and 'coord_target' in batch:
        coord_target = torch.flatten(batch['coord_target'].to(device), 0, 1).float()
    episode_returns = batch['episode_return'][batch['done'] & _episode_return_mask(position, batch["obs_type"])]
    if len(episode_returns) > 0:
        mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))
    with lock:
        learner_outputs = model(
            obs_z, obs_x, return_value=True,
            flags=flags,
            coord_input=None,
        )
        if position == "bidding":
            loss = compute_loss_for_bid(
                learner_outputs['values'], target, flat_sampling_weights)
        else:
            loss = compute_loss(
                learner_outputs['values'], target, flat_sampling_weights)
            # Module A: hidden-hand belief targets regularize the
            # imperfect-information value network.
            if use_module_a and belief_primary_target is not None and flags.belief_coef > 0:
                primary_belief_loss = compute_belief_loss(
                    learner_outputs['belief_primary_logits'],
                    belief_primary_target,
                    flat_sampling_weights)
                secondary_belief_loss = compute_belief_loss(
                    learner_outputs['belief_secondary_logits'],
                    belief_secondary_target,
                    flat_sampling_weights)
                belief_loss = 0.5 * (primary_belief_loss + secondary_belief_loss)
                loss = loss + flags.belief_coef * belief_loss
            # Module B: only farmer-side learners receive coordination losses.
            if use_module_b and coord_target is not None:
                sender_target = _sender_remaining_hand(obs_z)
                sender_loss = compute_belief_loss(
                    learner_outputs['coord_sender_logits'],
                    sender_target,
                    flat_sampling_weights)
                loss = loss + flags.coord_sender_coef * sender_loss
                receiver_loss = compute_belief_loss(
                    learner_outputs['coord_receiver_logits'],
                    coord_target,
                    flat_sampling_weights)
                loss = loss + flags.coord_receiver_coef * receiver_loss
        # PER priorities are refreshed from the latest value error after each
        # learner update.
        td_errors = (learner_outputs['values'].squeeze(-1) - target).detach()
        sample_priorities = td_errors.view(target_unroll.shape[0], target_unroll.shape[1]).abs().mean(dim=0)
        update_priorities(
            position,
            batch['replay_sample_ids'].cpu().numpy(),
            (sample_priorities + flags.priority_epsilon).cpu().numpy(),
        )
        if len(mean_episode_return_buf[position]) > 0:
            mean_episode_return = torch.mean(
                torch.stack([_r for _r in mean_episode_return_buf[position]])).item()
        else:
            mean_episode_return = 0.0
        stats = {
             'mean_episode_return_'+position: mean_episode_return,
             'loss_'+position: loss.item(),
        }

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()

        # Keep actor copies synchronized with the newest learner weights.
        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(model.state_dict())
        return stats

def train(flags):  
    """
    This is the main funtion for training. It will first
    initilize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then, it will call
    learning function with  multiple threads.
    """
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError("CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`")
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size
    training_positions = get_training_positions(flags)
    separate_farmer_seats = separate_farmer_seats_enabled(flags)

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'

    # Mirror the original DouZero setup:
    # - CPU actors keep shared CPU copies
    # - GPU actors keep device-resident copies on their simulation GPU
    # This keeps learner-to-actor weight sync aligned with the tensors the
    # rollout process actually uses.
    models = {}
    for device in device_iterator:
        actor_model_device = "cpu" if flags.actor_device_cpu else device
        model = Model(
            device=actor_model_device,
            separate_farmer_seats=separate_farmer_seats,
        )
        model.share_memory()
        model.eval()
        models[device] = model

    # Initialize queues
    actor_processes = []
    ctx = mp.get_context('spawn')
    batch_queues = {position: ctx.SimpleQueue() for position in training_positions}

    # Learner model owns the trainable weights and usually sits on the GPU.
    learner_model = Model(
        device=flags.training_device,
        separate_farmer_seats=separate_farmer_seats,
    )

    # Create optimizers
    optimizers = create_optimizers(flags, learner_model)

    # Stat Keys
    stat_keys = _stat_keys(training_positions)
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {position: 0 for position in training_positions}

    # Load models if any
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath, map_location=("cuda:"+str(flags.training_device) if flags.training_device != "cpu" else "cpu")
        )
        checkpoint_model_states = checkpoint_states["model_state_dict"]
        checkpoint_optimizer_states = checkpoint_states["optimizer_state_dict"]
        for k in training_positions:
            state_key = _checkpoint_state_key(k, checkpoint_model_states)
            if state_key is None:
                continue
            load_matching_weights(learner_model.get_model(k), checkpoint_model_states[state_key])
            optimizer_state = checkpoint_optimizer_states.get(k)
            if optimizer_state is None and k == 'landlord_exploiter':
                optimizer_state = checkpoint_optimizer_states.get('landlord')
            if optimizer_state is None and k == 'farmer_exploiter':
                optimizer_state = checkpoint_optimizer_states.get('farmer')
            if optimizer_state is None and k in ('landlord_up', 'landlord_down'):
                optimizer_state = checkpoint_optimizer_states.get('farmer')
            if optimizer_state is None and k == 'farmer':
                optimizer_state = checkpoint_optimizer_states.get('landlord_up')
            if optimizer_state is None and k == 'farmer':
                optimizer_state = checkpoint_optimizer_states.get('landlord_down')
            if optimizer_state is not None:
                optimizers[k].load_state_dict(optimizer_state)
            for device in device_iterator:
                load_matching_weights(models[device].get_model(k), checkpoint_model_states[state_key])
        restored_stats = _restore_stats(checkpoint_states["stats"], training_positions)
        stats = {k: restored_stats.get(k, 0.0) for k in stat_keys}
        frames = checkpoint_states["frames"]
        position_frames = _restore_position_frames(
            checkpoint_states["position_frames"], training_positions)
        log.info(f"Resuming preempted job, current stats:\n{stats}")

    # Starting actor processes
    for device in device_iterator:
        num_actors = flags.num_actors
        for i in range(flags.num_actors):
            # Actors must use the same spawn context as the queues. Forking here
            # after the learner has initialized CUDA can leave child actors
            # wedged before they ever publish rollout batches.
            actor = ctx.Process(
                target=act,
                args=(i, device, batch_queues, models[device], flags))
            # actor.setDaemon(True)
            actor.start()
            actor_processes.append(actor)

    exploiter_reset_frames = {
        position: position_frames.get(position, 0) if is_exploiter_position(position) else 0
        for position in training_positions
    }
    league_lock = threading.Lock()
    shutdown_event = threading.Event()

    def request_shutdown():
        if shutdown_event.is_set():
            return
        shutdown_event.set()

    def batch_and_learn(i, position, batch_lock, position_lock, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats
        while not shutdown_event.is_set() and frames < flags.total_frames:
            try:
                batch = get_batch(
                    batch_queues,
                    position,
                    flags,
                    batch_lock,
                    shutdown_event=shutdown_event,
                )
            except (BrokenPipeError, ConnectionResetError, EOFError, OSError):
                if shutdown_event.is_set():
                    return
                raise
            if batch is None or shutdown_event.is_set():
                return
            _stats = learn(position, models, learner_model.get_model(position), batch, 
                optimizers[position], flags, position_lock)
            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                # `frames` grows per learner update, not per environment step,
                # so a slow-starting replay buffer can keep this at 0 for a
                # while even when actors are already rolling out episodes.
                frames += T * B
                position_frames[position] += T * B
                if (
                    is_exploiter_position(position)
                    and flags.league_exploiter_reset_interval > 0
                    and position_frames[position] - exploiter_reset_frames[position]
                    >= flags.league_exploiter_reset_interval
                ):
                    with league_lock:
                        if position_frames[position] - exploiter_reset_frames[position] >= flags.league_exploiter_reset_interval:
                            _refresh_exploiter(
                                position, learner_model, models, optimizers, flags, position_locks)
                            exploiter_reset_frames[position] = position_frames[position]


    threads = []
    batch_locks = {position: threading.Lock() for position in training_positions}
    position_locks = {position: threading.Lock() for position in training_positions}

    for i in range(flags.num_threads):
        for position in training_positions:
            thread = threading.Thread(
                target=batch_and_learn,
                name='batch-and-learn-%s-%d' % (position, i),
                args=(i, position, batch_locks[position], position_locks[position]),
                daemon=True)
            thread.start()
            threads.append(thread)
    
    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        _models = learner_model.get_models()
        model_state_dict = {k: _models[k].state_dict() for k in _models}
        if not separate_farmer_seats and 'farmer' in model_state_dict:
            model_state_dict['landlord_up'] = model_state_dict['farmer']
            model_state_dict['landlord_down'] = model_state_dict['farmer']
        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': {k: optimizers[k].state_dict() for k in optimizers},
            "stats": stats,
            'flags': vars(flags),
            'frames': frames,
            'position_frames': position_frames
        }, checkpointpath)

        # Save the weights for evaluation purpose
        if separate_farmer_seats:
            eval_model_map = {
                'landlord': 'landlord',
                'landlord_up': 'landlord_up',
                'landlord_down': 'landlord_down',
            }
        else:
            eval_model_map = {
                'landlord': 'landlord',
                'farmer': 'farmer',
                'landlord_up': 'farmer',
                'landlord_down': 'farmer',
            }
        if 'bidding' in training_positions:
            eval_model_map['bidding'] = 'bidding'
        for position, model_key in eval_model_map.items():
            model_weights_dir = os.path.expandvars(os.path.expanduser(
                '%s/%s/%s' % (flags.savedir, flags.xpid, "general_"+position+'_'+str(frames)+'.ckpt')))
            torch.save(learner_model.get_model(model_key).state_dict(), model_weights_dir)
        save_opponent_pool_snapshot(flags, frames, learner_model, stats)

    fps_log = []
    timer = timeit.default_timer
    interrupted = False
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:  
                checkpoint(frames)
                last_checkpoint_time = timer()
            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            position_fps = {k:(position_frames[k]-position_start_frames[k])/(end_time-start_time) for k in position_frames}
            farmer_frames = (
                position_frames.get('farmer', 0)
                + position_frames.get('landlord_up', 0)
                + position_frames.get('landlord_down', 0)
            )
            farmer_fps = (
                position_fps.get('farmer', 0.0)
                + position_fps.get('landlord_up', 0.0)
                + position_fps.get('landlord_down', 0.0)
            )
            log.info('After %i (L:%i F:%i B:%i) frames: @ %.1f fps (avg@ %.1f fps) (L:%.1f F:%.1f B:%.1f) Stats:\n%s',
                     frames,
                     position_frames.get('landlord', 0),
                     farmer_frames,
                     position_frames.get('bidding', 0),
                     fps,
                     fps_avg,
                     position_fps.get('landlord', 0.0),
                     farmer_fps,
                     position_fps.get('bidding', 0.0),
                     pprint.pformat(stats))

    except KeyboardInterrupt:
        interrupted = True
        log.info('Learning interrupted after %d frames. Starting graceful shutdown.', frames)
    else:
        log.info('Learning finished after %d frames.', frames)
    finally:
        request_shutdown()
        for thread in threads:
            thread.join(timeout=5)
        for actor in actor_processes:
            if actor.is_alive():
                actor.terminate()
        for actor in actor_processes:
            actor.join(timeout=5)

    if interrupted:
        plogger.close()
        return

    checkpoint(frames)
    plogger.close()
