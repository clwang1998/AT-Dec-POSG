"""
Here, we wrap the original environment to make it easier
to use. When a game is finished, instead of mannualy reseting
the environment, we do it automatically.

This wrapper also forwards optional training-only targets such as
hidden-hand labels and coordination labels from the environment to the
learner. Actors consume them while generating trajectories; the real
game state still comes from `DMC.env`.
"""
import numpy as np
import torch


OPTIONAL_TRAINING_KEYS = (
    'belief_primary_target',
    'belief_secondary_target',
    'coord_target',
)

def _actor_device(device):
    if device != "cpu":
        device = 'cuda:' + str(device)
    return torch.device(device)

def _format_value_observation(obs, device):
    device = _actor_device(device)
    return {
        'x_batch': torch.from_numpy(obs['x_batch']).to(device),
        'z_batch': torch.from_numpy(obs['z_batch']).to(device),
    }

def _format_observation(obs, device):
    """
    A utility function to process observations and
    move them to CUDA.
    """
    position = obs['position']
    device = _actor_device(device)
    x_batch = torch.from_numpy(obs['x_batch']).to(device)
    z_batch = torch.from_numpy(obs['z_batch']).to(device)
    x_no_action = torch.from_numpy(obs['x_no_action'])
    z = torch.from_numpy(obs['z'])
    # These keys do not affect the game transition itself; they are
    # auxiliary supervision generated from privileged information.
    training_targets = {}
    for key in OPTIONAL_TRAINING_KEYS:
        if key in obs:
            training_targets[key] = torch.from_numpy(obs[key])
    obs = {'x_batch': x_batch,
           'z_batch': z_batch,
           'legal_actions': obs['legal_actions'],
           }
    return position, obs, x_no_action, z, training_targets

class Environment:
    def __init__(self, env, device):
        """ Initialzie this environment wrapper
        """
        self.env = env
        self.device = device
        self.episode_return = None

    def initial(self, model, device, flags=None):
        if callable(model):
            model = model()
        obs, buf = self.env.reset(model, device, flags=flags)
        initial_position, initial_obs, x_no_action, z, training_targets = _format_observation(obs, self.device)
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        initial_done = torch.ones(1, 1, dtype=torch.bool)
        env_output = dict(
            done=initial_done,
            episode_return=self.episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
        )
        env_output.update(training_targets)
        if buf is None:
            return initial_position, initial_obs, env_output
        else:
            env_output['begin_buf'] = buf
            return initial_position, initial_obs, env_output

    def step(self, action, model, device, flags=None):
        obs, reward, done, _ = self.env.step(action)

        self.episode_return = reward
        episode_return = self.episode_return
        buf = None
        if done:
            if callable(model):
                model = model()
            obs, buf = self.env.reset(model, device, flags=flags)
            self.episode_return = torch.zeros(1, 1)

        position, obs, x_no_action, z, training_targets = _format_observation(obs, self.device)
        # reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        env_output = dict(
            done=done,
            episode_return=episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
        )
        env_output.update(training_targets)

        if buf is None:
            return position, obs, env_output
        else:
            env_output['begin_buf'] = buf
            return position, obs, env_output

    def close(self):
        self.env.close()

    def get_infoset(self):
        return self.env.infoset

    def get_joint_farmer_observations(self):
        joint_obs = self.env.get_joint_farmer_obs()
        return {
            position: _format_value_observation(obs, self.device)
            for position, obs in joint_obs.items()
        }

    def get_value_observation(self, position):
        obs = self.env.get_obs_for_position(position)
        return _format_value_observation(obs, self.device)
