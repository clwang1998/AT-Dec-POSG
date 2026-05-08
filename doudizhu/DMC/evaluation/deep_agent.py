import copy
import os
from types import SimpleNamespace

import numpy as np
import torch

from DMC.env.env import _get_obs_for_bid, get_obs


BIDDING_POSITIONS = ('first', 'second', 'third')
POSITION_TO_BID_PLAYER_ID = {
    'first': 0,
    'second': 1,
    'third': 2,
}


def _infer_model_type(model_state, model_path=None):
    model_keys = tuple(model_state.keys())
    if any(key.startswith('history_lstm.') for key in model_keys):
        return 'general'
    if any(key.startswith('conv1.') for key in model_keys) and any(
        key.startswith('layer1.') for key in model_keys
    ):
        return 'resnet'
    if any(key.startswith('lstm.') for key in model_keys):
        return 'old'
    if any(key.startswith('dense1.') for key in model_keys):
        return 'bidding'
    if model_path and 'resnet' in os.path.basename(model_path).lower():
        return 'resnet'
    return 'old'


def _load_model(position, model_path):
    from DMC.dmc.models import BidModel, model_dict, model_dict_new, model_dict_resnet

    if torch.cuda.is_available():
        pretrained = torch.load(model_path, map_location='cuda:0')
    else:
        pretrained = torch.load(model_path, map_location='cpu')

    model_type = _infer_model_type(pretrained, model_path=model_path)
    if position in BIDDING_POSITIONS:
        model = BidModel()
    else:
        if model_type == 'general':
            model = model_dict_new[position]()
        elif model_type == 'resnet':
            model = model_dict_resnet[position]()
        else:
            model = model_dict[position]()

    model_state_dict = model.state_dict()
    filtered_state = {
        key: value for key, value in pretrained.items()
        if key in model_state_dict and model_state_dict[key].shape == value.shape
    }
    model_state_dict.update(filtered_state)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


def _load_model_flags(model_path):
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), 'model.tar')
    if not os.path.exists(checkpoint_path):
        return None
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception:
        return None
    checkpoint_flags = checkpoint.get('flags')
    if not isinstance(checkpoint_flags, dict):
        return None
    return SimpleNamespace(**checkpoint_flags)


def _reconstruct_bid_matrix(infoset):
    bid_matrix = np.full((4, 3), -1, dtype=np.int8)
    position_to_index = {
        'first': 0,
        'second': 1,
        'third': 2,
    }
    for step_index, action_record in enumerate(getattr(infoset, 'bid_action_seq', []) or []):
        if step_index >= 4:
            break
        acting_position, action = action_record
        bid_matrix[step_index] = np.array([0, 0, 0], dtype=np.int8)
        if len(action) > 0 and int(action[0]) == 1:
            bid_matrix[step_index, position_to_index[acting_position]] = 1
    return bid_matrix


def _reconstruct_position_mapping(bid_action_seq):
    bid_info = [-1, -1, -1, -1]
    for step_index, action_record in enumerate(bid_action_seq or []):
        if step_index >= 4:
            break
        _, action = action_record
        bid_info[step_index] = int(action[0])

    position = ['landlord', 'landlord_down', 'landlord_up']
    if bid_info[:3] == [0, 1, 0]:
        position = ['landlord_up', 'landlord', 'landlord_down']
    elif bid_info[:3] == [0, 0, 1]:
        position = ['landlord_down', 'landlord_up', 'landlord']
    elif bid_info[3] != -1:
        if bid_info[3] == 1:
            for seat_index in range(3):
                if bid_info[seat_index] == 1:
                    position[seat_index] = 'landlord'
                    position[(seat_index + 1) % 3] = 'landlord_down'
                    position[(seat_index + 2) % 3] = 'landlord_up'
                    break
        else:
            for seat_index in range(2, -1, -1):
                if bid_info[seat_index] == 1:
                    position[seat_index] = 'landlord'
                    position[(seat_index - 1) % 3] = 'landlord_up'
                    position[(seat_index + 1) % 3] = 'landlord_down'
                    break
    return {
        'first': position[0],
        'second': position[1],
        'third': position[2],
    }


def _patch_play_infoset(infoset):
    patched_infoset = copy.deepcopy(infoset)
    bid_matrix = _reconstruct_bid_matrix(infoset)
    position_mapping = _reconstruct_position_mapping(getattr(infoset, 'bid_action_seq', []) or [])
    play_to_seat = {
        play_position: seat_name
        for seat_name, play_position in position_mapping.items()
    }
    current_seat = play_to_seat[infoset.player_position]
    seat_index = POSITION_TO_BID_PLAYER_ID[current_seat]
    patched_infoset.bid_info = bid_matrix[:, [(seat_index - 1) % 3, seat_index, (seat_index + 1) % 3]]
    if not hasattr(patched_infoset, 'multiply_info'):
        patched_infoset.multiply_info = [1, 0, 0]
    return patched_infoset


class DeepAgent:

    def __init__(self, position, model_path):
        self.position = position
        self.model = _load_model(position, model_path)
        self.model_flags = _load_model_flags(model_path)
        self.model_type = _infer_model_type(self.model.state_dict(), model_path=model_path)

    def _bid_obs(self, infoset):
        bid_info = _reconstruct_bid_matrix(infoset)
        return _get_obs_for_bid(
            POSITION_TO_BID_PLAYER_ID[infoset.player_position],
            bid_info,
            infoset.player_hand_cards,
        )

    def _play_obs(self, infoset):
        return get_obs(_patch_play_infoset(infoset), self.model_type)

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        if infoset.player_position in BIDDING_POSITIONS:
            obs = self._bid_obs(infoset)
        else:
            obs = self._play_obs(infoset)

        z_batch = torch.from_numpy(np.asarray(obs['z_batch'])).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available():
            z_batch = z_batch.cuda()
            x_batch = x_batch.cuda()

        y_pred = self.model.forward(
            z_batch,
            x_batch,
            return_value=True,
            flags=self.model_flags,
        )['values']
        y_pred = y_pred.detach().cpu().numpy()

        best_action_index = int(np.argmax(y_pred, axis=0)[0])
        return infoset.legal_actions[best_action_index]
