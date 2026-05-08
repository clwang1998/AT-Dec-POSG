"""
This file includes the torch models. We wrap the three
models into one class for convenience.

The original DouZero-style LSTM models are kept for compatibility
(`LandlordLstmModel` / `FarmerLstmModel`). The main training path in this
project uses `GeneralModel`, which adds belief prediction, coordination
heads, and an explicit bidding model.
"""

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

POSITION_TO_GROUP = {
    'landlord': 'landlord',
    'landlord_up': 'farmer',
    'landlord_down': 'farmer',
    'farmer': 'farmer',
    'bidding': 'bidding',
}

FARMER_SEAT_POSITIONS = ('landlord_up', 'landlord_down')
MAIN_TRAINING_POSITIONS = ('landlord', 'farmer')
EXPLOITER_TRAINING_POSITIONS = ('landlord_exploiter', 'farmer_exploiter')
TRAINING_POSITIONS = MAIN_TRAINING_POSITIONS + EXPLOITER_TRAINING_POSITIONS + ('bidding',)
TRAINING_POSITION_TO_PLAY_GROUP = {
    'landlord': 'landlord',
    'farmer': 'farmer',
    'landlord_up': 'farmer',
    'landlord_down': 'farmer',
    'landlord_exploiter': 'landlord',
    'farmer_exploiter': 'farmer',
    'bidding': 'bidding',
}
HIDDEN_PLAYER_BELIEF_DIM = 54
PLAY_BELIEF_DIM = HIDDEN_PLAYER_BELIEF_DIM * 2
COORD_EMBED_DIM = 64


def _flag_enabled(flags, name, default_when_missing):
    if flags is None:
        return default_when_missing
    return bool(getattr(flags, name, default_when_missing))


def module_a_enabled(flags):
    return _flag_enabled(flags, 'enable_module_a', True)


def module_b_enabled(flags):
    return _flag_enabled(flags, 'enable_module_b', True)


def module_c_enabled(flags):
    return _flag_enabled(flags, 'enable_module_c', True)


def separate_farmer_seats_enabled(flags):
    return _flag_enabled(flags, 'separate_farmer_seats', False)


def bidding_training_enabled(flags):
    return _flag_enabled(flags, 'train_bidding', True)


def multiply_training_enabled(flags):
    return _flag_enabled(flags, 'train_multiply', False)


def get_main_training_positions(flags=None):
    if separate_farmer_seats_enabled(flags):
        return ('landlord',) + FARMER_SEAT_POSITIONS
    return MAIN_TRAINING_POSITIONS


def get_exploiter_training_positions(flags=None):
    if not module_c_enabled(flags):
        return ()
    return EXPLOITER_TRAINING_POSITIONS


def get_training_positions(flags=None):
    positions = list(get_main_training_positions(flags))
    positions.extend(get_exploiter_training_positions(flags))
    if bidding_training_enabled(flags):
        positions.append('bidding')
    return tuple(positions)


def canonical_position(position):
    return POSITION_TO_GROUP.get(position, position)


def play_group_for_training_position(position):
    return TRAINING_POSITION_TO_PLAY_GROUP.get(position, position)


def is_exploiter_position(position):
    return position in EXPLOITER_TRAINING_POSITIONS


def main_position_for_training_position(position):
    if position == 'landlord_exploiter':
        return 'landlord'
    if position == 'farmer_exploiter':
        return 'farmer'
    return position


class LandlordLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(373 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

class FarmerLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(484 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

class LandlordLstmNewModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(373 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

class FarmerLstmNewModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(484 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

class GeneralModel1(nn.Module):
    def __init__(self):
        super().__init__()
        # input: B * 32 * 57
        # self.lstm = nn.LSTM(162, 512, batch_first=True)
        self.conv_z_1 = torch.nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1,57)),  # B * 1 * 64 * 32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        # Squeeze(-1) B * 64 * 16
        self.conv_z_2 = torch.nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=(5,), padding=2),  # 128 * 16
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
        )
        self.conv_z_3 = torch.nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=(3,), padding=1), # 256 * 8
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),

        )
        self.conv_z_4 = torch.nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=(3,), padding=1), # 512 * 4
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),

        )

        self.dense1 = nn.Linear(519 + 1024, 1024)
        self.dense2 = nn.Linear(1024, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None, debug=False):
        z = z.unsqueeze(1)
        z = self.conv_z_1(z)
        z = z.squeeze(-1)
        z = torch.max_pool1d(z, 2)
        z = self.conv_z_2(z)
        z = torch.max_pool1d(z, 2)
        z = self.conv_z_3(z)
        z = torch.max_pool1d(z, 2)
        z = self.conv_z_4(z)
        z = torch.max_pool1d(z, 2)
        z = z.flatten(1,2)
        x = torch.cat([z,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action, max_value=torch.max(x))


# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=(3,),
                               stride=(stride,), padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=(3,),
                               stride=(1,), padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes,
                          kernel_size=(1,), stride=(stride,), bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GeneralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 80
        self.history_lstm = nn.LSTM(54, 128, batch_first=True)
        # input: B * 40 * 54
        self.conv1 = nn.Conv1d(40, 80, kernel_size=(3,),
                               stride=(2,), padding=1, bias=False)

        self.bn1 = nn.BatchNorm1d(80)
        # `coord_input` is now treated as a learner-side coordination context:
        # actors do not consume it for action selection, but the learner can
        # still use it as privileged training-time context.
        self.coord_encoder = nn.Linear(COORD_EMBED_DIM, 64)

        self.layer1 = self._make_layer(BasicBlock, 80, 2, stride=2)
        self.layer2 = self._make_layer(BasicBlock, 160, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 320, 2, stride=2)
        self.linear1 = nn.Linear(320 * BasicBlock.expansion * 4 + 15 * 2 + 128 + 64, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        # Belief heads predict the hidden cards of the two unseen players,
        # then feed the inferred belief back into the value head.
        self.belief_encoder = nn.Linear(PLAY_BELIEF_DIM, 128)
        self.value_head = nn.Linear(256 + 128, 1)
        self.belief_primary_head = nn.Linear(256, HIDDEN_PLAYER_BELIEF_DIM)
        self.belief_secondary_head = nn.Linear(256, HIDDEN_PLAYER_BELIEF_DIM)
        self.coord_head = nn.Linear(256, COORD_EMBED_DIM)
        self.coord_sender_head = nn.Linear(COORD_EMBED_DIM, 54)
        self.coord_receiver_head = nn.Linear(256, 54)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _encode(self, z, x, coord_input=None, flags=None):
        # The first slice of `z` stores the candidate action; later slices
        # store action history. The history LSTM summarizes long-term public
        # play context, while the residual stack focuses on local card
        # pattern extraction.
        history = z[:, 8:, :]
        history = torch.flip(history, dims=[1])
        history_out, _ = self.history_lstm(history)
        history_out = history_out[:, -1, :]
        if coord_input is None or not module_b_enabled(flags):
            coord_features = torch.zeros(
                z.shape[0], 64, device=z.device, dtype=z.dtype)
        else:
            coord_features = F.leaky_relu_(self.coord_encoder(coord_input))
        out = F.relu(self.bn1(self.conv1(z)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.flatten(1,2)
        out = torch.cat([x, x, history_out, coord_features, out], dim=-1)
        out = F.leaky_relu_(self.linear1(out))
        out = F.leaky_relu_(self.linear2(out))
        out = F.leaky_relu_(self.linear3(out))
        return out

    def forward(
        self,
        z,
        x,
        return_value=False,
        flags=None,
        debug=False,
        coord_input=None,
    ):
        features = self._encode(z, x, coord_input=coord_input, flags=flags)
        if module_a_enabled(flags):
            belief_primary_logits = self.belief_primary_head(features)
            belief_secondary_logits = self.belief_secondary_head(features)
            belief_probs = torch.cat([
                torch.sigmoid(belief_primary_logits),
                torch.sigmoid(belief_secondary_logits),
            ], dim=-1)
            belief_features = F.leaky_relu_(self.belief_encoder(belief_probs))
        else:
            belief_primary_logits = torch.zeros(
                features.shape[0], HIDDEN_PLAYER_BELIEF_DIM,
                device=features.device, dtype=features.dtype)
            belief_secondary_logits = torch.zeros_like(belief_primary_logits)
            belief_features = torch.zeros(
                features.shape[0], 128,
                device=features.device, dtype=features.dtype)
        value_inputs = torch.cat([features, belief_features], dim=-1)
        values = self.value_head(value_inputs)
        if module_b_enabled(flags):
            coord_embedding = torch.tanh(self.coord_head(features))
            coord_sender_logits = self.coord_sender_head(coord_embedding)
            coord_receiver_logits = self.coord_receiver_head(features)
        else:
            coord_embedding = torch.zeros(
                features.shape[0], COORD_EMBED_DIM,
                device=features.device, dtype=features.dtype)
            coord_sender_logits = torch.zeros(
                features.shape[0], 54,
                device=features.device, dtype=features.dtype)
            coord_receiver_logits = torch.zeros_like(coord_sender_logits)
        outputs = dict(
            values=values,
            belief_primary_logits=belief_primary_logits,
            belief_secondary_logits=belief_secondary_logits,
            coord_embedding=coord_embedding,
            coord_sender_logits=coord_sender_logits,
            coord_receiver_logits=coord_receiver_logits,
        )
        if return_value:
            return outputs
        else:
            # During acting, one row corresponds to one legal action, so
            # argmax over rows chooses the action with the best predicted
            # return under the current model.
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(values.shape[0], (1,))[0]
            else:
                action = torch.argmax(values,dim=0)[0]
            return dict(
                action=action,
                max_value=torch.max(values),
            )


class ResnetModel(nn.Module):
    """
    Compatibility model for `DouZero_For_HLDDZ_FullAuto` ResNet checkpoints.
    """

    def __init__(self):
        super().__init__()
        self.in_planes = 80
        self.conv1 = nn.Conv1d(
            40, 80, kernel_size=(3,), stride=(2,), padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(80)
        self.layer1 = self._make_layer(BasicBlock, 80, 2, stride=2)
        self.layer2 = self._make_layer(BasicBlock, 160, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 320, 2, stride=2)
        self.linear1 = nn.Linear(320 * BasicBlock.expansion * 4 + 15 * 4, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(
        self,
        z,
        x,
        return_value=False,
        flags=None,
        debug=False,
        coord_input=None,
    ):
        out = F.relu(self.bn1(self.conv1(z)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.flatten(1, 2)
        out = torch.cat([x, x, x, x, out], dim=-1)
        out = F.leaky_relu_(self.linear1(out))
        out = F.leaky_relu_(self.linear2(out))
        out = F.leaky_relu_(self.linear3(out))
        out = F.leaky_relu_(self.linear4(out))
        if return_value:
            return dict(values=out)
        if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
            action = torch.randint(out.shape[0], (1,))[0]
        else:
            action = torch.argmax(out, dim=0)[0]
        return dict(action=action, max_value=torch.max(out))





class BidModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.dense1 = nn.Linear(114, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(
        self,
        z,
        x,
        return_value=False,
        flags=None,
        debug=False,
        coord_input=None,
    ):
        x = self.dense1(x)
        x = F.leaky_relu(x)
        # x = F.relu(x)
        x = self.dense2(x)
        x = F.leaky_relu(x)
        # x = F.relu(x)
        x = self.dense3(x)
        x = F.leaky_relu(x)
        # x = F.relu(x)
        x = self.dense4(x)
        x = F.leaky_relu(x)
        # x = F.relu(x)
        x = self.dense5(x)
        # x = F.relu(x)
        x = F.leaky_relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action, max_value=torch.max(x))


# Model dict is only used in evaluation but not training
model_dict = {}
model_dict['landlord'] = LandlordLstmModel
model_dict['landlord_up'] = FarmerLstmModel
model_dict['landlord_down'] = FarmerLstmModel
model_dict_new = {}
model_dict_new['landlord'] = GeneralModel
model_dict_new['landlord_up'] = GeneralModel
model_dict_new['landlord_down'] = GeneralModel
model_dict_new['farmer'] = GeneralModel
model_dict_new['bidding'] = BidModel
model_dict_resnet = {}
model_dict_resnet['landlord'] = ResnetModel
model_dict_resnet['landlord_up'] = ResnetModel
model_dict_resnet['landlord_down'] = ResnetModel
model_dict_resnet['farmer'] = ResnetModel
model_dict_resnet['bidding'] = BidModel
model_dict_lstm = {}
model_dict_lstm['landlord'] = GeneralModel
model_dict_lstm['landlord_up'] = GeneralModel
model_dict_lstm['landlord_down'] = GeneralModel
model_dict_lstm['farmer'] = GeneralModel

canonical_model_dict = {
    'landlord': GeneralModel,
    'farmer': GeneralModel,
    'bidding': BidModel,
}

class General_Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        # model = GeneralModel().to(torch.device(device))
        self.models['landlord'] = GeneralModel1().to(torch.device(device))
        self.models['landlord_up'] = GeneralModel1().to(torch.device(device))
        self.models['landlord_down'] = GeneralModel1().to(torch.device(device))
        self.models['bidding'] = BidModel().to(torch.device(device))

    def forward(self, position, z, x, training=False, flags=None, debug=False):
        model = self.models[position]
        return model.forward(z, x, training, flags, debug)

    def share_memory(self):
        self.models['landlord'].share_memory()
        self.models['landlord_up'].share_memory()
        self.models['landlord_down'].share_memory()
        self.models['bidding'].share_memory()

    def eval(self):
        self.models['landlord'].eval()
        self.models['landlord_up'].eval()
        self.models['landlord_down'].eval()
        self.models['bidding'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models

class OldModel:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.models['landlord'] = LandlordLstmModel().to(torch.device(device))
        self.models['landlord_up'] = FarmerLstmModel().to(torch.device(device))
        self.models['landlord_down'] = FarmerLstmModel().to(torch.device(device))

    def forward(self, position, z, x, training=False, flags=None):
        model = self.models[position]
        return model.forward(z, x, training, flags)

    def share_memory(self):
        self.models['landlord'].share_memory()
        self.models['landlord_up'].share_memory()
        self.models['landlord_down'].share_memory()

    def eval(self):
        self.models['landlord'].eval()
        self.models['landlord_up'].eval()
        self.models['landlord_down'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models


class Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0, separate_farmer_seats=False):
        if not device == "cpu":
            device = 'cuda:' + str(device)
        device = torch.device(device)
        self.separate_farmer_seats = bool(separate_farmer_seats)
        landlord_model = GeneralModel().to(device)
        landlord_up_model = GeneralModel().to(device)
        landlord_down_model = GeneralModel().to(device)
        landlord_exploiter_model = GeneralModel().to(device)
        farmer_exploiter_model = GeneralModel().to(device)
        # Exploiters start as copies of the main policies and are later
        # reset from them periodically during league training.
        landlord_exploiter_model.load_state_dict(landlord_model.state_dict())
        farmer_exploiter_model.load_state_dict(landlord_up_model.state_dict())
        bidding_model = BidModel().to(device)
        canonical_models = {
            'landlord': landlord_model,
            'landlord_exploiter': landlord_exploiter_model,
            'farmer_exploiter': farmer_exploiter_model,
            'bidding': bidding_model,
        }
        models = {
            'landlord': landlord_model,
            'landlord_exploiter': landlord_exploiter_model,
            'farmer_exploiter': farmer_exploiter_model,
            'bidding': bidding_model,
        }
        if self.separate_farmer_seats:
            canonical_models['landlord_up'] = landlord_up_model
            canonical_models['landlord_down'] = landlord_down_model
            models['landlord_up'] = landlord_up_model
            models['landlord_down'] = landlord_down_model
        else:
            canonical_models['farmer'] = landlord_up_model
            # Both farmers share parameters in the canonical play model.
            models['landlord_up'] = landlord_up_model
            models['landlord_down'] = landlord_up_model
            models['farmer'] = landlord_up_model
        self.canonical_models = canonical_models
        self.models = models

    def forward(
        self,
        position,
        z,
        x,
        training=False,
        flags=None,
        debug=False,
        coord_input=None,
    ):
        model = self.get_model(position)
        return model.forward(
            z, x, training, flags, debug,
            coord_input=coord_input,
        )

    def share_memory(self):
        for model in self.canonical_models.values():
            model.share_memory()

    def eval(self):
        for model in self.canonical_models.values():
            model.eval()

    def parameters(self, position):
        return self.get_model(position).parameters()

    def get_model(self, position):
        if position in self.models:
            return self.models[position]
        return self.models[canonical_position(position)]

    def get_models(self):
        return self.canonical_models
