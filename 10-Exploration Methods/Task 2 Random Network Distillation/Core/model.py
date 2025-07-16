# model.py
# This module defines the neural networks used in the RND+PPO agent.
# Students are expected to implement the TargetModel and PredictorModel architectures and their initialization.

from abc import ABC
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical


def conv_shape(input, kernel_size, stride, padding=0):
    return (input + 2 * padding - kernel_size) // stride + 1


# === Policy Network ===
class PolicyModel(nn.Module, ABC):
    def __init__(self, state_shape, n_actions):
        super(PolicyModel, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions

        c, w, h = state_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        flatten_size = 32 * conv_shape(w, 3, 1, 1) * conv_shape(h, 3, 1, 1)

        self.fc1 = nn.Linear(flatten_size, 256)
        self.gru = nn.GRUCell(256, 256)

        self.extra_value_fc = nn.Linear(256, 256)
        self.extra_policy_fc = nn.Linear(256, 256)

        self.policy = nn.Linear(256, self.n_actions)
        self.int_value = nn.Linear(256, 1)
        self.ext_value = nn.Linear(256, 1)

        # Orthogonal initialization
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

    def forward(self, inputs, hidden_state):
        if inputs.ndim == 5:
            inputs = inputs.squeeze(1)

        x = inputs / 255.
        x = self.conv(x)
        x = F.relu(self.fc1(x))
        h = self.gru(x, hidden_state)

        x_v = h + F.relu(self.extra_value_fc(h))
        x_pi = h + F.relu(self.extra_policy_fc(h))

        int_value = self.int_value(x_v)
        ext_value = self.ext_value(x_v)

        policy_logits = self.policy(x_pi)
        probs = F.softmax(policy_logits, dim=1)
        dist = Categorical(probs)

        return dist, int_value, ext_value, probs, h


# === Target Model ===
class TargetModel(nn.Module, ABC):
    def __init__(self, state_shape):
        super(TargetModel, self).__init__()
        # === TODO: Implement Target Model architecture ===
        # Define 3 convolutional layers followed by a fully connected layer.
        # The output should be a 512-dimensional encoded feature vector.
        # Example:
        # self.conv1 = nn.Conv2d(...)
        # self.conv2 = nn.Conv2d(...)
        # self.conv3 = nn.Conv2d(...)
        # self.encoded_features = nn.Linear(...)
        c, w, h = state_shape
        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.encoded_features = nn.Linear(128 * conv_shape(w, 3, 1, 1) * conv_shape(h, 3, 1, 1), 512)
        
        self._init_weights()  # Call this after defining layers

    def _init_weights(self):
        # === TODO: Initialize all layers with orthogonal weights ===
        # For most layers use gain=np.sqrt(2).
        # Call orthogonal_ on each conv and linear layer.
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

    def forward(self, inputs):
        # === TODO: Implement forward pass ===
        # Normalize input, pass through conv layers, flatten, and return encoded features.
        x = inputs / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.encoded_features(x)

# === Predictor Model ===
class PredictorModel(nn.Module, ABC):
    def __init__(self, state_shape):
        super(PredictorModel, self).__init__()
        # === TODO: Implement Predictor Model architecture ===
        # It should match the target model up to encoded features,
        # and then include 1 or 2 additional linear layers.
        # End with a layer that outputs a 512-dim feature vector (same as TargetModel).
        c, w, h = state_shape
        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.encoded_features = nn.Linear(128 * conv_shape(w, 3, 1, 1) * conv_shape(h, 3, 1, 1), 512)
        self.extra_fc = nn.Linear(512, 512)
        
        self._init_weights()  # Call this after defining layers

    def _init_weights(self):
        # === TODO: Initialize all layers with orthogonal weights ===
        # Use gain=np.sqrt(2) for hidden layers.
        # Use gain=np.sqrt(0.01) if you want to slow learning on final output layer (optional).
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                gain = np.sqrt(2) if layer is not self.encoded_features else np.sqrt(0.01)
                nn.init.orthogonal_(layer.weight, gain=gain)
                layer.bias.data.zero_()

    def forward(self, inputs):
        # === TODO: Implement forward pass ===
        # Normalize input, pass through conv layers and extra FC layers, then return final encoded vector.
        x = inputs / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.encoded_features(x))
        return self.extra_fc(x)
