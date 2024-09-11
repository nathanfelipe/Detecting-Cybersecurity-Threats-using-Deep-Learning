import torch.nn as nn


class AnomalyDetectionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AnomalyDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
