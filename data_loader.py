import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


class DataLoader:
    def __init__(self, train_path, test_path, validation_path):
        self.train_path = train_path
        self.test_path = test_path
        self.validation_path = validation_path
        self.scaler = StandardScaler()

    def load_data(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        validation_df = pd.read_csv(self.validation_path)

        train_features = train_df.drop(columns=['sus_label'])
        train_labels = train_df['sus_label']

        test_features = test_df.drop(columns=['sus_label'])
        test_labels = test_df['sus_label']

        validation_features = validation_df.drop(columns=['sus_label'])
        validation_labels = validation_df['sus_label']

        return train_features, train_labels, test_features, test_labels, validation_features, validation_labels

    def preprocess_data(self, features, labels):
        features_scaled = self.scaler.fit_transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        labels_tensor = torch.tensor(labels.values.reshape(-1, 1), dtype=torch.float32)
        return features_tensor, labels_tensor

    def preprocess_test_data(self, features, labels):
        features_scaled = self.scaler.transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        labels_tensor = torch.tensor(labels.values.reshape(-1, 1), dtype=torch.float32)
        return features_tensor, labels_tensor
