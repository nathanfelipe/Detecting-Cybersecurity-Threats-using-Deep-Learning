import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score
from data_loader import DataLoader
from model import AnomalyDetectionModel


class TrainModel:
    def __init__(self, data_paths, input_size, hidden_size, output_size, learning_rate, num_epochs, OPTMZ):
        self.data_loader = DataLoader(*data_paths)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.OPTMZ = OPTMZ
        self.model = AnomalyDetectionModel(input_size, hidden_size, output_size)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = self.select_optimizer()

    def select_optimizer(self):
        # Select the optimizer based on the OPTMZ value
        if self.OPTMZ == 'Adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.OPTMZ == 'SGD':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        else:
            raise ValueError(f"Unknown optimizer: {self.OPTMZ}")

    def train(self):
        train_features, train_labels, test_features, test_labels, validation_features, validation_labels = self.data_loader.load_data()

        train_features_tensor, train_labels_tensor = self.data_loader.preprocess_data(train_features, train_labels)
        test_features_tensor, test_labels_tensor = self.data_loader.preprocess_test_data(test_features, test_labels)
        validation_features_tensor, validation_labels_tensor = self.data_loader.preprocess_test_data(validation_features, validation_labels)

        for epoch in range(self.num_epochs):
            self.model.train()

            # Forward pass
            outputs = self.model(train_features_tensor)
            loss = self.criterion(outputs, train_labels_tensor)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch+1) % 1 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            test_features, test_labels, _, _, _, _ = self.data_loader.load_data()
            test_features_tensor, test_labels_tensor = self.data_loader.preprocess_test_data(test_features, test_labels)
            outputs = self.model(test_features_tensor)
            predicted = (outputs > 0.5).float()
            accuracy = accuracy_score(test_labels_tensor, predicted)
            print(f'Accuracy: {accuracy * 100:.2f}%')
