import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import torch.nn as nn
from torchmetrics import Accuracy


class DataProcessor:
    def __init__(self, train_path, test_path, validation_path):
        self.train_path = train_path
        self.test_path = test_path
        self.validation_path = validation_path
        self.scaler = StandardScaler()

    def load_data(self):
        # Load the datasets
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        validation_df = pd.read_csv(self.validation_path)

        print("Dataset Information:")
        print(train_df.info())

        # Step 1: Separate features and labels
        self.train_features = train_df.drop(columns=['sus_label'])
        self.train_labels = train_df['sus_label']

        self.test_features = test_df.drop(columns=['sus_label'])
        self.test_labels = test_df['sus_label']

        self.validation_features = validation_df.drop(columns=['sus_label'])
        self.validation_labels = validation_df['sus_label']

    def preprocess_data(self):
        # Step 2: Scaling Features
        self.train_features_scaled = self.scaler.fit_transform(self.train_features)
        self.test_features_scaled = self.scaler.transform(self.test_features)
        self.validation_features_scaled = self.scaler.transform(self.validation_features)

    def convert_to_tensors(self):
        # Step 3: Converting to PyTorch Tensors
        self.train_features_tensor = torch.tensor(self.train_features_scaled, dtype=torch.float32)
        self.train_labels_tensor = torch.tensor(self.train_labels.values.reshape(-1, 1), dtype=torch.float32)

        self.test_features_tensor = torch.tensor(self.test_features_scaled, dtype=torch.float32)
        self.test_labels_tensor = torch.tensor(self.test_labels.values.reshape(-1, 1), dtype=torch.float32)

        self.validation_features_tensor = torch.tensor(self.validation_features_scaled, dtype=torch.float32)
        self.validation_labels_tensor = torch.tensor(self.validation_labels.values.reshape(-1, 1), dtype=torch.float32)

        # Display the shapes to confirm
        print("Tensor Shapes:")
        print((self.train_features_tensor.shape, self.train_labels_tensor.shape))
        print((self.test_features_tensor.shape, self.test_labels_tensor.shape))
        print((self.validation_features_tensor.shape, self.validation_labels_tensor.shape))


class HyperparameterTuning:
    def __init__(self, train_features_tensor, train_labels_tensor, test_features_tensor, test_labels_tensor):
        self.train_features_tensor = train_features_tensor
        self.train_labels_tensor = train_labels_tensor
        self.test_features_tensor = test_features_tensor
        self.test_labels_tensor = test_labels_tensor

    def hyperparameter_search(self):
        # Define a list of hyperparameters to try
        learning_rates = [1e-4, 1e-3, 1e-2]
        batch_sizes = [32, 64, 128]
        optimizers = ['SGD', 'Adam']

        epochs = 50
        criterion = nn.CrossEntropyLoss()
        accuracy_metric = Accuracy(task="binary").to('cpu')  # Adjust 'cpu' to 'cuda' if using a GPU

        best_test_acc = 0
        best_hyperparams = {}

        # Iterate over the hyperparameter space
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for opt_name in optimizers:
                    print(f"Training with lr={lr}, batch_size={batch_size}, optimizer={opt_name}")

                    # Reinitialize the model for each hyperparameter set
                    model = nn.Sequential(
                        nn.Linear(in_features=self.train_features_tensor.shape[1], out_features=32),
                        nn.ReLU(),
                        nn.Linear(in_features=32, out_features=16),
                        nn.ReLU(),
                        nn.Linear(in_features=16, out_features=2)
                    )

                    # Select the optimizer
                    if opt_name == 'SGD':
                        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
                    elif opt_name == 'Adam':
                        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

                    # Training loop (simplified for demonstration)
                    for epoch in range(epochs):
                        model.train()
                        optimizer.zero_grad()
                        outputs = model(self.train_features_tensor)
                        loss = criterion(outputs, self.train_labels_tensor.long().view(-1))
                        loss.backward()
                        optimizer.step()

                    # Evaluate on test set after training
                    model.eval()
                    with torch.no_grad():
                        test_outputs = model(self.test_features_tensor)
                        test_preds = torch.argmax(test_outputs, dim=1)
                        test_acc = accuracy_metric(test_preds, self.test_labels_tensor.long().view(-1)).item()

                    print(f"Test Accuracy with lr={lr}, batch_size={batch_size}, optimizer={opt_name}: {test_acc}")

                    # Track the best hyperparameters
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        best_hyperparams = {'lr': lr, 'batch_size': batch_size, 'optimizer': opt_name}

        # Print the best hyperparameters
        print(f"Best Test Accuracy: {best_test_acc}")
        print(f"Best Hyperparameters: {best_hyperparams}")


if __name__ == "__main__":
    # File paths
    train_path = 'labelled_train.csv'
    test_path = 'labelled_test.csv'
    validation_path = 'labelled_validation.csv'

    # Step 1: Data Processing
    data_processor = DataProcessor(train_path, test_path, validation_path)
    data_processor.load_data()
    data_processor.preprocess_data()
    data_processor.convert_to_tensors()

    # Step 2: Hyperparameter Search
    hyper_search = HyperparameterTuning(
        data_processor.train_features_tensor,
        data_processor.train_labels_tensor,
        data_processor.test_features_tensor,
        data_processor.test_labels_tensor
    )
    hyper_search.hyperparameter_search()