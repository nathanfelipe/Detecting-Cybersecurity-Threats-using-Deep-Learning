# Configuration file for hyperparameters
learning_rate = 0.01
input_size = 7  # Number of features
hidden_size = 64  # Number of neurons in the hidden layer
output_size = 1  # Binary classification (anomalous or not)
num_epochs = 50

# Paths to the data files
train_path = 'labelled_train.csv'
test_path = 'labelled_test.csv'
validation_path = 'labelled_validation.csv'

data_paths = (train_path, test_path, validation_path)

# Optimizer choice (can be 'Adam' or 'SGD')
OPTMZ = 'Adam'
