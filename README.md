# Cybersecurity Threat Detection using Deep Learning

This project is part of the **"Detecting Cybersecurity Threats using Deep Learning"** course on DataCamp. The goal of this project is to implement a machine learning pipeline using PyTorch to detect cybersecurity anomalies from labeled datasets. The project focuses on modular, object-oriented programming principles and includes various components for data preprocessing, model training, and hyperparameter tuning.

## Project Structure

The project is organized into different files, each with a specific responsibility. Here’s an overview of the files and their purpose:

### 1. **`data_loader.py`**: 
   - This module is responsible for loading, preprocessing, and converting data into PyTorch tensors.
   - Key tasks:
     - Load data from CSV files.
     - Scale features using `StandardScaler`.
     - Convert features and labels into PyTorch tensors for model training.
  
### 2. **`model.py`**:
   - Defines the neural network architecture using PyTorch's `nn.Module`.
   - The model consists of fully connected layers with ReLU activations to handle the binary classification of cybersecurity anomalies.

### 3. **`train.py`**:
   - Handles the model training and evaluation process.
   - The class `TrainModel` is designed to:
     - Load and preprocess the data using `DataLoader`.
     - Train the neural network using the specified optimizer (Adam or SGD).
     - Evaluate the model on the test set.
   - The optimizer can be chosen in the `config.py` file (either `'Adam'` or `'SGD'`).

### 4. **`config.py`**:
   - Stores the hyperparameters and paths used throughout the project.
   - Key parameters include:
     - Learning rate
     - Batch size
     - Number of epochs
     - Optimizer choice (`OPTMZ`)
   - Example:
     ```python
     learning_rate = 0.001
     num_epochs = 50
     OPTMZ = 'Adam'  # Change to 'SGD' if needed
     ```

### 5. **`hyperparameter_search.py`**:
   - Implements hyperparameter tuning by testing different combinations of learning rates, batch sizes, and optimizers.
   - It tracks the best combination of hyperparameters and prints them after evaluation.
   - The search space includes:
     - Learning rates: [1e-4, 1e-3, 1e-2]
     - Batch sizes: [32, 64, 128]
     - Optimizers: ['SGD', 'Adam']

### 6. **`test_data.py`**:
   - A separate module to test the integrity of the data and check for data overlap between training, testing, and validation sets.
   - It also visualizes the feature distributions across these datasets to help detect potential issues with the data.

### 7. **`README.md`**:
   - This file, which explains the project and its structure.

---

## How to Run the Project

1. **Install Dependencies**:
   Ensure that you have the required dependencies installed. The core dependencies are:
   - `torch`
   - `pandas`
   - `scikit-learn`
   - `torchmetrics`
   - `matplotlib` (for plotting)
   - `seaborn` (for correlation heatmap)

   You can install them using:
   ```bash
   pip install torch pandas scikit-learn torchmetrics matplotlib seaborn
   ```

2. **Configure Hyperparameters**:
   Open the `config.py` file to adjust the hyperparameters and optimizer. For example:
   ```python
   learning_rate = 0.001
   num_epochs = 50
   OPTMZ = 'Adam'  # Or change to 'SGD'
   ```

3. **Train the Model**:
   To start training the model, run the `train.py` file:
   ```bash
   python train.py
   ```

4. **Run Hyperparameter Search**:
   If you'd like to perform a hyperparameter search, run the `hyperparameter_search.py` script:
   ```bash
   python hyperparameter_search.py
   ```

5. **Test the Data**:
   To check for data overlap and visualize feature distributions, run `test_data.py`:
   ```bash
   python test_data.py
   ```

---

## Project Flow

1. **Data Loading**: Data is loaded from CSV files (`labelled_train.csv`, `labelled_test.csv`, `labelled_validation.csv`) and processed (scaled and converted to tensors).
2. **Model Training**: The `TrainModel` class is responsible for training a deep learning model on the data. The training process can be customized by choosing different optimizers (SGD or Adam).
3. **Evaluation**: After training, the model's performance is evaluated on the test set, and the results are printed.
4. **Hyperparameter Tuning**: The project includes hyperparameter search to find the best combination of learning rates, batch sizes, and optimizers for the model.
5. **Data Testing**: A separate script ensures that there is no data overlap between different sets and visualizes the feature distributions for better understanding of the data.

---

## License
This project is developed as part of the **"Detecting Cybersecurity Threats using Deep Learning"** course on DataCamp. It is intended for educational purposes only.

---

## Acknowledgements
Special thanks to DataCamp for providing the project and dataset for this learning opportunity.

---

Feel free to explore the code, modify hyperparameters, and train the model on your own! Let me know if you have any questions or encounter any issues while running the project.

