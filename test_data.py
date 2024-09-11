import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_loader import DataLoader


class TestData:
    def __init__(self, data_paths):
        self.data_loader = DataLoader(*data_paths)
        self.train_features, self.train_labels, self.test_features, self.test_labels, self.validation_features, self.validation_labels = self.data_loader.load_data()

    def check_data_leakage(self):
        # Ensure no overlap between train and test data (to prevent leakage)
        combined_data = self.train_features._append(self.test_features)
        if combined_data.duplicated().sum() > 0:
            print("Warning: Data leakage detected between training and test sets.")
        else:
            print("No data leakage detected.")

    def check_data_shapes(self):
        # Checking if the dimensions of features and labels match
        assert self.train_features.shape[0] == self.train_labels.shape[0], "Train features and labels have different number of rows!"
        assert self.test_features.shape[0] == self.test_labels.shape[0], "Test features and labels have different number of rows!"
        assert self.validation_features.shape[0] == self.validation_labels.shape[0], "Validation features and labels have different number of rows!"
        print("Data shape checks passed.")

    def plot_feature_distribution(self):
        # Plotting the distribution of each feature in the training data
        plt.figure(figsize=(12, 8))
        self.train_features.hist(bins=30, figsize=(12, 8), layout=(2, 4))
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self):
        # Plotting the correlation matrix of the training features
        correlation_matrix = self.train_features.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.show()

    def run_all_tests(self):
        print("Running data leakage test...")
        self.check_data_leakage()

        print("Running data shape checks...")
        self.check_data_shapes()

        print("Plotting feature distributions...")
        self.plot_feature_distribution()

        print("Plotting correlation heatmap...")
        self.plot_correlation_heatmap()


# Load the data (assumes train_df, test_df, validation_df are loaded as pandas DataFrames)
train_df = pd.read_csv('labelled_train.csv')
test_df = pd.read_csv('labelled_test.csv')
validation_df = pd.read_csv('labelled_validation.csv')

# Part 1: Plot Feature Distributions
# Select a few features to plot (you can adjust the number or select specific features)
features_to_plot = train_df.columns[:-1][:3]  # Adjust the slicing as needed

for feature in features_to_plot:
    plt.figure(figsize=(10, 6))

    # Plot histograms of the feature for train, test, and validation sets
    plt.hist(train_df[feature], bins=50, alpha=0.5, label='Train', color='blue', density=True)
    plt.hist(test_df[feature], bins=50, alpha=0.5, label='Test', color='orange', density=True)
    plt.hist(validation_df[feature], bins=50, alpha=0.5, label='Validation', color='green', density=True)

    # Customize the plot
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Part 2: Check for Data Overlap
# Convert DataFrames to tuples of values for comparison
train_tuples = set([tuple(row) for row in train_df.values])
test_tuples = set([tuple(row) for row in test_df.values])
validation_tuples = set([tuple(row) for row in validation_df.values])

# Check for overlap between training and test sets
overlap_test_train = train_tuples.intersection(test_tuples)
overlap_train_validation = train_tuples.intersection(validation_tuples)
overlap_test_validation = test_tuples.intersection(validation_tuples)

# Print results of the overlap checks
print(f"Overlap between training and test set: {len(overlap_test_train)} rows")
print(f"Overlap between training and validation set: {len(overlap_train_validation)} rows")
print(f"Overlap between test and validation set: {len(overlap_test_validation)} rows")


if __name__ == "__main__":
    from config import data_paths
    tester = TestData(data_paths)
    tester.run_all_tests()