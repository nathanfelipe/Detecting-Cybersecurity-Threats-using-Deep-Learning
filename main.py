from data_loader import DataLoader
from train import TrainModel
from config import data_paths, input_size, hidden_size, output_size, learning_rate, num_epochs, OPTMZ


def main():
    # Initialize the training process
    trainer = TrainModel(data_paths, input_size, hidden_size, output_size, learning_rate, num_epochs, OPTMZ)

    # Train the model
    print("Starting training...")
    trainer.train()

    # Evaluate the model
    print("Evaluating the model on test data...")
    trainer.evaluate()


if __name__ == "__main__":
    main()