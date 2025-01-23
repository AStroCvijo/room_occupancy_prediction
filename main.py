import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# Import untils
from utils.argparser import arg_parse
from utils.seed import *

# Import the data preprocessing functions
from data.data import *

# Import the models
from model.LSTM import LSTM
from model.GRU import GRU

# Import train and evaluation functions
from train.train import train
from train.eval import evaluate

if __name__ == "__main__":

    # Set the seed
    seed = 42
    set_seed(seed)

    # Parse the arguments
    args = arg_parse()

    # ------------------------------------------------------------------------------------------------------
    # Data handeling
    # ------------------------------------------------------------------------------------------------------

    # Load the csv file
    file_path = args.file_path
    data = pd.read_csv(file_path)

    # Preprocess the data
    data = data_preprocess(data)
    X, y = create_sequences(data, args.sequence_length)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create train and test Datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    # Create train and test loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("Data preprocessed.")

    # ------------------------------------------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------------------------------------------   

    # Model parameters
    input_size = X_train.shape[2]
    output_size = 1
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers

    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the Model
    if (args.model == "LSTM"):
        model = LSTM(input_size, hidden_dim, num_layers, output_size).to(device)
        print("LSTM model initialized.")
    if (args.model == "GRU"):
        model = GRU(input_size, hidden_dim, num_layers, output_size).to(device)
        print("GRU model initialized.")
    model.apply(init_weights)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # ------------------------------------------------------------------------------------------------------
    # Model training and evaluation
    # ------------------------------------------------------------------------------------------------------

    print(f"Using {device} for training.")

    # Train the model
    train(args.epochs, device, train_loader, model, optimizer, criterion)

    # Evaulate the model
    evaluate(device, test_loader, model)