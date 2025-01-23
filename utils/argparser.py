import argparse

# Function for parsing arguments
def arg_parse():

    # Initialize the parser
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument('-hs',  '--hidden_dim',        type=int,   default = 64,                              help="Size of the models hidden layer")
    parser.add_argument('-nl',  '--num_layers',        type=int,   default = 1,                               help="Number of layers in the model")

    # Training arguments
    parser.add_argument('-e',   '--epochs',            type=int,   default = 20,                              help="Number of epochs in training")
    parser.add_argument('-lr',  '--learning_rate',     type=float, default = 0.001,                           help="Learning rate in training")

    # Data arguments
    parser.add_argument('-fp',  '--file_path',         type=str,   default = "data/Occupancy_Estimation.csv", help="Path to data")
    parser.add_argument('-sl',  '--sequence_length',   type=int,   default = 3,                               help="Length of input sequences")
    parser.add_argument('-bs',  '--batch_size',        type=int,   default = 32,                              help="Batch size for training")

    # Parse the arguments
    return parser.parse_args()
