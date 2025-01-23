# Room occupancy prediction

A project made for the subject "Numerical algorithms and numerical software"

Date of creation: January, 2025

## Quickstart
1. Clone the repository:
    ```bash
    git clone https://github.com/AStroCvijo/room_occupancy_prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd room_occupancy_prediction
    ```

3. Set up the environment:
    ```bash
    source ./setup.sh
    ```

4. Train the model using the default settings:
    ```bash
    python main.py
    ```

## Arguments guide 

`-m or --model`  
Specify the model to use (e.g., LSTM, GRU). Default: `LSTM`.  

`-hs or --hidden_dim`  
Size of the model's hidden layer. Default: `64`.  

`-nl or --num_layers`  
Number of layers in the model. Default: `2`.  

### Training Arguments
`-e or --epochs`  
Number of epochs in training. Default: `20`.  

`-lr or --learning_rate`  
Learning rate in training. Default: `0.001`.  

### Data Arguments
`-fp or --file_path`  
Path to the data file. Default: `data/Occupancy_Estimation.csv`.  

`-sl or --sequence_length`  
Length of input sequences. Default: `3`.  

`-bs or --batch_size`  
Batch size for training. Default: `32`.  

## How to Use

 ### Training Example: 
`python main.py --model LSTM --epochs 30 --learning_rate 0.0001 --sequence_length 5`
