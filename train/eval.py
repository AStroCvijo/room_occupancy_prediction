import numpy as np
import torch
import torch.nn as nn

# Function for evaluating the model
def evaluate(device, test_loader, model):
    
    # Switch the model to training mode
    model.eval()
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Get predictions
            outputs = model(X_batch)

            # Note (Since the models output are not int type abs() and round() are used)
            predictions.extend(abs(np.round(outputs.cpu().numpy())).flatten())
            ground_truth.extend(y_batch.cpu().numpy())

    # Convert to numpy arrays
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Calculate accuracy
    correct_predictions = np.sum(predictions == ground_truth)
    total_predictions = len(predictions)
    accuracy = correct_predictions / total_predictions * 100

    print(f"Accuracy: {accuracy:.2f}%")
