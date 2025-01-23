import torch
import torch.nn as nn

# Function for training the model
def train(epochs, device, train_loader, model, optimizer, criterion):

    # Training loop
    for epoch in range(epochs):

        # Switch the model to training mode
        model.train()

        # Initialize training loss
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader)}")