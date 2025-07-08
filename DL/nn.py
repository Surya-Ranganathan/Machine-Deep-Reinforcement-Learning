import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input to hidden
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden to output
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    # Create model instance
    input_size = 2   # Input features
    hidden_size = 4  # Hidden layer neurons
    output_size = 1  # Output neuron (e.g., for binary classification)
    model = SimpleNN(input_size, hidden_size, output_size)

    # Loss and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

    # Sample training data (random for demonstration)
    X_train = torch.randn(10, input_size)  # 10 samples, 2 features each
    y_train = torch.randn(10, output_size)  # 10 target values

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()  # Zero gradients
        outputs = model(X_train)  # Forward pass
        loss = criterion(outputs, y_train)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        if (epoch+1) % 10 == 0:  # Print every 10 epochs
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Test the model with a random input
    test_input = torch.randn(1, input_size)
    print("Test Prediction:", model(test_input).item())


if __name__ == "__main__":
    main()