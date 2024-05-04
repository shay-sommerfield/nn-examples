import torch
import torch.nn as nn
import torch.optim as optim

### This is a translation that chatGPT gave me to modern pytorch code

# Define the non-linear activation function
def nonlin(x):
    return torch.sigmoid(x)

# Define the derivative of the non-linear activation function
def nonlin_deriv(x):
    return x * (1 - x)

# Convert input and target data to PyTorch tensors
X = torch.tensor([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.syn0 = nn.Linear(3, 4)
        self.syn1 = nn.Linear(4, 1)
    
    def forward(self, x):
        x = nonlin(self.syn0(x))
        x = nonlin(self.syn1(x))
        return x

# Create an instance of the neural network
net = NeuralNetwork()

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.SGD(net.parameters(), lr=0.1)

# Training loop
for j in range(60000):

    # Forward pass
    output = net(X)

    # Compute loss
    loss = criterion(output, y)

    # Print error every 10000 iterations
    if (j % 10000) == 0:
        print("Error:", loss.item())

    # Zero the gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()
