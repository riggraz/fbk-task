from torch import nn

class NN(nn.Module):
  def __init__(self, n_neurons) -> None:
    super(NN, self).__init__()

    self.layer1 = nn.Linear(5, n_neurons)
    self.layer2 = nn.Linear(n_neurons, 1)

    self.dropout = nn.Dropout(0.3)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.layer1(x))
    x = self.dropout(x)
    
    return self.layer2(x)