from torch import nn

class NN(nn.Module):
  def __init__(self) -> None:
    super(NN, self).__init__()

    self.layer1 = nn.Linear(5, 30)
    self.layer2 = nn.Linear(30, 10)
    self.out = nn.Linear(10, 1)
    self.dropout = nn.Dropout(0.4)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.layer1(x))
    x = self.dropout(x)
    x = self.relu(self.layer2(x))
    x = self.dropout(x)
    
    return self.out(x)