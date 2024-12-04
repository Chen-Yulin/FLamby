import torch
import torch.nn as nn

from flamby.datasets.fed_tcga_brca import FedTcgaBrca


class EnhancedModel(nn.Module):
    """
    Enhanced model with additional layers.
    """

    def __init__(self):
        super(EnhancedModel, self).__init__()
        input_size = 39
        hidden_size = 64  # Example hidden layer size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # Dropout with 50% probability
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.batch_norm(out)
        out = self.fc2(out)
        return out


if __name__ == "__main__":

    mydataset = FedTcgaBrca(train=True, pooled=True)

    model = EnhancedModel()

    for i in range(10):
        X = torch.unsqueeze(mydataset[i][0], 0)
        y = torch.unsqueeze(mydataset[i][1], 0)
        print(X.shape)
        print(y)
        print(model(X))
