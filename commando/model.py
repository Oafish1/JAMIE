import torch
import torch.nn as nn


class edModel(nn.Module):
    """
    Encoder-decoder model for use in dimensionality reduction.
    In the style of UnionCom's ``Model.py``
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.num_modalities = len(input_dim)
        self.encoders = []
        for i in range(self.num_modalities):
            self.encoders.append(nn.Sequential(
                nn.Linear(input_dim[i], 2*input_dim[i]),
                nn.BatchNorm1d(2*input_dim[i]),
                nn.LeakyReLU(0.1, True),

                nn.Linear(2*input_dim[i], 2*input_dim[i]),
                nn.BatchNorm1d(2*input_dim[i]),
                nn.LeakyReLU(0.1, True),

                nn.Linear(2*input_dim[i], input_dim[i]),
                nn.BatchNorm1d(input_dim[i]),
                nn.LeakyReLU(0.1, True),

                nn.Linear(input_dim[i], output_dim),
                nn.BatchNorm1d(output_dim),
                nn.LeakyReLU(0.1, True),
            ))
        self.encoders = nn.ModuleList(self.encoders)

        self.decoders = []
        for i in range(self.num_modalities):
            self.decoders.append(nn.Sequential(
                nn.Linear(output_dim, input_dim[i]),
                nn.BatchNorm1d(input_dim[i]),
                nn.LeakyReLU(0.1, True),

                nn.Linear(input_dim[i], 2*input_dim[i]),
                nn.BatchNorm1d(2*input_dim[i]),
                nn.LeakyReLU(0.1, True),

                nn.Linear(2*input_dim[i], 2*input_dim[i]),
                nn.BatchNorm1d(2*input_dim[i]),
                nn.LeakyReLU(0.1, True),

                nn.Linear(2*input_dim[i], input_dim[i]),
                nn.BatchNorm1d(input_dim[i]),
                nn.LeakyReLU(0.1, True),
            ))
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, *X):
        """Regular forward method"""
        embedded = [self.encoders[i](X[i]) for i in range(self.num_modalities)]
        combined = torch.stack(embedded, dim=0).sum(dim=0)  # Needs to change for partial align
        reconstructed = [self.decoders[i](combined) for i in range(self.num_modalities)]

        return embedded, reconstructed
