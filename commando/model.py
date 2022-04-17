import torch
import torch.nn as nn


class edModel(nn.Module):
    """
    Encoder-decoder model for use in dimensionality reduction.
    In the style of UnionCom's ``Model.py``
    """
    def __init__(self, input_dim, output_dim, preprocessing=None):
        super().__init__()

        self.num_modalities = len(input_dim)
        # For outputting the model with preprocessing included
        if preprocessing is None:
            self.preprocessing = self.num_modalities * [lambda x: x]
        else:
            self.preprocessing = preprocessing
        self.encoders = []
        for i in range(self.num_modalities):
            self.encoders.append(nn.Sequential(
                nn.Linear(input_dim[i], 2*input_dim[i]),
                nn.BatchNorm1d(2*input_dim[i]),
                nn.LeakyReLU(),

                nn.Linear(2*input_dim[i], 2*input_dim[i]),
                nn.BatchNorm1d(2*input_dim[i]),
                nn.LeakyReLU(),

                nn.Linear(2*input_dim[i], input_dim[i]),
                nn.BatchNorm1d(input_dim[i]),
                nn.LeakyReLU(),

                nn.Linear(input_dim[i], output_dim),
                nn.BatchNorm1d(output_dim),
            ))
        self.encoders = nn.ModuleList(self.encoders)

        self.decoders = []
        for i in range(self.num_modalities):
            self.decoders.append(nn.Sequential(
                nn.Linear(output_dim, input_dim[i]),
                nn.BatchNorm1d(input_dim[i]),
                nn.LeakyReLU(),

                nn.Linear(input_dim[i], 2*input_dim[i]),
                nn.BatchNorm1d(2*input_dim[i]),
                nn.LeakyReLU(),

                nn.Linear(2*input_dim[i], 2*input_dim[i]),
                nn.BatchNorm1d(2*input_dim[i]),
                nn.LeakyReLU(),

                nn.Linear(2*input_dim[i], input_dim[i]),
                nn.BatchNorm1d(input_dim[i]),
            ))
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, *X, corr=None):
        """
        Regular forward method.

        corr: Correspondence matrix
        """
        assert corr is not None, '``corr`` must be provided.'
        embedded = [self.encoders[i](X[i]) for i in range(self.num_modalities)]
        combined = [
            (
                embedded[i]
                + torch.mm(
                    corr if i == 0 else torch.t(corr),
                    embedded[(i + 1) % 2])
            ) / (1. + corr.sum((i + 1) % 2).reshape(-1, 1))
            for i in range(self.num_modalities)
        ]
        reconstructed = [self.decoders[i](combined[i]) for i in range(self.num_modalities)]

        return embedded, reconstructed
