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

    def forward(self, *X, aligned_idx=None):
        """
        Regular forward method.

        aligned_idx: List of idxs for aligned pairs.  Currently given as
            ((1, 3, 10, etc.), (1, 3, 15, etc.)).  Support for duplicate
            alignments would require revision on the averaging.
        """
        assert aligned_idx is not None, '``aligned_idx`` must be provided.'
        embedded = [self.encoders[i](X[i]) for i in range(self.num_modalities)]

        # For full, ordered alignment
        # combined = torch.stack(embedded, dim=0).sum(dim=0)
        # reconstructed = [self.decoders[i](combined) for i in range(self.num_modalities)]

        # Optimize this
        overlap = [embedded[i][aligned_idx[i]] for i in range(self.num_modalities)]
        combined = torch.stack(overlap, dim=0).sum(dim=0) / 2
        unaligned_idx = [
            [j for j in range(len(X[i])) if j not in aligned_idx[i]]
            for i in range(self.num_modalities)
        ]
        assembled = [
            torch.cat([embedded[i][unaligned_idx[i]], combined], dim=0)
            for i in range(self.num_modalities)
        ]

        reconstructed = [self.decoders[i](assembled[i]) for i in range(self.num_modalities)]

        return embedded, reconstructed
