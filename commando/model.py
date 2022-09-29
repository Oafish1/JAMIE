import torch
import torch.nn as nn

from .utilities import identity


class edModel(nn.Module):
    """
    Encoder-decoder model for use in dimensionality reduction.
    In the style of UnionCom's ``Model.py``
    """
    def __init__(self, input_dim, output_dim, preprocessing=None, preprocessing_inverse=None, sigma=None):
        super().__init__()

        self.num_modalities = len(input_dim)
        # For outputting the model with preprocessing included
        if preprocessing is None:
            self.preprocessing = self.num_modalities * [identity]
        else:
            self.preprocessing = preprocessing
        if preprocessing_inverse is None:
            self.preprocessing_inverse = self.num_modalities * [identity]
        else:
            self.preprocessing_inverse = preprocessing_inverse

        self.encoders = []
        for i in range(self.num_modalities):
            self.encoders.append(nn.Sequential(
                # nn.Linear(input_dim[i], 2*input_dim[i]),
                # nn.BatchNorm1d(2*input_dim[i]),
                # nn.LeakyReLU(),
                #
                # nn.Linear(2*input_dim[i], 2*input_dim[i]),
                # nn.BatchNorm1d(2*input_dim[i]),
                # nn.LeakyReLU(),
                #
                # nn.Linear(2*input_dim[i], input_dim[i]),
                # nn.BatchNorm1d(input_dim[i]),
                # nn.LeakyReLU(),

                nn.Linear(input_dim[i], input_dim[i]),
                nn.BatchNorm1d(input_dim[i]),
                nn.LeakyReLU(),

                nn.Linear(input_dim[i], output_dim),
                nn.BatchNorm1d(output_dim),
            ))
        self.encoders = nn.ModuleList(self.encoders)

        self.fc_mus = []
        for i in range(self.num_modalities):
            self.fc_mus.append(nn.Linear(input_dim[i], output_dim))
        self.fc_mus = nn.ModuleList(self.fc_mus)

        self.fc_vars = []
        for i in range(self.num_modalities):
            self.fc_vars.append(nn.Linear(input_dim[i], output_dim))
        self.fc_vars = nn.ModuleList(self.fc_vars)

        self.decoders = []
        for i in range(self.num_modalities):
            self.decoders.append(nn.Sequential(
                nn.Linear(output_dim, input_dim[i]),
                nn.BatchNorm1d(input_dim[i]),
                nn.LeakyReLU(),

                # nn.Linear(input_dim[i], 2*input_dim[i]),
                # nn.BatchNorm1d(2*input_dim[i]),
                # nn.LeakyReLU(),
                #
                # nn.Linear(2*input_dim[i], 2*input_dim[i]),
                # nn.BatchNorm1d(2*input_dim[i]),
                # nn.LeakyReLU(),
                #
                # nn.Linear(2*input_dim[i], input_dim[i]),
                # nn.BatchNorm1d(input_dim[i]),

                nn.Linear(input_dim[i], input_dim[i]),
                nn.BatchNorm1d(input_dim[i]),
            ))
        self.decoders = nn.ModuleList(self.decoders)

        self.sigma = nn.Parameter(torch.rand(self.num_modalities))
        self.log_scale = nn.Parameter(torch.zeros(self.num_modalities))

    def encode(self, X):
        return [self.encoders[i](X[i]) for i in range(self.num_modalities)]

    def combine(self, X, corr):
        return [
            (
                self.sigma[i] * X[i]
                + self.sigma[(i + 1) % 2] * torch.mm(
                    corr if i == 0 else torch.t(corr),
                    X[(i + 1) % 2])
            ) / (
                self.sigma[i]
                + self.sigma[(i + 1) % 2] * corr.sum((i + 1) % 2).reshape(-1, 1)
            )
            for i in range(self.num_modalities)
        ]

    def decode(self, X):
        return [self.decoders[i](X[i]) for i in range(self.num_modalities)]

    def forward(self, *X, corr):
        """
        Regular forward method.

        corr: Correspondence matrix
        """
        # AE
        embedded = self.encode(X)
        combined = self.combine(embedded, corr)
        reconstructed = self.decode(combined)

        return embedded, combined, reconstructed, None, None

    def impute(self, X, compose):
        from_mod, to_mod = compose
        prep = torch.Tensor(self.preprocessing[from_mod](X))
        encoded = self.encoders[from_mod](prep)
        decoded = self.decoders[to_mod](encoded)
        return self.preprocessing_inverse[to_mod](decoded.detach())


class edModelVar(nn.Module):
    """
    Variational encoder-decoder model for use in dimensionality reduction.
    In the style of UnionCom's ``Model.py``
    """
    def __init__(self, input_dim, output_dim, preprocessing=None, preprocessing_inverse=None, sigma=None):
        super().__init__()

        self.num_modalities = len(input_dim)
        # For outputting the model with preprocessing included
        if preprocessing is None:
            self.preprocessing = self.num_modalities * [identity]
        else:
            self.preprocessing = preprocessing
        if preprocessing_inverse is None:
            self.preprocessing_inverse = self.num_modalities * [identity]
        else:
            self.preprocessing_inverse = preprocessing_inverse

        self.encoders = []
        for i in range(self.num_modalities):
            self.encoders.append(nn.Sequential(
                # UnionCom Like
                nn.Linear(input_dim[i], 2*input_dim[i]),
                nn.BatchNorm1d(2*input_dim[i]),
                nn.LeakyReLU(),

                nn.Linear(2*input_dim[i], input_dim[i]),
                nn.BatchNorm1d(input_dim[i]),
                nn.LeakyReLU(),

                # # BABEL Like
                # nn.Linear(input_dim[i], 2*output_dim),
                # nn.BatchNorm1d(2*output_dim),
                # nn.PReLU(),
            ))
        self.encoders = nn.ModuleList(self.encoders)

        intermediate_dim = []
        for i in range(self.num_modalities):
            intermediate_dim.append(input_dim[i])  # UnionCom Like
            # intermediate_dim.append(2*output_dim)  # BABEL Like

        self.fc_mus = []
        for i in range(self.num_modalities):
            self.fc_mus.append(nn.Linear(intermediate_dim[i], output_dim))
        self.fc_mus = nn.ModuleList(self.fc_mus)

        self.fc_vars = []
        for i in range(self.num_modalities):
            self.fc_vars.append(nn.Linear(intermediate_dim[i], output_dim))
        self.fc_vars = nn.ModuleList(self.fc_vars)

        self.decoders = []
        for i in range(self.num_modalities):
            self.decoders.append(nn.Sequential(
                # UnionCom Like
                nn.Linear(output_dim, input_dim[i]),
                nn.BatchNorm1d(input_dim[i]),
                nn.LeakyReLU(),

                nn.Linear(input_dim[i], 2*input_dim[i]),
                nn.BatchNorm1d(2*input_dim[i]),
                nn.LeakyReLU(),

                nn.Linear(2*input_dim[i], input_dim[i]),
                # nn.BatchNorm1d(input_dim[i]),
                # nn.LeakyReLU(),

                # # BABEL Like
                # nn.Linear(output_dim, 2*output_dim),
                # nn.BatchNorm1d(2*output_dim),
                # nn.PReLU(),
                #
                # nn.Linear(2*output_dim, input_dim[i]),
                # nn.BatchNorm1d(input_dim[i]),
                # nn.PReLU(),
            ))
        self.decoders = nn.ModuleList(self.decoders)

        self.sigma = nn.Parameter(torch.rand(self.num_modalities))
        self.log_scale = nn.Parameter(torch.zeros(self.num_modalities))

    def encode(self, X):
        return [self.encoders[i](X[i]) for i in range(self.num_modalities)]

    def refactor(self, X, index=None):
        if index is None:
            index = range(self.num_modalities)
        zs = []; mus = []; stds = []
        for x, i in zip(X, index):
            mu = self.fc_mus[i](x)
            log_var = self.fc_vars[i](x)
            std = torch.exp(log_var)
            if not self.training:
                zs.append(mu)
            else:
                q = torch.distributions.Normal(mu, std)
                zs.append(q.rsample())
            mus.append(mu)
            stds.append(std)
        return zs, mus, stds

    def combine(self, X, corr):
        return [
            (
                self.sigma[i] * X[i]
                + self.sigma[(i + 1) % 2] * torch.mm(
                    corr if i == 0 else torch.t(corr),
                    X[(i + 1) % 2])
            ) / (
                self.sigma[i]
                + self.sigma[(i + 1) % 2] * corr.sum((i + 1) % 2).reshape(-1, 1)
            )
            for i in range(self.num_modalities)
        ]

    def decode(self, X):
        return [self.decoders[i](X[i]) for i in range(self.num_modalities)]

    def forward(self, *X, corr):
        """
        Regular forward method.

        corr: Correspondence matrix
        """
        # VAE
        zs, mus, stds = self.refactor(self.encode(X))
        combined = self.combine(zs, corr)
        X_hat = self.decode(combined)

        return zs, combined, X_hat, mus, stds

    def impute(self, X, compose):
        from_mod, to_mod = compose
        prep = torch.Tensor(self.preprocessing[from_mod](X))
        encoded = self.encoders[from_mod](prep)
        refactored = self.refactor([encoded], [from_mod])[0][0]
        decoded = self.decoders[to_mod](refactored)
        return self.preprocessing_inverse[to_mod](decoded.detach())
