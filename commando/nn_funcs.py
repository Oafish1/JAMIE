import torch


def uc_loss(primes, F, pairwise=False):
    """Select loss term from UnionCom"""
    if pairwise:
        uc_loss = 0
        for i in range(primes[0].shape[0]):
            partial_sum = 0
            for j in range(primes[1].shape[0]):
                partial_sum += primes[1][j] * F[i, j]
            norm = primes[0][i] - partial_sum
            norm = torch.square(norm).sum()
            uc_loss += norm
    else:
        norm = primes[0] - torch.mm(F, primes[1])
        uc_loss = torch.square(norm).sum()
    return uc_loss


def nlma_loss(
    primes,
    Wx,
    Wy,
    Wxy,
    mu,
    fg=True,
    ff=True,
    gg=True,
):
    """Compute NLMA loss"""
    if not (fg and ff and gg):
        nlma_loss = 0
        for i in range(primes[0].shape[0]):
            for j in range(primes[1].shape[0]):
                if fg:
                    norm = primes[0][i] - primes[1][j]
                    norm = torch.square(norm).sum()
                    nlma_loss += norm * Wxy[i, j] * (1 - mu)

                if ff:
                    norm = primes[0][i] - primes[0][j]
                    norm = torch.square(norm).sum()
                    nlma_loss += norm * Wx[i, j] * mu

                if gg:
                    norm = primes[1][i] - primes[1][j]
                    norm = torch.square(norm).sum()
                    nlma_loss += norm * Wy[i, j] * mu
    else:
        num_cells = Wxy.shape[0]

        Dx = torch.sum(Wx, dim=0)
        Dy = torch.sum(Wy, dim=0)
        D = torch.diag(torch.cat((Dx, Dy), dim=0))
        W = torch.block_diag(Wx, Wy)
        W[:num_cells][:, num_cells:] += Wxy
        W[num_cells:][:, :num_cells] += torch.t(Wxy)

        L = D - W
        P = torch.cat(primes, dim=0)

        nlma_loss = torch.trace(torch.mm(torch.mm(torch.t(P), L), P))
    return nlma_loss


def gw_loss(primes):
    """Calculate Gromov-Wasserstein Distance"""
    # ASDF Implement fast approximation
    assert all(len(primes[0]) == len(p) for p in primes), (
        'Datasets must be aligned'
    )

    num_cells = len(primes[0])
    loss = 0
    for i in range(num_cells):
        for j in range(num_cells):
            set1 = torch.norm(primes[0][i] - primes[0][j])
            set2 = torch.norm(primes[1][i] - primes[1][j])
            loss += torch.square(set1 - set2)
    return loss
