import math

import torch


class TorchKernelDensity:
    gaussian = "gaussian"
    triangular = "triangular"
    epanechnikov = "epanechnikov"
    laplace = "laplace"

    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.X_fit_ = None

    def fit(self, X):
        self.X_fit_ = X.clone().detach()

    def score_samples(self, X):
        X = X.clone().detach()
        n_samples = self.X_fit_.shape[0]

        distances = self._pairwise_distances(X, self.X_fit_)
        kernel_values = self._apply_kernel(distances, X.device)

        log_density = torch.logsumexp(torch.log(kernel_values), dim=1) - torch.log(
            torch.tensor(n_samples, device=X.device))
        probs = to_probs(log_density)
        return probs

    def _pairwise_distances(self, X, Y):
        return torch.cdist(X, Y) / self.bandwidth

    def _apply_kernel(self, distances, device):
        if self.kernel == self.gaussian:
            pi = torch.tensor(math.pi).to(device)
            return torch.exp(-0.5 * distances ** 2) / (self.bandwidth * torch.sqrt(2 * pi))
        elif self.kernel == self.triangular:
            return torch.maximum(1 - torch.abs(distances) / self.bandwidth,
                                 torch.tensor(0.0).to(device)) / self.bandwidth
        elif self.kernel == self.epanechnikov:
            mask = (distances <= self.bandwidth).to(device)
            return (3 / 4 * (1 - (distances / self.bandwidth) ** 2) * mask) / self.bandwidth
        elif self.kernel == self.laplace:
            return torch.exp(-torch.abs(distances) / self.bandwidth) / (2 * self.bandwidth)
        else:
            raise ValueError(
                "Kernel not recognized. Supported kernels are: 'gaussian', 'triangular', 'epanechnikov', 'laplace'")


def to_probs(x):
    weights = torch.exp(x)
    weights /= torch.clamp(torch.max(weights), 1e-5)
    return weights

