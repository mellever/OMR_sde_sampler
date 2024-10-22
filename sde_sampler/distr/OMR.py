from __future__ import annotations

import math

import plotly.graph_objects as go
import torch

from torch.nn.init import trunc_normal_
from torch import distributions

from sde_sampler.eval.plots import plot_marginal

from .base import Distribution, rejection_sampling
from .gauss import GMM, IsotropicGauss


class Normal(Distribution):
    def __init__(
        self,
        dim: int = 1,
        mu: float = 0.0,
        sigma: float = 1.0,
        truncate_quartile: float | None = None,
        grid_points: int = 2001,
        domain_delta: float = 2.5,
        **kwargs,
    ):

        super().__init__(dim=dim, grid_points=grid_points, **kwargs)
        
        # Initialize mu as a 1D tensor of size (dim,) with each value being mu
        self.register_buffer("mu", torch.full((dim,), mu), persistent=False)
        
        # Initialize sigma as a diagonal matrix with sigma on the diagonal
        self.register_buffer("sigma", torch.diag_embed(torch.full((dim,), sigma)), persistent=False)


        # Set domain
        if self.domain is None:
            domain = domain_delta * torch.tensor([[-1.0, 1.0]])
            self.set_domain(domain)

        # Calculate truncation values --> This significantly increases training efficiency
        if truncate_quartile is not None:
            quartiles = torch.tensor(
                [truncate_quartile / 2, 1 - truncate_quartile / 2],
                device=self.domain.device,
            )
            truncate_quartile = self.marginal_distr().icdf(quartiles).tolist()
        self.truncate_quartile = truncate_quartile


    # Log of the probability density
    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        var = self.sigma[0,0]**2
        return -0.5 * self.dim * (x-self.mu[0])**2/var - 0.5 * (2*math.pi*var).log()

    # Gradient of the log normal distribution
    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return (self.mu[0]-x)/self.sigma[0,0]**2

    # Marginal distribution for a specific dimension
    def marginal_distr(self, dim=0) -> torch.distributions.Distribution:
        #return distributions.Normal(self.mu, self.sigma)
        return distributions.Normal(self.mu[0], self.sigma[0, 0])

    # Marginal for all dimensions
    # First log and then exp is to handle very small numbers
    def marginal(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.marginal_distr(dim=self.dim).log_prob(x).exp()

    # Sample from the paths
    # A good sampling choice dramatically improves the training efficiency, but requires information on the distribution
    # When we have no idea, best to use uniform sampling
    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()

        if self.truncate_quartile is None:
            return self.mu[0] + self.sigma[0,0] * torch.randn(*shape, dim, device=self.domain.device)
        
        tensor = torch.empty(*shape, self.dim, device=self.domain.device)
        return trunc_normal_(tensor, mean=self.mu[0], std=self.sigma[0,0], a=self.truncate_quartile[0], b=self.truncate_quartile[1])