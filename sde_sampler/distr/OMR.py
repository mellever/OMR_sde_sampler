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
        mu: float = 0,
        sigma: float = 1,
        truncate_quartile: float | None = None,
        grid_points: int = 2001,
        domain_delta: float = 2.5,
        **kwargs,
    ):

        super().__init__(dim=dim, grid_points=grid_points, **kwargs)
        
        self.register_buffer("mu", torch.tensor(mu), persistent=False)
        self.register_buffer("sigma", torch.tensor(sigma), persistent=False)

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
        var = self.sigma**2
        return -0.5 * (x-self.mu)**2/var - 0.5 * (2*math.pi*var).log()

    # Gradient of the log normal distribution
    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return (self.mu-x)/self.sigma**2

    # Marginal distribution
    def marginal_distr(self, dim=0) -> torch.distributions.Distribution:
        return distributions.Normal(self.mu, self.sigma)

    # Marginal
    def marginal(self, x: torch.Tensor, dim=0) -> torch.Tensor:
        return self.marginal_distr(dim=dim).log_prob(x).exp()

    # Sample from the paths
    # A good sampling choice dramatically improves the training efficiency, but requires information on the distribution
    # When we have no idea, best to use uniform sampling
    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()

        if self.truncate_quartile is None:
            return self.mu + self.sigma * torch.randn(*shape, self.dim, device=self.domain.device)
        
        tensor = torch.empty(*shape, self.dim, device=self.domain.device)
        return trunc_normal_(tensor, mean=self.mu, std=self.sigma, a=self.truncate_quartile[0], b=self.truncate_quartile[1])

class NormalRejection(Distribution):
    def __init__(
        self,
        dim: int = 1,
        shift: float = 0.0,
        grid_points: int = 2001,
        rejection_sampling_scaling: float = 3.0,
        domain_delta: float = 2.5,
        mu: float = 0,
        sigma: float = 0.2,
        **kwargs,
    ):

        super().__init__(dim=1, grid_points=grid_points, **kwargs)
        
        self.rejection_sampling_scaling = rejection_sampling_scaling
        self.register_buffer("mu", torch.tensor(mu), persistent=False)
        self.register_bufferfloat("sigma", torch.tensor(sigma), persistent=False)

        # Set domain
        if self.domain is None:
            domain = domain_delta * torch.tensor([[-1.0, 1.0]])
            self.set_domain(domain)

    # Log of the probability density
    # Here we take the normal distribution
    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -1/2 * ((x-self.mu)/self.sigma)**2 - (self.sigma*(2*math.pi)).sqrt().log()

    # Gradient of the log normal distribution
    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return (x-self.mu)/self.sigma**2

    def marginal(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.pdf(x)

    def get_proposal_distr(self):
        device = self.domain.device
        proposal = IsotropicGauss(dim=1, loc=self.mu, scale=5*self.sigma)
        proposal.to(device)
        return proposal

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        proposal = self.get_proposal_distr()
        return rejection_sampling(
            shape=shape,
            target=self,
            proposal=proposal,
            scaling=self.rejection_sampling_scaling,
        )

    def plots(self, samples, nbins=100) -> torch.Tensor:
        samples = self.sample((samples.shape[0],))
        fig = plot_marginal(
            x=samples,
            marginal=lambda x, **kwargs: self.pdf(x),
            dim=0,
            nbins=nbins,
            domain=self.domain,
        )

        x = torch.linspace(*self.domain[0], steps=nbins, device=self.domain.device)
        y = (
            self.get_proposal_distr().pdf(x.unsqueeze(-1))
            * self.rejection_sampling_scaling
        )
        fig.add_trace(
            go.Scatter(
                x=x.cpu(),
                y=y.squeeze(-1).cpu(),
                mode="lines",
                name="proposal",
            )
        )
        return {"plots/rejection_sampling": fig}
