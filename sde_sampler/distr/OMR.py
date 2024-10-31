from __future__ import annotations

import math

import plotly.graph_objects as go
import torch

from torch.nn.init import trunc_normal_
from torch import distributions

from sde_sampler.eval.plots import plot_marginal

from .base import Distribution, rejection_sampling
from .gauss import GMM, IsotropicGauss

from torch.distributions.multivariate_normal import MultivariateNormal


class MultiNormal(Distribution):
    def __init__(
        self,
        name: str = "n5",
        dim: int = 5,
        grid_points: int = 2001,
        domain_delta: float = 1,
        **kwargs,
    ):

        super().__init__(dim=dim, grid_points=grid_points, **kwargs)

        self.name = name

        # Set domain
        if self.domain is None:
            domain = domain_delta * torch.tensor([[-5., 20.]])
            self.set_domain(domain)

        if self.name == "n5":
            #Define mu
            self.mu =  torch.tensor([3.,  5.,  7.,  9., 11.]) 

            #Define covariance matrix
            self.cov_matrix = torch.tensor([
                [0.8, 0.8, 0.8, 0.8, 0.8],
                [0.8, 1.6, 1.6, 1.6, 1.6],
                [0.8, 1.6, 2.4, 2.4, 2.4],
                [0.8, 1.6, 2.4, 3.2, 3.2],
                [0.8, 1.6, 2.4, 3.2, 4.0]
            ])

        elif self.name == "n10":
            # Define mu
            self.mu = torch.tensor([2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]) 

            # Define covariance matrix
            self.cov_matrix = torch.tensor([
                [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                [0.4, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                [0.4, 0.8, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2],
                [0.4, 0.8, 1.2, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],
                [0.4, 0.8, 1.2, 1.6, 2., 2., 2., 2., 2., 2.],
                [0.4, 0.8, 1.2, 1.6, 2., 2.4, 2.4, 2.4, 2.4, 2.4],
                [0.4, 0.8, 1.2, 1.6, 2., 2.4, 2.8, 2.8, 2.8, 2.8],
                [0.4, 0.8, 1.2, 1.6, 2., 2.4, 2.8, 3.2, 3.2, 3.2],
                [0.4, 0.8, 1.2, 1.6, 2., 2.4, 2.8, 3.2, 3.6, 3.6],
                [0.4, 0.8, 1.2, 1.6, 2., 2.4, 2.8, 3.2, 3.6, 4.]
            ])

        elif self.name == "n15":
            # Define mu
            self.mu = torch.tensor([
                1.66666667, 2.33333333, 3.0, 3.66666667, 4.33333333, 5.0,
                5.66666667, 6.33333333, 7.0, 7.66666667, 8.33333333, 9.0,
                9.66666667, 10.33333333, 11.0
            ])

            # Define the covariance matrix
            self.cov_matrix = torch.tensor([
                [0.26666667, 0.26666667, 0.26666667, 0.26666667, 0.26666667, 0.26666667, 0.26666667, 0.26666667, 0.26666667, 0.26666667, 0.26666667, 0.26666667, 0.26666667, 0.26666667, 0.26666667],
                [0.26666667, 0.53333333, 0.53333333, 0.53333333, 0.53333333, 0.53333333, 0.53333333, 0.53333333, 0.53333333, 0.53333333, 0.53333333, 0.53333333, 0.53333333, 0.53333333, 0.53333333],
                [0.26666667, 0.53333333, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                [0.26666667, 0.53333333, 0.8, 1.06666667, 1.06666667, 1.06666667, 1.06666667, 1.06666667, 1.06666667, 1.06666667, 1.06666667, 1.06666667, 1.06666667, 1.06666667, 1.06666667],
                [0.26666667, 0.53333333, 0.8, 1.06666667, 1.33333333, 1.33333333, 1.33333333, 1.33333333, 1.33333333, 1.33333333, 1.33333333, 1.33333333, 1.33333333, 1.33333333, 1.33333333],
                [0.26666667, 0.53333333, 0.8, 1.06666667, 1.33333333, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],
                [0.26666667, 0.53333333, 0.8, 1.06666667, 1.33333333, 1.6, 1.86666667, 1.86666667, 1.86666667, 1.86666667, 1.86666667, 1.86666667, 1.86666667, 1.86666667, 1.86666667],
                [0.26666667, 0.53333333, 0.8, 1.06666667, 1.33333333, 1.6, 1.86666667, 2.13333333, 2.13333333, 2.13333333, 2.13333333, 2.13333333, 2.13333333, 2.13333333, 2.13333333],
                [0.26666667, 0.53333333, 0.8, 1.06666667, 1.33333333, 1.6, 1.86666667, 2.13333333, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4],
                [0.26666667, 0.53333333, 0.8, 1.06666667, 1.33333333, 1.6, 1.86666667, 2.13333333, 2.4, 2.66666667, 2.66666667, 2.66666667, 2.66666667, 2.66666667, 2.66666667],
                [0.26666667, 0.53333333, 0.8, 1.06666667, 1.33333333, 1.6, 1.86666667, 2.13333333, 2.4, 2.66666667, 2.93333333, 2.93333333, 2.93333333, 2.93333333, 2.93333333],
                [0.26666667, 0.53333333, 0.8, 1.06666667, 1.33333333, 1.6, 1.86666667, 2.13333333, 2.4, 2.66666667, 2.93333333, 3.2, 3.2, 3.2, 3.2],
                [0.26666667, 0.53333333, 0.8, 1.06666667, 1.33333333, 1.6, 1.86666667, 2.13333333, 2.4, 2.66666667, 2.93333333, 3.2, 3.46666667, 3.46666667, 3.46666667],
                [0.26666667, 0.53333333, 0.8, 1.06666667, 1.33333333, 1.6, 1.86666667, 2.13333333, 2.4, 2.66666667, 2.93333333, 3.2, 3.46666667, 3.73333333, 3.73333333],
                [0.26666667, 0.53333333, 0.8, 1.06666667, 1.33333333, 1.6, 1.86666667, 2.13333333, 2.4, 2.66666667, 2.93333333, 3.2, 3.46666667, 3.73333333, 4.0]
            ])

        elif self.name == "n20":
            # Define mu
            self.mu = torch.tensor([
                1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
                6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0
            ])

            # Define the covariance matrix
            self.cov_matrix = torch.tensor([
                [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                [0.2, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
                [0.2, 0.4, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2],
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4],
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6],
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8],
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2],
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4],
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6],
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8],
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.2, 3.2, 3.2, 3.2],
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.4, 3.4, 3.4],
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.6, 3.6],
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 3.8],
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
            ])
            
        else:
            print("Not a valid name option.")
            return 0

        #Compute inverse cov_matrix
        #Small determinant might give problems
        self.inv_cov_matrix = torch.inverse(self.cov_matrix)

    # Log of probability density
    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        # Assume delta is calculated as follows
        delta = x - self.mu  # Shape: [6000, 5]

        #Calculate the matrix product with inv_cov_matrix
        delta_transformed = torch.matmul(delta, self.inv_cov_matrix)  # Shape: [6000, 5]

        #Compute product
        exponent = torch.matmul(delta_transformed.unsqueeze(1), delta.unsqueeze(2))  # Shape: [6000, 1, 5] x [6000, 5, 1]

        # Reshape to [6000, 6000]
        exponent = exponent.squeeze(1)
        
        # Compute the normalization term
        normalization = self.dim * torch.log(torch.tensor(2 * torch.pi)) + torch.logdet(self.cov_matrix)

        return -0.5*(exponent + normalization)

    # Gradient of the log density (score function)
    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Compute (x - mu)
        delta = x - self.mu

        # Calculate the score
        return -torch.matmul(delta, self.inv_cov_matrix)

    # Returns a univariate normal distribution representing the marginal distribution for the specified dimension.
    def marginal_distr(self, dim=0) -> torch.distributions.Distribution:
        # Extract mean and variance for the specified dimension
        mean = self.mu[dim]  # Mean for the given dimension
        variance = self.cov_matrix[dim, dim]  # Variance for the given dimension

        # Return a univariate Normal distribution
        return distributions.Normal(mean, torch.sqrt(variance))

    # Marginal
    def marginal(self, x: torch.Tensor, dim=0) -> torch.Tensor:
        # Get the marginal distribution for the given dimension
        marginal_distribution = self.marginal_distr(dim=dim)

        # Compute the probability density at the given point x
        return marginal_distribution.log_prob(x).exp()

    # Sample from multivariate normal distribution
    # Seems a bit cheaty 
    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()

        mvn = MultivariateNormal(self.mu, self.cov_matrix)

        return mvn.sample(shape)

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
        self.register_buffer("mu", torch.tensor(mu), persistent=False)
        
        # Initialize sigma as a diagonal matrix with sigma on the diagonal
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
        return -0.5 * self.dim * (x-self.mu)**2/var - 0.5 * (2*math.pi*var).log()

    # Gradient of the log normal distribution
    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return (self.mu-x)/self.sigma**2

    # Marginal distribution for a specific dimension
    def marginal_distr(self, dim=0) -> torch.distributions.Distribution:
        #return distributions.Normal(self.mu, self.sigma)
        return distributions.Normal(self.mu, self.sigma)

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

        #if self.truncate_quartile is None:
        return self.mu + self.sigma * torch.randn(*shape, self.dim, device=self.domain.device)
        
        tensor = torch.empty(*shape, self.dim, device=self.domain.device)
        return trunc_normal_(tensor, mean=self.mu, std=self.sigma, a=self.truncate_quartile[0], b=self.truncate_quartile[1])

# sin^2(x)+2 on [-pi, pi] as pdf
class Periodic(Distribution):
    def __init__(
        self,
        dim: int = 1,
        grid_points: int = 2001,
        domain_delta: float = math.pi,
        **kwargs,
    ):

        super().__init__(dim=dim, grid_points=grid_points, **kwargs)
        
        # Set domain
        if self.domain is None:
            domain = domain_delta * torch.tensor([[-1.0, 1.0]])
            self.set_domain(domain)

    # Log of the probability density
    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.sin(x)**2 + 2)

    # Gradient of the log normal distribution
    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return 2*torch.sin(x)*torch.cos(x)

    # Marginal for all dimensions
    def marginal(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.pdf(x)

    # Sample from the paths
    # A good sampling choice dramatically improves the training efficiency, but requires information on the distribution
    # When we have no idea, best to use uniform sampling
    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()

        a = -math.pi
        b = -a
        
        return a + (b - a) * torch.rand(*shape, self.dim, device=self.domain.device)



class LogNormal(Distribution):
    def __init__(
        self,
        dim: int = 1,
        mu: float = 0.0,
        sigma: float = 1.0,
        truncate_quartile: float | None = None,
        grid_points: int = 2001,
        domain_delta: float = 3,
        **kwargs,
    ):

        super().__init__(dim=dim, grid_points=grid_points, **kwargs)
        
        # Initialize mu as a 1D tensor of size (dim,) with each value being mu
        self.register_buffer("mu", torch.full((dim,), mu), persistent=False)
        
        # Initialize sigma as a diagonal matrix with sigma on the diagonal
        self.register_buffer("sigma", torch.diag_embed(torch.full((dim,), sigma)), persistent=False)

        # Set domain
        if self.domain is None:
            domain = domain_delta * torch.tensor([[0.0001, 1.0]])
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
        # Compute log(x)
        log_x = torch.log(x)

        return -log_x - torch.log(self.sigma) - 1/2*math.log(2*torch.pi) - (log_x - self.mu)**2/(2*self.sigma[0,0]**2)

    # Gradient of the log normal distribution
    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Compute log(x)
        log_x = torch.log(x)
        
        # Compute the exponential term
        exp_term = torch.exp(-((log_x - self.mu[0]) ** 2) / (2 * self.sigma[0,0] ** 2))
        
        # Compute the derivative based on the formula
        pdf_derivative = (exp_term / (x ** 2 * self.sigma[0,0] * torch.sqrt(torch.tensor(2.0 * torch.pi)))) * ((log_x - self.mu[0]) / (self.sigma[0,0] ** 2) - 1)
        
        return pdf_derivative

    # Marginal distribution for a specific dimension
    def marginal_distr(self, dim=0) -> torch.distributions.Distribution:
        #return distributions.Normal(self.mu, self.sigma)
        return distributions.log_normal.LogNormal(self.mu[0], self.sigma[0, 0])

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