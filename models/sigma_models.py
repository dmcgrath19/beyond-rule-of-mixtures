from dataclasses import dataclass

import torch

from lib.physics import ELEMENTAL_BCC_VOLUMES


@dataclass(frozen=True)
class SigmaModelSpec:
    variant: str = "analytic"
    hidden_dim: int = 8
    log_bound: float = 0.10


@dataclass(frozen=True)
class SigmaDiagnostics:
    sigma: torch.Tensor
    effective_volumes: torch.Tensor | None = None
    delta_volumes: torch.Tensor | None = None
    contributions: torch.Tensor | None = None


class AnalyticSigmaModel(torch.nn.Module):
    def forward(self, sigma_rom: torch.Tensor, composition: torch.Tensor) -> torch.Tensor:
        del composition
        return sigma_rom.clamp(min=1e-30)

    def diagnostics(self, sigma_rom: torch.Tensor, composition: torch.Tensor) -> SigmaDiagnostics:
        del composition
        return SigmaDiagnostics(sigma=sigma_rom.clamp(min=1e-30))


class SharedSigmaModel(torch.nn.Module):
    def __init__(self, n_composition_dims: int, hidden_dim: int, log_bound: float):
        super().__init__()
        self.log_bound = log_bound
        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_composition_dims, hidden_dim, dtype=torch.float64),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1, dtype=torch.float64),
        )
        torch.nn.init.xavier_uniform_(self.network[0].weight)
        torch.nn.init.zeros_(self.network[0].bias)
        torch.nn.init.zeros_(self.network[2].weight)
        torch.nn.init.zeros_(self.network[2].bias)

    def forward(self, sigma_rom: torch.Tensor, composition: torch.Tensor) -> torch.Tensor:
        log_sigma_residual = self.log_bound * torch.tanh(self.network(composition).squeeze(-1))
        return sigma_rom.clamp(min=1e-30) * torch.exp(log_sigma_residual)

    def diagnostics(self, sigma_rom: torch.Tensor, composition: torch.Tensor) -> SigmaDiagnostics:
        sigma = self.forward(sigma_rom=sigma_rom, composition=composition)
        return SigmaDiagnostics(sigma=sigma)


class AffineVolumeSigmaModel(torch.nn.Module):
    def __init__(self, n_composition_dims: int, log_bound: float):
        super().__init__()
        self.log_bound = log_bound
        self.log_volume_offsets = torch.nn.Parameter(torch.zeros(n_composition_dims, dtype=torch.float64))
        self.register_buffer(
            "base_volumes",
            torch.tensor(ELEMENTAL_BCC_VOLUMES, dtype=torch.float64),
        )

    def _effective_volumes(self, composition: torch.Tensor) -> torch.Tensor:
        del composition
        log_volume_residual = self.log_bound * torch.tanh(self.log_volume_offsets)
        return self.base_volumes * torch.exp(log_volume_residual)

    def forward(self, sigma_rom: torch.Tensor, composition: torch.Tensor) -> torch.Tensor:
        del sigma_rom
        effective_volumes = self._effective_volumes(composition=composition)
        equilibrium_volume = (composition * effective_volumes).sum(dim=-1, keepdim=True)
        delta_volumes = effective_volumes - equilibrium_volume
        sigma = (composition * delta_volumes.pow(2)).sum(dim=-1)
        return sigma.clamp(min=1e-30)

    def diagnostics(self, sigma_rom: torch.Tensor, composition: torch.Tensor) -> SigmaDiagnostics:
        del sigma_rom
        effective_volumes = self._effective_volumes(composition=composition)
        equilibrium_volume = (composition * effective_volumes).sum(dim=-1, keepdim=True)
        delta_volumes = effective_volumes - equilibrium_volume
        contributions = composition * delta_volumes.pow(2)
        sigma = contributions.sum(dim=-1).clamp(min=1e-30)
        return SigmaDiagnostics(
            sigma=sigma,
            effective_volumes=effective_volumes.expand_as(composition),
            delta_volumes=delta_volumes,
            contributions=contributions,
        )


class EffectiveVolumeSigmaModel(torch.nn.Module):
    def __init__(self, n_composition_dims: int, hidden_dim: int, log_bound: float):
        super().__init__()
        self.log_bound = log_bound
        self.context = torch.nn.Sequential(
            torch.nn.Linear(n_composition_dims, hidden_dim, dtype=torch.float64),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, n_composition_dims, dtype=torch.float64),
        )
        torch.nn.init.xavier_uniform_(self.context[0].weight)
        torch.nn.init.zeros_(self.context[0].bias)
        torch.nn.init.zeros_(self.context[2].weight)
        torch.nn.init.zeros_(self.context[2].bias)
        self.register_buffer(
            "base_volumes",
            torch.tensor(ELEMENTAL_BCC_VOLUMES, dtype=torch.float64),
        )

    def forward(self, sigma_rom: torch.Tensor, composition: torch.Tensor) -> torch.Tensor:
        del sigma_rom
        log_volume_residual = self.log_bound * torch.tanh(self.context(composition))
        effective_volumes = self.base_volumes * torch.exp(log_volume_residual)
        equilibrium_volume = (composition * effective_volumes).sum(dim=-1, keepdim=True)
        delta_volumes = effective_volumes - equilibrium_volume
        sigma = (composition * delta_volumes.pow(2)).sum(dim=-1)
        return sigma.clamp(min=1e-30)

    def diagnostics(self, sigma_rom: torch.Tensor, composition: torch.Tensor) -> SigmaDiagnostics:
        del sigma_rom
        log_volume_residual = self.log_bound * torch.tanh(self.context(composition))
        effective_volumes = self.base_volumes * torch.exp(log_volume_residual)
        equilibrium_volume = (composition * effective_volumes).sum(dim=-1, keepdim=True)
        delta_volumes = effective_volumes - equilibrium_volume
        contributions = composition * delta_volumes.pow(2)
        sigma = contributions.sum(dim=-1).clamp(min=1e-30)
        return SigmaDiagnostics(
            sigma=sigma,
            effective_volumes=effective_volumes,
            delta_volumes=delta_volumes,
            contributions=contributions,
        )


class PolynomialVolumeSigmaModel(torch.nn.Module):
    def __init__(self, n_composition_dims: int, log_bound: float):
        super().__init__()
        self.log_bound = log_bound
        self.n_composition_dims = n_composition_dims
        basis_dim = 1 + n_composition_dims + (n_composition_dims * (n_composition_dims + 1)) // 2
        self.linear = torch.nn.Linear(basis_dim, n_composition_dims, bias=False, dtype=torch.float64)
        torch.nn.init.zeros_(self.linear.weight)
        self.register_buffer(
            "base_volumes",
            torch.tensor(ELEMENTAL_BCC_VOLUMES, dtype=torch.float64),
        )
        self.register_buffer(
            "quadratic_index",
            torch.triu_indices(n_composition_dims, n_composition_dims),
        )

    def _basis(self, composition: torch.Tensor) -> torch.Tensor:
        ones = torch.ones_like(composition[..., :1])
        outer = composition.unsqueeze(-1) * composition.unsqueeze(-2)
        quadratic = outer[..., self.quadratic_index[0], self.quadratic_index[1]]
        return torch.cat([ones, composition, quadratic], dim=-1)

    def forward(self, sigma_rom: torch.Tensor, composition: torch.Tensor) -> torch.Tensor:
        del sigma_rom
        basis = self._basis(composition)
        log_volume_residual = self.log_bound * torch.tanh(self.linear(basis))
        effective_volumes = self.base_volumes * torch.exp(log_volume_residual)
        equilibrium_volume = (composition * effective_volumes).sum(dim=-1, keepdim=True)
        delta_volumes = effective_volumes - equilibrium_volume
        sigma = (composition * delta_volumes.pow(2)).sum(dim=-1)
        return sigma.clamp(min=1e-30)

    def diagnostics(self, sigma_rom: torch.Tensor, composition: torch.Tensor) -> SigmaDiagnostics:
        del sigma_rom
        basis = self._basis(composition)
        log_volume_residual = self.log_bound * torch.tanh(self.linear(basis))
        effective_volumes = self.base_volumes * torch.exp(log_volume_residual)
        equilibrium_volume = (composition * effective_volumes).sum(dim=-1, keepdim=True)
        delta_volumes = effective_volumes - equilibrium_volume
        contributions = composition * delta_volumes.pow(2)
        sigma = contributions.sum(dim=-1).clamp(min=1e-30)
        return SigmaDiagnostics(
            sigma=sigma,
            effective_volumes=effective_volumes,
            delta_volumes=delta_volumes,
            contributions=contributions,
        )


def make_sigma_model(n_composition_dims: int, spec: SigmaModelSpec) -> torch.nn.Module:
    if spec.variant == "analytic":
        return AnalyticSigmaModel()
    if spec.variant == "shared":
        return SharedSigmaModel(
            n_composition_dims=n_composition_dims,
            hidden_dim=spec.hidden_dim,
            log_bound=spec.log_bound,
        )
    if spec.variant == "affine_volume":
        return AffineVolumeSigmaModel(
            n_composition_dims=n_composition_dims,
            log_bound=spec.log_bound,
        )
    if spec.variant == "effective_volume":
        return EffectiveVolumeSigmaModel(
            n_composition_dims=n_composition_dims,
            hidden_dim=spec.hidden_dim,
            log_bound=spec.log_bound,
        )
    if spec.variant == "polynomial_volume":
        return PolynomialVolumeSigmaModel(
            n_composition_dims=n_composition_dims,
            log_bound=spec.log_bound,
        )
    raise ValueError(f"Unknown sigma model variant: {spec.variant}")
