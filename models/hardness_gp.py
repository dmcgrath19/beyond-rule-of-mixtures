import math
from dataclasses import dataclass

import gpytorch
import torch

from lib.features import ATOMIC_RADIUS_PM
from lib.physics import SUPPORTED_ELEMENTS
from models.sigma_models import SigmaDiagnostics, SigmaModelSpec, make_sigma_model


@dataclass(frozen=True)
class MechanismDiagnostics:
    sigma: torch.Tensor
    yield_strength_edge: torch.Tensor
    hardness_edge: torch.Tensor
    hardness_mean: torch.Tensor
    yield_strength_screw: torch.Tensor | None = None
    hardness_screw: torch.Tensor | None = None
    edge_probability: torch.Tensor | None = None
    edge_blend_weight: torch.Tensor | None = None
    mechanism_gate_strength: torch.Tensor | None = None
    mechanism_delta: torch.Tensor | None = None
    mechanism_delta_threshold: torch.Tensor | None = None
    mechanism_delta_sharpness: torch.Tensor | None = None
    controlling_is_screw: torch.Tensor | None = None


class BaruffiMechanismDiagnostic(torch.nn.Module):
    """Baruffi-style edge/screw applicability score from atomic-size delta."""

    def __init__(
        self,
        delta_threshold: float = 0.035,
        delta_sharpness: float = 250.0,
        probability_mode: str = "fixed_sigmoid",
    ):
        super().__init__()
        if probability_mode not in {
            "hard_threshold",
            "fixed_sigmoid",
            "learned_threshold",
            "learned_threshold_sharpness",
        }:
            raise ValueError(f"Unknown mechanism probability mode: {probability_mode}")
        self.probability_mode = probability_mode
        if probability_mode in {"learned_threshold", "learned_threshold_sharpness"}:
            self.delta_threshold = torch.nn.Parameter(torch.tensor(delta_threshold, dtype=torch.float64))
        else:
            self.register_buffer("delta_threshold", torch.tensor(delta_threshold, dtype=torch.float64))
        if probability_mode == "learned_threshold_sharpness":
            self.log_delta_sharpness = torch.nn.Parameter(torch.tensor(math.log(delta_sharpness), dtype=torch.float64))
        else:
            self.register_buffer("delta_sharpness", torch.tensor(delta_sharpness, dtype=torch.float64))
        self.register_buffer(
            "base_radii",
            torch.tensor([ATOMIC_RADIUS_PM[element] for element in SUPPORTED_ELEMENTS], dtype=torch.float64),
        )

    def threshold(self) -> torch.Tensor:
        return self.delta_threshold

    def sharpness(self) -> torch.Tensor:
        if self.probability_mode == "learned_threshold_sharpness":
            return self.log_delta_sharpness.exp().clamp(min=1.0, max=1000.0)
        return self.delta_sharpness

    def radius_delta(self, composition: torch.Tensor) -> torch.Tensor:
        average_radius = (composition * self.base_radii).sum(dim=-1, keepdim=True).clamp(min=1e-30)
        return torch.sqrt((composition * (1.0 - self.base_radii / average_radius).pow(2)).sum(dim=-1))

    def forward(self, composition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.radius_delta(composition=composition)
        threshold = self.threshold()
        if self.probability_mode == "hard_threshold":
            edge_probability = (delta >= threshold).to(dtype=composition.dtype)
        else:
            edge_probability = torch.sigmoid(self.sharpness() * (delta - threshold))
        return edge_probability, delta


class CurtinHardnessMean(gpytorch.means.Mean):
    def __init__(
        self,
        physics_start: int,
        composition_start: int,
        n_composition_dims: int,
        sigma_model_spec: SigmaModelSpec,
        use_dual_mechanism: bool = False,
        mechanism_mode: str = "minimum",
        mechanism_delta_threshold: float = 0.035,
        mechanism_probability_mode: str = "fixed_sigmoid",
    ):
        super().__init__()
        if mechanism_mode not in {"minimum", "diagnostic", "blend", "gated"}:
            raise ValueError(f"Unknown mechanism mode: {mechanism_mode}")
        if mechanism_mode in {"diagnostic", "blend", "gated"} and not use_dual_mechanism:
            raise ValueError(f"{mechanism_mode} mechanism mode requires use_dual_mechanism=True")
        self.p = physics_start
        self.composition_start = composition_start
        self.n_composition_dims = n_composition_dims
        self.use_dual_mechanism = use_dual_mechanism
        self.mechanism_mode = mechanism_mode
        self.log_alpha = torch.nn.Parameter(torch.tensor(math.log(0.04), dtype=torch.float64))
        self.log_taylor = torch.nn.Parameter(torch.tensor(math.log(3.0), dtype=torch.float64))
        self.log_hv_scale = torch.nn.Parameter(torch.tensor(math.log(3.0 / 9.81), dtype=torch.float64))
        self.log_thermal_inv_c = torch.nn.Parameter(torch.tensor(math.log(1.0 / 0.55), dtype=torch.float64))
        self.log_thermal_exp = torch.nn.Parameter(torch.tensor(math.log(0.91), dtype=torch.float64))
        self.log_line_tension = torch.nn.Parameter(torch.tensor(math.log(1.0 / 12.0), dtype=torch.float64))
        if use_dual_mechanism:
            self.log_line_tension_screw = torch.nn.Parameter(torch.tensor(math.log(0.5), dtype=torch.float64))
        if mechanism_mode in {"diagnostic", "blend", "gated"}:
            self.mechanism_diagnostic = BaruffiMechanismDiagnostic(
                delta_threshold=mechanism_delta_threshold,
                probability_mode=mechanism_probability_mode,
            )
        if mechanism_mode == "gated":
            initial_gate_strength = 0.10
            initial_logit = math.log(initial_gate_strength / (1.0 - initial_gate_strength))
            self.logit_mechanism_gate_strength = torch.nn.Parameter(torch.tensor(initial_logit, dtype=torch.float64))
        self.sigma_model = make_sigma_model(n_composition_dims=n_composition_dims, spec=sigma_model_spec)

    def _composition(self, x: torch.Tensor) -> torch.Tensor:
        return x[
            ...,
            self.composition_start:self.composition_start + self.n_composition_dims,
        ]

    def sigma(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigma_diagnostics(x).sigma

    def sigma_diagnostics(self, x: torch.Tensor) -> SigmaDiagnostics:
        sigma_rom = x[..., self.p + 3].clamp(min=1e-30)
        return self.sigma_model.diagnostics(sigma_rom=sigma_rom, composition=self._composition(x))

    def mechanism_diagnostics(self, x: torch.Tensor) -> MechanismDiagnostics:
        alpha = self.log_alpha.exp()
        taylor = self.log_taylor.exp()
        hv_scale = self.log_hv_scale.exp()
        inv_c = self.log_thermal_inv_c.exp()
        thermal_exp = self.log_thermal_exp.exp()

        shear_modulus = x[..., self.p]
        poisson = x[..., self.p + 1]
        burgers = x[..., self.p + 2]
        temperature = x[..., self.p + 4]
        sigma = self.sigma(x)

        ys_t_edge = self._yield_stress_at_temperature(
            line_tension=self.log_line_tension.exp(),
            shear_modulus=shear_modulus,
            poisson=poisson,
            burgers=burgers,
            sigma=sigma,
            temperature=temperature,
            alpha=alpha,
            taylor=taylor,
            inv_c=inv_c,
            thermal_exp=thermal_exp,
        )
        hardness_edge = ys_t_edge * hv_scale

        if not self.use_dual_mechanism:
            return MechanismDiagnostics(
                sigma=sigma,
                yield_strength_edge=ys_t_edge,
                hardness_edge=hardness_edge,
                hardness_mean=hardness_edge,
            )

        ys_t_screw = self._yield_stress_at_temperature(
            line_tension=self.log_line_tension_screw.exp(),
            shear_modulus=shear_modulus,
            poisson=poisson,
            burgers=burgers,
            sigma=sigma,
            temperature=temperature,
            alpha=alpha,
            taylor=taylor,
            inv_c=inv_c,
            thermal_exp=thermal_exp,
        )
        hardness_screw = ys_t_screw * hv_scale

        edge_blend_weight = None
        mechanism_gate_strength = None
        mechanism_delta_threshold = None
        mechanism_delta_sharpness = None
        if self.mechanism_mode in {"diagnostic", "blend", "gated"}:
            edge_probability, mechanism_delta = self.mechanism_diagnostic(self._composition(x))
            mechanism_delta_threshold = torch.ones_like(edge_probability) * self.mechanism_diagnostic.threshold()
            mechanism_delta_sharpness = torch.ones_like(edge_probability) * self.mechanism_diagnostic.sharpness()
            controlling_is_screw = edge_probability < 0.5
            if self.mechanism_mode == "diagnostic":
                edge_blend_weight = torch.ones_like(edge_probability)
                hardness_mean = hardness_edge
            elif self.mechanism_mode == "blend":
                edge_blend_weight = edge_probability
                hardness_mean = edge_blend_weight * hardness_edge + (1.0 - edge_blend_weight) * hardness_screw
            else:
                mechanism_gate_strength = torch.sigmoid(self.logit_mechanism_gate_strength)
                edge_blend_weight = 1.0 - mechanism_gate_strength * (1.0 - edge_probability)
                hardness_mean = edge_blend_weight * hardness_edge + (1.0 - edge_blend_weight) * hardness_screw
        else:
            controlling_is_screw = ys_t_screw < ys_t_edge
            edge_probability = (~controlling_is_screw).to(dtype=hardness_edge.dtype)
            edge_blend_weight = edge_probability
            mechanism_delta = None
            hardness_mean = torch.minimum(hardness_edge, hardness_screw)

        return MechanismDiagnostics(
            sigma=sigma,
            yield_strength_edge=ys_t_edge,
            yield_strength_screw=ys_t_screw,
            hardness_edge=hardness_edge,
            hardness_screw=hardness_screw,
            hardness_mean=hardness_mean,
            edge_probability=edge_probability,
            edge_blend_weight=edge_blend_weight,
            mechanism_gate_strength=(
                None
                if mechanism_gate_strength is None
                else torch.ones_like(hardness_mean) * mechanism_gate_strength
            ),
            mechanism_delta=mechanism_delta,
            mechanism_delta_threshold=mechanism_delta_threshold,
            mechanism_delta_sharpness=mechanism_delta_sharpness,
            controlling_is_screw=controlling_is_screw,
        )

    def _yield_stress_at_temperature(
        self,
        line_tension: torch.Tensor,
        shear_modulus: torch.Tensor,
        poisson: torch.Tensor,
        burgers: torch.Tensor,
        sigma: torch.Tensor,
        temperature: torch.Tensor,
        alpha: torch.Tensor,
        taylor: torch.Tensor,
        inv_c: torch.Tensor,
        thermal_exp: torch.Tensor,
    ) -> torch.Tensor:
        ys_0k = (
            alpha
            * line_tension.pow(-1.0 / 3.0)
            * shear_modulus
            * ((1 + poisson) / (1 - poisson)) ** (4.0 / 3.0)
            * (sigma / burgers.pow(6)).pow(2.0 / 3.0)
            * 1000.0
            * taylor
        )
        activation_energy = (
            2.0
            * line_tension.pow(1.0 / 3.0)
            * shear_modulus
            * burgers.pow(3)
            * ((1 + poisson) / (1 - poisson)) ** (2.0 / 3.0)
            * (sigma / burgers.pow(6)).pow(1.0 / 3.0)
            / 160.21766208
        ).clamp(min=1e-30)
        thermal_arg = ((8.617333262e-5 * temperature / activation_energy) * math.log(1e7)).clamp(min=1e-30)
        return ys_0k * torch.exp(-inv_c * thermal_arg.pow(thermal_exp))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mechanism_diagnostics(x).hardness_mean


class HardnessGP(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        n_features: int,
        physics_start: int,
        composition_start: int,
        n_composition_dims: int,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        sigma_model_spec: SigmaModelSpec | None = None,
        use_dual_mechanism: bool = False,
        mechanism_mode: str = "minimum",
        mechanism_delta_threshold: float = 0.035,
        mechanism_probability_mode: str = "fixed_sigmoid",
        mean_type: str = "physics",
    ):
        super().__init__(train_x, train_y, likelihood)
        if mean_type not in ("physics", "constant"):
            raise ValueError(f"Unknown mean_type {mean_type!r}")
        self.mean_type = mean_type
        self.constant_mean = torch.nn.Parameter(torch.zeros(()))
        if sigma_model_spec is None:
            sigma_model_spec = SigmaModelSpec()
        self.mean_module = CurtinHardnessMean(
            physics_start=physics_start,
            composition_start=composition_start,
            n_composition_dims=n_composition_dims,
            sigma_model_spec=sigma_model_spec,
            use_dual_mechanism=use_dual_mechanism,
            mechanism_mode=mechanism_mode,
            mechanism_delta_threshold=mechanism_delta_threshold,
            mechanism_probability_mode=mechanism_probability_mode,
        )
        self.register_buffer("y_mean", y_mean)
        self.register_buffer("y_std", y_std)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=n_features,
                active_dims=list(range(n_features)),
            )
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        if self.mean_type == "constant":
            # Composition-only control: same kernel and descriptors, no physics prior.
            mean_x = self.constant_mean.expand(x.shape[0])
        else:
            raw_mean = self.mean_module(x)
            mean_x = (raw_mean - self.y_mean) / self.y_std
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
