from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    use_gpytorch: bool = False
    use_vela_ys: bool = False
    mean_type: str = "physics"
    sigma_variant: str = "analytic"
    sigma_hidden_dim: int = 8
    sigma_log_bound: float = 0.10
    stage1_lr: float = 0.05
    stage2_lr: float = 0.01
    sigma_stage2_lr: float = 0.01
    sigma_weight_decay: float = 0.0
    sigma_warmup_iters: int = 0
    include_xrd_peak_count: bool = False
    include_microstructure: bool = False
    sklearn_model: str | None = None
    use_dual_mechanism: bool = False
    mechanism_mode: str = "minimum"
    mechanism_delta_threshold: float = 0.035
    mechanism_probability_mode: str = "fixed_sigmoid"
    kernel: str = "rbf"


def _effective_volume_config(name: str, **overrides: object) -> ModelConfig:
    values = {
        "name": name,
        "use_gpytorch": True,
        "sigma_variant": "effective_volume",
        "sigma_hidden_dim": 4,
        "sigma_log_bound": 0.05,
        "stage2_lr": 0.005,
        "sigma_stage2_lr": 0.001,
        "sigma_weight_decay": 1e-2,
        "sigma_warmup_iters": 600,
    }
    values.update(overrides)
    return ModelConfig(**values)


def get_model_config(model_name: str) -> ModelConfig:
    if model_name == "curtin":
        return ModelConfig(name="curtin")
    if model_name == "gao_rf":
        return ModelConfig(name="gao_rf", sklearn_model="gao_rf")
    if model_name == "gao_svr":
        return ModelConfig(name="gao_svr", sklearn_model="gao_svr")
    if model_name == "gpytorch":
        return ModelConfig(name="gpytorch", use_gpytorch=True)
    if model_name == "gpytorch_nonlinear_sigma":
        return _effective_volume_config(name="gpytorch_nonlinear_sigma")
    if model_name == "gpytorch_polynomial_sigma":
        return ModelConfig(
            name="gpytorch_polynomial_sigma",
            use_gpytorch=True,
            sigma_variant="polynomial_volume",
            sigma_log_bound=0.05,
            stage2_lr=0.005,
            sigma_stage2_lr=0.001,
            sigma_weight_decay=1e-2,
            sigma_warmup_iters=600,
        )
    if model_name == "vela_ys":
        return ModelConfig(name="vela_ys", use_vela_ys=True)
    if model_name == "xrd_peak_count":
        return ModelConfig(
            name="xrd_peak_count",
            use_gpytorch=True,
            include_xrd_peak_count=True,
        )
    if model_name == "microstructure":
        return ModelConfig(
            name="microstructure",
            use_gpytorch=True,
            include_microstructure=True,
        )
    if model_name == "xrd_and_microstructure":
        return ModelConfig(
            name="xrd_and_microstructure",
            use_gpytorch=True,
            include_xrd_peak_count=True,
            include_microstructure=True,
        )
    if model_name == "gpytorch_dual":
        return ModelConfig(
            name="gpytorch_dual",
            use_gpytorch=True,
            use_dual_mechanism=True,
        )
    if model_name == "gpytorch_nonlinear_sigma_dual":
        return _effective_volume_config(
            name="gpytorch_nonlinear_sigma_dual",
            use_dual_mechanism=True,
        )
    if model_name == "gpytorch_nonlinear_sigma_dual_diagnostic":
        return _effective_volume_config(
            name="gpytorch_nonlinear_sigma_dual_diagnostic",
            use_dual_mechanism=True,
            mechanism_mode="diagnostic",
        )
    if model_name == "gpytorch_nonlinear_sigma_dual_gated":
        return _effective_volume_config(
            name="gpytorch_nonlinear_sigma_dual_gated",
            use_dual_mechanism=True,
            mechanism_mode="gated",
        )
    if model_name == "gpytorch_polynomial_sigma_dual":
        return ModelConfig(
            name="gpytorch_polynomial_sigma_dual",
            use_gpytorch=True,
            use_dual_mechanism=True,
            sigma_variant="polynomial_volume",
            sigma_log_bound=0.05,
            stage2_lr=0.005,
            sigma_stage2_lr=0.001,
            sigma_weight_decay=1e-2,
            sigma_warmup_iters=600,
        )
    raise ValueError(f"Unknown model config: {model_name}")
