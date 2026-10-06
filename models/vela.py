from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class VelaFeatureSelection:
    hv_feature_cols: list[str]
    ys_feature_cols: list[str]


@dataclass(frozen=True)
class VelaFoldIndices:
    repeat_id: int
    fold_id: int
    train_index: np.ndarray
    test_index: np.ndarray


@dataclass(frozen=True)
class ResidualPrediction:
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class VelaFoldMetrics:
    repeat_id: int
    fold_id: int
    n_train: int
    n_test: int
    mae_mpa: float
    rmse_mpa: float
    spearman_r: float
    kendall_tau: float
    hv_feature_cols: str
    ys_feature_cols: str


def select_vela_features(
    df: pd.DataFrame,
    target_col: str,
    candidate_cols: list[str],
    forced_cols: list[str],
    max_feature_count: int,
    corr_threshold: float,
) -> list[str]:
    train = df.dropna(subset=candidate_cols + [target_col]).reset_index(drop=True)
    X = train[candidate_cols].to_numpy()
    y = train[target_col].to_numpy()
    mutual_info = mutual_info_regression(X, y, random_state=0)

    ranked = pd.DataFrame({
        "feature": candidate_cols,
        "score": mutual_info,
    }).sort_values(["score", "feature"], ascending=[False, True])

    selected: list[str] = []
    corr = train[candidate_cols].corr().abs()
    for feature_name in ranked["feature"]:
        if len(selected) >= max_feature_count:
            break
        too_correlated = False
        for selected_name in selected:
            if float(corr.loc[feature_name, selected_name]) > corr_threshold:
                too_correlated = True
                break
        if not too_correlated:
            selected.append(str(feature_name))

    for forced_name in forced_cols:
        if forced_name not in selected:
            selected.append(forced_name)
    return selected


def select_vela_feature_sets(
    hv_df: pd.DataFrame,
    ys_df: pd.DataFrame,
    candidate_cols: list[str],
    forced_cols: list[str],
    max_feature_count: int = 8,
    corr_threshold: float = 0.85,
) -> VelaFeatureSelection:
    hv_feature_cols = select_vela_features(
        hv_df,
        target_col="HV",
        candidate_cols=candidate_cols,
        forced_cols=forced_cols,
        max_feature_count=max_feature_count,
        corr_threshold=corr_threshold,
    )
    ys_feature_cols = select_vela_features(
        ys_df,
        target_col="YS_MPa",
        candidate_cols=candidate_cols,
        forced_cols=forced_cols,
        max_feature_count=max_feature_count,
        corr_threshold=corr_threshold,
    )
    return VelaFeatureSelection(
        hv_feature_cols=hv_feature_cols,
        ys_feature_cols=ys_feature_cols,
    )


def build_repeated_grouped_twofolds(
    groups: pd.Series,
    n_repeats: int,
    seed: int,
) -> list[VelaFoldIndices]:
    value_counts = groups.value_counts()
    unique_groups = value_counts.index.to_list()
    folds: list[VelaFoldIndices] = []

    for repeat_id in range(n_repeats):
        rng = np.random.default_rng(seed + repeat_id)
        shuffled_groups = list(unique_groups)
        rng.shuffle(shuffled_groups)
        ordered_groups = sorted(
            shuffled_groups,
            key=lambda group_name: (int(value_counts.loc[group_name]), group_name),
            reverse=True,
        )

        fold_a_groups: list[str] = []
        fold_b_groups: list[str] = []
        fold_a_size = 0
        fold_b_size = 0
        for group_name in ordered_groups:
            group_size = int(value_counts.loc[group_name])
            if fold_a_size <= fold_b_size:
                fold_a_groups.append(group_name)
                fold_a_size += group_size
            else:
                fold_b_groups.append(group_name)
                fold_b_size += group_size

        for fold_id, test_groups in enumerate([fold_a_groups, fold_b_groups]):
            train_groups = fold_b_groups if fold_id == 0 else fold_a_groups
            train_mask = groups.isin(train_groups).to_numpy()
            test_mask = groups.isin(test_groups).to_numpy()
            folds.append(
                VelaFoldIndices(
                    repeat_id=repeat_id,
                    fold_id=fold_id,
                    train_index=np.flatnonzero(train_mask),
                    test_index=np.flatnonzero(test_mask),
                )
            )
    return folds


def fit_residual_gp_and_predict(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    prior_col: str,
    random_state: int,
) -> ResidualPrediction:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].to_numpy())
    X_pred = scaler.transform(pred_df[feature_cols].to_numpy())
    y_train = train_df[target_col].to_numpy() - train_df[prior_col].to_numpy()

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(len(feature_cols)), length_scale_bounds=(1e-2, 1e4))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e2))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=random_state,
    )
    gp.fit(X_train, y_train)
    residual_mean, residual_std = gp.predict(X_pred, return_std=True)
    prior = pred_df[prior_col].to_numpy()
    return ResidualPrediction(
        mean=prior + residual_mean,
        std=residual_std,
    )


def metrics_from_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    repeat_id: int,
    fold_id: int,
    n_train: int,
    hv_feature_cols: list[str],
    ys_feature_cols: list[str],
) -> VelaFoldMetrics:
    residual = actual - predicted
    actual_series = pd.Series(actual)
    predicted_series = pd.Series(predicted)
    return VelaFoldMetrics(
        repeat_id=repeat_id,
        fold_id=fold_id,
        n_train=n_train,
        n_test=len(actual),
        mae_mpa=float(np.mean(np.abs(residual))),
        rmse_mpa=float(np.sqrt(np.mean(residual ** 2))),
        spearman_r=float(actual_series.corr(predicted_series, method="spearman")),
        kendall_tau=float(actual_series.corr(predicted_series, method="kendall")),
        hv_feature_cols=" | ".join(hv_feature_cols),
        ys_feature_cols=" | ".join(ys_feature_cols),
    )
