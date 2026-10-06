"""Gao et al. (2023) baseline models for hardness prediction.

Hyperparameters from: 'Machine learning prediction of hardness in solid
solution high entropy alloys'
https://www.sciencedirect.com/science/article/abs/pii/S2352492823017932

RF:  maxDepth=15, numFeatures=5, numIterations=76
SVR: epsilon-SVR, kernel=RBF, C=400, gamma=2.9, epsilon=0.1, tol=0.001
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def make_gao_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        max_depth=15,
        max_features=5,
        n_estimators=76,
        random_state=42,
    )


def make_gao_svr() -> SVR:
    return SVR(
        kernel="rbf",
        C=400.0,
        gamma=2.9,
        epsilon=0.1,
        tol=0.001,
    )
