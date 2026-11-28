"""Đối chiếu optimizer tự cài với sklearn trên dữ liệu tổng hợp nhỏ."""

import numpy as np
from scipy import sparse
from sklearn.linear_model import Lasso, Ridge

from .ridge_lasso import fit_numpy


def main() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(600, 8))
    true_weights = np.array([2.0, -1.5, 0.7, 0, 0, 0, 0.3, 0])
    y = 3.4 + x @ true_weights + rng.normal(scale=0.05, size=len(x))
    x_sparse = sparse.csr_matrix(x)

    for penalty, alpha in [("Lasso", 0.03), ("Ridge", 0.02)]:
        weights, _, _ = fit_numpy(x_sparse, y, alpha, penalty, max_iterations=10_000)
        if penalty == "Lasso":
            reference = Lasso(alpha=alpha, fit_intercept=True, tol=1e-12, max_iter=100_000).fit(x, y)
        else:
            reference = Ridge(alpha=len(y) * alpha, fit_intercept=True, tol=1e-12).fit(x, y)
        np.testing.assert_allclose(weights[0], reference.intercept_, atol=2e-3, rtol=2e-3)
        np.testing.assert_allclose(weights[1:], reference.coef_, atol=2e-3, rtol=2e-3)
    print("NumPy FISTA coefficients match sklearn reference for Ridge and Lasso.")


if __name__ == "__main__":
    main()
