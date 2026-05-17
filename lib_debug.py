import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube, scale
from scipy.stats import norm

plt.style.use("./graph_preset.mplstyle")

# -------------------- Test Functions --------------------


def test_1D(x: np.ndarray) -> np.ndarray:
    """True function to be approximated (1D)."""
    return np.cos(2 * np.pi * x) - 0.3 * np.sin(8 * np.pi * x) + 0.5 * x


def ackley_2d(x: np.ndarray) -> np.ndarray:
    """Ackley function (2D). Global Minimum: f(0,0) = 0."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x1 = x[:, 0]
    x2 = x[:, 1]
    a = 20
    b = 0.2
    c = 2 * np.pi

    sum_sq = x1**2 + x2**2
    cos_sum = np.cos(c * x1) + np.cos(c * x2)

    term1 = -a * np.exp(-b * np.sqrt(0.5 * sum_sq))
    term2 = -np.exp(0.5 * cos_sum)

    return term1 + term2 + a + np.exp(1)


def griewank_2d(x: np.ndarray) -> np.ndarray:
    """Griewank function (2D). Global Minimum: f(0,0) = 0."""
    if x.ndim == 1:
        x = x.reshape(1, -1)

    x1 = x[:, 0]
    x2 = x[:, 1]

    term_sum = (x1**2 + x2**2) / 4000

    term_prod = np.cos(x1) * np.cos(x2 / np.sqrt(2))

    return 1 + term_sum - term_prod


def ackley_nd(x: np.ndarray) -> np.ndarray:

    if x.ndim == 1:
        x = x.reshape(1, -1)

    n_samples, d = x.shape

    a = 20
    b = 0.2
    c = 2 * np.pi

    sum_sq = np.sum(x**2, axis=1)
    cos_sum = np.sum(np.cos(c * x), axis=1)

    term1 = -a * np.exp(-b * np.sqrt((1 / d) * sum_sq))
    term2 = -np.exp((1 / d) * cos_sum)

    return term1 + term2 + a + np.exp(1)


def griewank_nd(x: np.ndarray) -> np.ndarray:

    if x.ndim == 1:
        x = x.reshape(1, -1)

    n_samples, d = x.shape

    sum_sq = np.sum(x**2, axis=1)
    term_sum = sum_sq / 4000

    i_array = np.arange(1, d + 1)

    cos_terms = np.cos(x / np.sqrt(i_array))

    term_prod = np.prod(cos_terms, axis=1)

    return 1 + term_sum - term_prod


# -------------------- Core GP Functions --------------------


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, gamma: float) -> np.ndarray:
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) - 2 * np.dot(x1, x2.T) + np.sum(x2**2, 1)
    return np.exp(-gamma * sqdist)


def LHSsampler(dims, n_samples, lower_bounds, upper_bounds):
    sampler = LatinHypercube(d=dims, seed=42)
    samples_continuous = sampler.random(n=n_samples)
    X_initial = scale(samples_continuous, lower_bounds, upper_bounds)
    return X_initial


def fit_gp(kernel, X_train, gamma, noise_var):
    """Calculates Ky_inv with jitter for stability."""
    K = kernel(X_train, X_train, gamma)
    Ky = K + noise_var * np.identity(len(X_train)) + 1e-6 * np.identity(len(X_train))
    try:
        Ky_inv = np.linalg.inv(Ky)
    except np.linalg.LinAlgError:
        Ky_inv = np.linalg.inv(Ky + 1e-5 * np.identity(len(X_train)))
    return Ky_inv


def negative_log_marginal_likelihood(params, X, y, noise_var):
    """
    Negative Log Marginal Likelihood for Hyperparameter Tuning.
    Assumes params[0] is gamma (length scale parameter).
    """
    gamma = params[0]
    if gamma <= 0:
        return np.inf

    n = len(X)
    # Note: Hardcoded rbf_kernel usage here for simplicity within optimization
    K = rbf_kernel(X, X, gamma)
    Ky = K + noise_var * np.identity(n) + 1e-6 * np.identity(n)

    try:
        L = np.linalg.cholesky(Ky)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        log_det_Ky = 2 * np.sum(np.log(np.diag(L)))
        likelihood = (
            0.5 * (y.T @ alpha) + 0.5 * log_det_Ky + 0.5 * n * np.log(2 * np.pi)
        )
        return likelihood.item()
    except np.linalg.LinAlgError:
        return np.inf


# -------------------- Acquisition Functions --------------------


def get_posterior(kernel, x_new, X_sample, y_sample, Ky_opt_inv, length_scale):
    x_new = x_new.reshape(1, -1)
    K_star = kernel(X_sample, x_new, length_scale)
    mu_post = K_star.T @ Ky_opt_inv @ y_sample
    K_star_star = 1.0
    cov_post = K_star_star - K_star.T @ Ky_opt_inv @ K_star
    s2_post = np.maximum(0, cov_post.item())
    return mu_post.item(), np.sqrt(s2_post)


def expected_improvement(
    kernel, x_new, X_sample, y_sample, Ky_opt_inv, length_scale, xi=0.01
):
    mu, sigma = get_posterior(
        kernel, x_new, X_sample, y_sample, Ky_opt_inv, length_scale
    )
    y_best = np.min(y_sample)
    if sigma == 0:
        return 0
    imp = y_best - mu - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei


def lower_confidence_bound(
    kernel, x_new, X_sample, y_sample, Ky_opt_inv, length_scale, kappa=2.0
):
    mu, sigma = get_posterior(
        kernel, x_new, X_sample, y_sample, Ky_opt_inv, length_scale
    )
    # LCB (minimization) wants small mu - kappa*sigma.
    # We return NEGATIVE LCB because optimize_acquisition (below) maximizes the return value.
    return -mu + kappa * sigma


# -------------------- Optimization Function (NEW) --------------------


def optimize_acquisition(
    kernel,
    acq_func,
    X_sample,
    y_sample,
    Ky_opt_inv,
    length_scale,
    lower_bounds,
    upper_bounds,
    acq_params,
):
    """
    Optimizes the acquisition function using L-BFGS-B with random restarts.
    Adapted to accept 'kernel' and generic 'acq_func' (EI or LCB).
    """
    dims = len(lower_bounds)
    best_acq_value = -np.inf
    best_x = None
    n_restarts = 25
    bounds = list(zip(lower_bounds, upper_bounds))

    # Wrapper to negate the acquisition function (because minimize finds the minimum)
    def negative_acq_wrapper(x):
        val = acq_func(
            kernel, x, X_sample, y_sample, Ky_opt_inv, length_scale, **acq_params
        )
        return -val

    for i in range(n_restarts):
        x0 = np.random.uniform(lower_bounds, upper_bounds, dims)
        res = minimize(
            fun=negative_acq_wrapper, x0=x0, bounds=bounds, method="L-BFGS-B"
        )
        # res.fun is negative value, so -res.fun is the positive acquisition score
        if res.success and -res.fun > best_acq_value:
            best_acq_value = -res.fun
            best_x = res.x

    if best_x is None:
        print("  > WARNING: Acquisition optimization failed. Using a random point.")
        best_x = np.random.uniform(lower_bounds, upper_bounds, dims)

    return best_x, best_acq_value


# -------------------- Visualization --------------------


def plot_covariance_matrix(K):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=80)
    im = ax.imshow(K, cmap="viridis", vmin=0, vmax=1, origin="lower")
    ax.set_title("Covariance Matrix (Ky)")
    fig.colorbar(im, ax=ax)
    plt.show()


def plot_gp_results(
    X_grid,
    true_func,
    X_init,
    y_init,
    best_x,
    best_y,
    mu,
    sigma,
    acq_values,
    strategy_name,
):
    """Visualizes the GP regression results and acquisition function."""
    fig = plt.figure(figsize=(8, 7), dpi=80)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

    # Top plot: GP Prediction
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(X_grid, true_func(X_grid), "k-", label="True Function", linewidth=2)

    # Plot Initial Samples (White dots)
    ax1.scatter(
        X_init,
        y_init,
        c="w",
        s=100,
        alpha=1,
        edgecolors="k",
        lw=2,
        label="Initial Samples",
        zorder=3,
    )

    # Plot Best Found Point (Red Star)
    ax1.scatter(
        best_x,
        best_y,
        c="red",
        s=100,
        marker="o",
        alpha=1,
        edgecolors="k",
        lw=1.5,
        label="Best Found",
        zorder=5,
    )

    # Plot GP Prediction
    ax1.plot(X_grid, mu, "b--", label="Prediction", linewidth=2)
    ax1.fill_between(
        X_grid.ravel(),
        (mu - 1.96 * sigma),
        (mu + 1.96 * sigma),
        color="blue",
        alpha=0.2,
        label="95% CI",
    )

    ax1.set_ylabel("Value")
    # ax1.legend(loc="lower left")
    ax1.grid(False)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-3.2, 3.2)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Bottom plot: Acquisition Function
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(X_grid, acq_values, "k-", linewidth=2)

    # Highlight max Acquisition point
    next_sample_index = np.argmax(acq_values)
    next_sample_x = X_grid.ravel()[next_sample_index]
    ax2.axvline(next_sample_x, color="red", linestyle="--", alpha=0.5)

    ax2.set_xlabel("Variable")
    ax2.set_ylabel(f"Acquisition ({strategy_name})")
    ax2.grid(False)
    ax2.set_xlim(0, 1)

    ax2.set_ylim(-2.4, 2.4)

    plt.show()


def plot_gp_results_enhanced(
    X_grid,
    true_func,
    X_init,
    y_init,
    best_x,
    best_y,
    mu_post,
    sigma_post,
    acq_values,
    strategy_name,
):

    fig = plt.figure(figsize=(9, 7), dpi=80)

    gs = gridspec.GridSpec(
        2,
        2,
        height_ratios=[3, 1],
        width_ratios=[4, 1],
    )
    gs.update(hspace=0.0, wspace=0.0)  # ← スキマを完全に 0

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax3, sharey=ax2)

    # ax1: GP
    ax1.plot(X_grid, true_func(X_grid), "k-", linewidth=2)
    ax1.scatter(X_init, y_init, c="w", s=100, edgecolors="k", lw=2, zorder=3)
    ax1.scatter(
        best_x, best_y, c="red", s=100, marker="o", edgecolors="k", lw=1.5, zorder=5
    )
    ax1.plot(X_grid, mu_post, "b--", linewidth=2)
    ax1.fill_between(
        X_grid.ravel(),
        (mu_post - 1.96 * sigma_post),
        (mu_post + 1.96 * sigma_post),
        color="blue",
        alpha=0.2,
    )
    ax1.set_ylim(-3.2, 3.2)
    ax1.set_xlim(0, 1)
    ax1.set_ylabel("Value")

    # ax2: acquisition
    ax2.plot(X_grid, acq_values, "k-", linewidth=2)
    next_sample_index = np.argmax(acq_values)
    next_sample_x = X_grid.ravel()[next_sample_index]
    ax2.axvline(next_sample_x, color="red", linestyle="--", alpha=0.5)
    ax2.set_ylim(-2.4, 2.4)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Variable")
    ax2.set_ylabel(f"Acquisition ({strategy_name})")

    # 例: 何か pdf を描くならここに

    # ax4: 完全に非表示にしたいなら
    ax4.axis("off")
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)

    return fig, ax1, ax2, ax3, ax4
