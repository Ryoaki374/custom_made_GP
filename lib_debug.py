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


# -------------------- Stochastic EI Bayesian Optimization --------------------


class KernelGPRegressor:
    """Small adapter exposing predict(X, return_std=True) for the existing GP code."""

    def __init__(self, kernel=rbf_kernel, gamma=4.0, noise_var=1e-6):
        self.kernel = kernel
        self.gamma = gamma
        self.noise_var = noise_var
        self.X_train = None
        self.y_train = None
        self.Ky_inv = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float).reshape(-1, 1)
        self.Ky_inv = fit_gp(self.kernel, self.X_train, self.gamma, self.noise_var)
        return self

    def predict(self, X, return_std=True):
        if self.X_train is None or self.y_train is None or self.Ky_inv is None:
            raise RuntimeError("KernelGPRegressor must be fit before predict is called.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        K_star = self.kernel(self.X_train, X, self.gamma)
        mu = (K_star.T @ self.Ky_inv @ self.y_train).ravel()
        K_xx_diag = np.ones(X.shape[0])
        cov_reduction = np.sum(K_star * (self.Ky_inv @ K_star), axis=0)
        var = np.maximum(0.0, K_xx_diag - cov_reduction)
        std = np.sqrt(var)
        if return_std:
            return mu, std
        return mu


def _as_bounds_array(bounds):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (d, 2).")
    return bounds


def sample_input_perturbations(x, Sigma, n_samples, bounds, rng):
    """Sample clipped Gaussian input perturbations around a design point."""
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(x, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    bounds = _as_bounds_array(bounds)
    if Sigma.shape != (x.size, x.size):
        raise ValueError("Sigma must have shape (d, d), where x has shape (d,).")
    if bounds.shape[0] != x.size:
        raise ValueError("bounds must have one [lower, upper] row per x dimension.")

    deltas = rng.multivariate_normal(np.zeros(x.size), Sigma, size=n_samples)
    X_perturbed = x.reshape(1, -1) + deltas
    return np.clip(X_perturbed, bounds[:, 0], bounds[:, 1])


def stochastic_expected_improvement(
    x,
    gp,
    y_best,
    Sigma,
    bounds,
    n_perturb=64,
    xi=0.0,
    rng=None,
):
    """Monte Carlo stochastic EI for minimization under input perturbations."""
    rng = np.random.default_rng() if rng is None else rng
    X_perturbed = sample_input_perturbations(x, Sigma, n_perturb, bounds, rng)
    mu, sigma = gp.predict(X_perturbed, return_std=True)
    improvement = float(y_best) - mu - xi

    eps = 1e-12
    ei = np.empty_like(improvement, dtype=float)
    near_zero = sigma <= eps
    ei[near_zero] = np.maximum(improvement[near_zero], 0.0)
    stable = ~near_zero
    if np.any(stable):
        Z = improvement[stable] / sigma[stable]
        ei[stable] = improvement[stable] * norm.cdf(Z) + sigma[stable] * norm.pdf(Z)
    return float(np.mean(ei))


def stochastic_expected_improvement_existing_gp(
    kernel,
    x_new,
    X_sample,
    y_sample,
    Ky_opt_inv,
    length_scale,
    Sigma,
    bounds,
    n_perturb=64,
    xi=0.0,
    rng=None,
):
    """sEI acquisition adapter for optimize_acquisition and the existing GP arrays."""
    gp = KernelGPRegressor(kernel=kernel, gamma=length_scale)
    gp.X_train = np.asarray(X_sample, dtype=float)
    gp.y_train = np.asarray(y_sample, dtype=float).reshape(-1, 1)
    gp.Ky_inv = Ky_opt_inv
    return stochastic_expected_improvement(
        x_new,
        gp,
        np.min(y_sample),
        Sigma,
        bounds,
        n_perturb=n_perturb,
        xi=xi,
        rng=rng,
    )


ACQUISITION_FUNCTIONS = {
    "EI": expected_improvement,
    "LCB": lower_confidence_bound,
    "sEI": stochastic_expected_improvement_existing_gp,
}


def get_acquisition_function(name):
    """Return an acquisition function by name, including the stochastic EI option."""
    try:
        return ACQUISITION_FUNCTIONS[name]
    except KeyError as exc:
        available = ", ".join(sorted(ACQUISITION_FUNCTIONS))
        raise ValueError(f"Unknown acquisition '{name}'. Available: {available}") from exc


def branin(x):
    """Standard Branin function in 2D for minimization."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x1 = x[:, 0]
    x2 = x[:, 1]
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1.0 - t) * np.cos(x1) + s


def hartmann6(x):
    """Hartmann-6 function with the conventional negative-valued minimization form."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ],
        dtype=float,
    )
    P = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ],
        dtype=float,
    )
    inner = np.sum(A[None, :, :] * (x[:, None, :] - P[None, :, :]) ** 2, axis=2)
    return -np.sum(alpha[None, :] * np.exp(-inner), axis=1)


def default_sigma_for_problem(function_name, bounds):
    """Default diagonal perturbation covariance for the supported synthetic functions."""
    bounds = _as_bounds_array(bounds)
    name = function_name.lower()
    d = bounds.shape[0]
    if name == "branin":
        widths = bounds[:, 1] - bounds[:, 0]
        return np.diag((0.05 * widths) ** 2)
    if name in {"hartmann6", "hartmann-6"}:
        return (0.03**2) * np.eye(d)
    if name == "ackley":
        return (0.5**2) * np.eye(d)
    raise ValueError(f"Unknown function_name: {function_name}")


def get_synthetic_problem(function_name="ackley", d=None, Sigma=None):
    """Return (f_true, bounds, Sigma, canonical_name) for supported minimization problems."""
    name = function_name.lower()
    if name == "branin":
        bounds = np.array([[-5.0, 10.0], [0.0, 15.0]], dtype=float)
        f_true = branin
        canonical_name = "Branin"
    elif name in {"hartmann6", "hartmann-6"}:
        bounds = np.array([[0.0, 1.0]] * 6, dtype=float)
        f_true = hartmann6
        canonical_name = "Hartmann-6"
    elif name == "ackley":
        d = 6 if d is None else int(d)
        if d not in {2, 6}:
            raise ValueError("Ackley dimension must be 2 or 6 for this PoC.")
        bounds = np.array([[-5.0, 5.0]] * d, dtype=float)
        f_true = ackley_nd
        canonical_name = "Ackley"
    else:
        raise ValueError("function_name must be one of: branin, hartmann6, ackley.")
    Sigma = (
        default_sigma_for_problem(name, bounds)
        if Sigma is None
        else np.asarray(Sigma, dtype=float)
    )
    return f_true, bounds, Sigma, canonical_name


def optimize_acquisition_by_random_search(acq_func, bounds, rng, n_candidates=None):
    """Maximize an acquisition function by uniform random search."""
    bounds = _as_bounds_array(bounds)
    d = bounds.shape[0]
    if n_candidates is None:
        n_candidates = 2000 if d <= 2 else 5000
    X_cand = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_candidates, d))
    values = np.array([acq_func(x) for x in X_cand])
    best_idx = int(np.argmax(values))
    return X_cand[best_idx], float(values[best_idx])


def recommend_by_posterior_mean(gp, bounds, rng, n_candidates=None):
    """Recommend the point with the lowest GP posterior mean on random candidates."""
    bounds = _as_bounds_array(bounds)
    d = bounds.shape[0]
    if n_candidates is None:
        n_candidates = 2000 if d <= 2 else 5000
    X_cand = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_candidates, d))
    mu = gp.predict(X_cand, return_std=False)
    best_idx = int(np.argmin(mu))
    return X_cand[best_idx], float(mu[best_idx])


def validate_true_function_mc(candidates, f_true, Sigma, bounds, n_mc=2048, rng=None):
    """Validate candidate robustness with true-function Monte Carlo perturbations."""
    rng = np.random.default_rng() if rng is None else rng
    results = {}
    for name, x in candidates.items():
        X_mc = sample_input_perturbations(x, Sigma, n_mc, bounds, rng)
        y_mc = np.asarray(f_true(X_mc), dtype=float).reshape(-1)
        nominal_value = float(
            np.asarray(f_true(np.asarray(x).reshape(1, -1))).reshape(-1)[0]
        )
        results[name] = {
            "mean": float(np.mean(y_mc)),
            "std": float(np.std(y_mc)),
            "q05": float(np.quantile(y_mc, 0.05)),
            "q50": float(np.quantile(y_mc, 0.50)),
            "q95": float(np.quantile(y_mc, 0.95)),
            "nominal_value": nominal_value,
        }
    return results


def run_sei_bo(
    function_name="ackley",
    d=None,
    Sigma=None,
    n_initial=None,
    n_iter=30,
    n_perturb_sEI=64,
    n_mc_validation=2048,
    random_seed=0,
    gamma=1.0,
    noise_std=0.0,
    xi=0.0,
    n_candidates=None,
):
    """Run a minimal sEI Bayesian Optimization loop on a supported synthetic function."""
    rng = np.random.default_rng(random_seed)
    f_true, bounds, Sigma, canonical_name = get_synthetic_problem(
        function_name, d=d, Sigma=Sigma
    )
    dim = bounds.shape[0]
    n_initial = 5 * dim if n_initial is None else int(n_initial)
    noise_var = noise_std**2

    X_train = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_initial, dim))
    y_train = f_true(X_train).reshape(-1, 1)
    if noise_std > 0:
        y_train = y_train + rng.normal(0.0, noise_std, size=y_train.shape)

    for _ in range(n_iter):
        gp = KernelGPRegressor(gamma=gamma, noise_var=noise_var).fit(X_train, y_train)
        y_best = float(np.min(y_train))
        perturb_seed = int(rng.integers(0, np.iinfo(np.uint32).max))

        def acq(x):
            # Common random numbers within each BO iteration for reproducible candidate comparisons.
            local_rng = np.random.default_rng(perturb_seed)
            return stochastic_expected_improvement(
                x,
                gp,
                y_best,
                Sigma,
                bounds,
                n_perturb=n_perturb_sEI,
                xi=xi,
                rng=local_rng,
            )

        x_next, _ = optimize_acquisition_by_random_search(acq, bounds, rng, n_candidates)
        y_next = f_true(x_next.reshape(1, -1)).reshape(1, 1)
        if noise_std > 0:
            y_next = y_next + rng.normal(0.0, noise_std, size=(1, 1))
        X_train = np.vstack([X_train, x_next])
        y_train = np.vstack([y_train, y_next])

    gp = KernelGPRegressor(gamma=gamma, noise_var=noise_var).fit(X_train, y_train)
    best_idx = int(np.argmin(y_train))
    best_observed_x = X_train[best_idx]
    best_observed_y = float(y_train[best_idx, 0])
    final_recommended_x, _ = recommend_by_posterior_mean(gp, bounds, rng, n_candidates)
    candidates = {
        "best_observed": best_observed_x,
        "best_sEI_recommended": final_recommended_x,
    }
    validation = validate_true_function_mc(
        candidates, f_true, Sigma, bounds, n_mc=n_mc_validation, rng=rng
    )
    robust_best = min(validation, key=lambda name: validation[name]["mean"])

    return {
        "function_name": canonical_name,
        "dimension": dim,
        "n_initial": n_initial,
        "n_iter": n_iter,
        "n_perturb_sEI": n_perturb_sEI,
        "Sigma": Sigma,
        "bounds": bounds,
        "X_train": X_train,
        "y_train": y_train,
        "best_observed_y": best_observed_y,
        "best_observed_x": best_observed_x,
        "final_recommended_x": final_recommended_x,
        "validation": validation,
        "robust_best": robust_best,
    }


def print_sei_bo_summary(result):
    """Print BO and validation summaries for run_sei_bo output."""
    print("BO summary")
    print(f"function name: {result['function_name']}")
    print(f"dimension d: {result['dimension']}")
    print(f"n_initial: {result['n_initial']}")
    print(f"n_iter: {result['n_iter']}")
    print(f"n_perturb_sEI: {result['n_perturb_sEI']}")
    print(f"Sigma:\n{result['Sigma']}")
    print(f"best observed y: {result['best_observed_y']:.6f}")
    print(f"best observed x: {result['best_observed_x']}")
    print(f"final recommended x: {result['final_recommended_x']}")
    print("\nvalidation summary")
    header = (
        f"{'candidate':<24} {'nominal_value':>14} {'perturbed_mean':>16} "
        f"{'perturbed_std':>15} {'q05':>12} {'q50':>12} {'q95':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, stats in result["validation"].items():
        print(
            f"{name:<24} {stats['nominal_value']:>14.6f} {stats['mean']:>16.6f} "
            f"{stats['std']:>15.6f} {stats['q05']:>12.6f} "
            f"{stats['q50']:>12.6f} {stats['q95']:>12.6f}"
        )
    print(f"\nrobust best by perturbed_mean: {result['robust_best']}")
