import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Literal, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


Array = np.ndarray


def f_good(x: Array) -> Array:
    x1, x2 = x[:, 0], x[:, 1]
    return 1.5 * np.sin(np.pi * x1) + 0.25 * x1**2 + 0.8 * np.cos(np.pi * x2)


def f_bad_int(x: Array) -> Array:
    x1, x2 = x[:, 0], x[:, 1]
    return np.sin(np.pi * x1 * x2) + 0.25 * x1


def f_bad_corr(x: Array) -> Array:
    x1, x2 = x[:, 0], x[:, 1]
    return np.exp(-5.0 * (x1 - x2) ** 2) + 0.2 * np.sin(2 * np.pi * (x1 + x2))


@dataclass
class ALE1DResult:
    grid: Array
    effect: Array
    counts: Array


@dataclass
class ALE2DResult:
    grid1: Array
    grid2: Array
    effect: Array
    counts: Array
    mask: Array


def _quantile_edges(x: Array, n_bins: int) -> Array:
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, q)
    # ensure strictly increasing by merging duplicates
    edges = np.unique(edges)
    if edges.size < 3:
        lo, hi = np.min(x), np.max(x)
        if np.isclose(lo, hi):
            hi = lo + 1e-6
        edges = np.linspace(lo, hi, min(max(n_bins, 2), 4) + 1)
    return edges


def _bin_index(x: Array, edges: Array) -> Array:
    idx = np.searchsorted(edges, x, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def ale_1d(
    predictor: Callable[[Array], Array],
    X: Array,
    feat: int,
    n_bins: int = 20,
) -> ALE1DResult:
    xj = X[:, feat]
    edges = _quantile_edges(xj, n_bins)
    k = len(edges) - 1
    idx = _bin_index(xj, edges)

    diffs = np.zeros(k)
    counts = np.zeros(k, dtype=int)

    for b in range(k):
        mask = idx == b
        counts[b] = mask.sum()
        if counts[b] == 0:
            continue
        X_lo = X[mask].copy()
        X_hi = X[mask].copy()
        X_lo[:, feat] = edges[b]
        X_hi[:, feat] = edges[b + 1]
        deltas = predictor(X_hi) - predictor(X_lo)
        diffs[b] = np.mean(deltas)

    cum = np.concatenate([[0.0], np.cumsum(diffs)])
    mids = 0.5 * (edges[:-1] + edges[1:])
    eff_at_mids = 0.5 * (cum[:-1] + cum[1:])

    sample_effect = eff_at_mids[idx]
    centered = eff_at_mids - np.mean(sample_effect)

    return ALE1DResult(grid=mids, effect=centered, counts=counts)


def ale_2d_interaction(
    predictor: Callable[[Array], Array],
    X: Array,
    bins1: int = 20,
    bins2: int = 20,
    empty_policy: Literal["mask", "zero"] = "mask",
) -> ALE2DResult:
    e1 = _quantile_edges(X[:, 0], bins1)
    e2 = _quantile_edges(X[:, 1], bins2)
    k1, k2 = len(e1) - 1, len(e2) - 1

    i1 = _bin_index(X[:, 0], e1)
    i2 = _bin_index(X[:, 1], e2)

    local = np.zeros((k1, k2), dtype=float)
    counts = np.zeros((k1, k2), dtype=int)

    for a in range(k1):
        for b in range(k2):
            mask = (i1 == a) & (i2 == b)
            n = mask.sum()
            counts[a, b] = n
            if n == 0:
                continue
            X00 = X[mask].copy()
            X10 = X[mask].copy()
            X01 = X[mask].copy()
            X11 = X[mask].copy()

            x1_lo, x1_hi = e1[a], e1[a + 1]
            x2_lo, x2_hi = e2[b], e2[b + 1]

            X00[:, 0], X00[:, 1] = x1_lo, x2_lo
            X10[:, 0], X10[:, 1] = x1_hi, x2_lo
            X01[:, 0], X01[:, 1] = x1_lo, x2_hi
            X11[:, 0], X11[:, 1] = x1_hi, x2_hi

            delta = predictor(X11) - predictor(X10) - predictor(X01) + predictor(X00)
            local[a, b] = np.mean(delta)

    raw = np.cumsum(np.cumsum(local, axis=0), axis=1)

    # weighted additive projection removal: interaction orthogonal to const, row, col spaces
    w = counts.astype(float)
    valid = w > 0
    if empty_policy == "zero":
        valid = np.ones_like(valid, dtype=bool)
        w = np.where(counts > 0, w, 1.0)

    h = raw.copy()
    h[~valid] = 0.0

    # alternating weighted projections (two-way ANOVA style)
    c = np.sum(w * h) / np.sum(w)
    h = h - c
    for _ in range(30):
        row_adj = np.divide(np.sum(w * h, axis=1), np.sum(w, axis=1), out=np.zeros(k1), where=np.sum(w, axis=1) > 0)
        h = h - row_adj[:, None]
        col_adj = np.divide(np.sum(w * h, axis=0), np.sum(w, axis=0), out=np.zeros(k2), where=np.sum(w, axis=0) > 0)
        h = h - col_adj[None, :]
        c2 = np.sum(w * h) / np.sum(w)
        h = h - c2

    if empty_policy == "mask":
        h = np.where(valid, h, np.nan)

    g1 = 0.5 * (e1[:-1] + e1[1:])
    g2 = 0.5 * (e2[:-1] + e2[1:])
    return ALE2DResult(grid1=g1, grid2=g2, effect=h, counts=counts, mask=valid)


def pdp_1d(predictor: Callable[[Array], Array], X: Array, feat: int, grid: Array) -> Array:
    out = np.zeros_like(grid, dtype=float)
    for i, v in enumerate(grid):
        Xc = X.copy()
        Xc[:, feat] = v
        out[i] = np.mean(predictor(Xc))
    return out


def ice_curves(predictor: Callable[[Array], Array], X: Array, feat: int, grid: Array, n_lines: int = 50) -> Array:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(n_lines, len(X)), replace=False)
    lines = np.zeros((len(idx), len(grid)))
    for i, r in enumerate(idx):
        Xc = np.repeat(X[r:r+1], len(grid), axis=0)
        Xc[:, feat] = grid
        lines[i] = predictor(Xc)
    return lines


def fit_gp(X: Array, y: Array) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF([1.0, 1.0], (1e-2, 1e2)) + WhiteKernel(1e-3, (1e-6, 1e-1))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0, n_restarts_optimizer=3)
    gp.fit(X, y)
    return gp


def make_dataset(case: int, n: int = 1200, seed: int = 42):
    rng = np.random.default_rng(seed)
    if case in (1, 2):
        X = rng.uniform(-2, 2, size=(n, 2))
    else:
        x1 = rng.uniform(-2, 2, size=n)
        eta = rng.normal(0.0, 0.1, size=n)
        x2 = x1 + eta
        X = np.column_stack([x1, x2])

    f_map = {1: f_good, 2: f_bad_int, 3: f_bad_corr}
    y = f_map[case](X)
    return X, y, f_map[case]


def reconstruct(yhat: Array, a1: ALE1DResult, a2: ALE1DResult, a12: ALE2DResult, X: Array) -> Array:
    i1 = np.searchsorted(a1.grid, X[:, 0], side="left")
    i1 = np.clip(i1, 0, len(a1.grid)-1)
    i2 = np.searchsorted(a2.grid, X[:, 1], side="left")
    i2 = np.clip(i2, 0, len(a2.grid)-1)
    base = np.mean(yhat)
    int_term = a12.effect[i1, i2]
    int_term = np.where(np.isnan(int_term), 0.0, int_term)
    return base + a1.effect[i1] + a2.effect[i2] + int_term


def run_case(case: int, n_bins=20):
    X, y, f_true = make_dataset(case)
    gp = fit_gp(X, y)

    predictors = {
        "true": lambda z: f_true(z),
        "gp_mean": lambda z: gp.predict(z, return_std=False),
    }

    fig, axs = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle(f"Case {case}: ALE / PDP / ICE")

    axs[0, 0].scatter(X[:, 0], X[:, 1], s=8, alpha=0.4)
    axs[0, 0].set_title("Input scatter")

    for r, (name, pred) in enumerate(predictors.items(), start=1):
        a1 = ale_1d(pred, X, feat=0, n_bins=n_bins)
        a2 = ale_1d(pred, X, feat=1, n_bins=n_bins)
        a12 = ale_2d_interaction(pred, X, bins1=n_bins, bins2=n_bins, empty_policy="mask")

        g1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 80)
        ice1 = ice_curves(pred, X, feat=0, grid=g1)
        pdp1 = pdp_1d(pred, X, feat=0, grid=g1)

        ax = axs[r-1, 1]
        for line in ice1:
            ax.plot(g1, line, color="gray", alpha=0.2)
        ax.plot(g1, pdp1, color="tab:red", lw=2, label="PDP")
        ax.set_title(f"{name}: ICE/PDP x1")

        axs[r-1, 2].plot(a1.grid, a1.effect, label="A1")
        axs[r-1, 2].plot(a2.grid, a2.effect, label="A2")
        axs[r-1, 2].set_title(f"{name}: 1D ALE")
        axs[r-1, 2].legend()

        im = axs[r-1, 3].imshow(
            a12.effect.T,
            origin="lower",
            aspect="auto",
            extent=[a12.grid1[0], a12.grid1[-1], a12.grid2[0], a12.grid2[-1]],
            cmap="coolwarm",
        )
        axs[r-1, 3].set_title(f"{name}: A12")
        plt.colorbar(im, ax=axs[r-1, 3], fraction=0.046, pad=0.04)

        yhat = pred(X)
        yrec = reconstruct(yhat, a1, a2, a12, X)
        rmse = np.sqrt(np.mean((yhat - yrec) ** 2))
        print(f"Case {case} {name} reconstruction RMSE: {rmse:.4f}")

    for ax in axs.ravel():
        ax.grid(alpha=0.2)
    plt.tight_layout()
    out = f"ale_case_{case}.png"
    plt.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    for c in [1, 2, 3]:
        run_case(c, n_bins=20)
