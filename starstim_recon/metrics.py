"""Scores: coarse8 (the 8x8 colour-layout correlation), pairwise identification, permutation nulls, paired bootstrap."""
import numpy as np
import torch
import torch.nn.functional as F


def coarse8(pred32, true32, local) -> tuple[np.ndarray, np.ndarray]:
    """Per-example coarse8 and the example x candidate matrix.

    Both 8x8 layouts are centred on the mean true layout of the candidate set, then correlated (cosine of the
    mean-removed rows). local[i] = index of example i's true image among the candidates (rows of true32).
    """
    def unit_rows(x):
        x = x - x.mean(1, keepdim=True)
        return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)

    p8 = F.adaptive_avg_pool2d(torch.as_tensor(pred32).float(), 8).flatten(1)
    t8 = F.adaptive_avg_pool2d(torch.as_tensor(true32).float(), 8).flatten(1)
    mu = t8.mean(0, keepdim=True)
    matrix = (unit_rows(p8 - mu) @ unit_rows(t8 - mu).T).numpy()
    return matrix[np.arange(len(local)), local], matrix


def permutation_p(matrix, local, groups, observed, n_perm, rng) -> float:
    """Image labels shuffled within each group (subject or session); one-sided p of the mean coarse8."""
    rows = np.arange(len(local)); idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]; ge = 0
    for _ in range(n_perm):
        permuted = local.copy()
        for g in idx:
            permuted[g] = rng.permutation(local[g])
        ge += matrix[rows, permuted].mean() >= observed
    return (ge + 1) / (n_perm + 1)


def pairwise_identification(matrix, local) -> float:
    """Fraction of wrong candidates the true image beats (chance 0.5)."""
    true = matrix[np.arange(len(local)), local][:, None]
    return float(((true > matrix).sum(1) / (matrix.shape[1] - 1)).mean())


def score(pred32, true32, local, groups, n_perm=2000, seed=1) -> tuple[dict, np.ndarray]:
    values, matrix = coarse8(pred32, true32, local)
    observed = float(values.mean())
    p = permutation_p(matrix, local, groups, observed, n_perm, np.random.default_rng(seed))
    return {"n_examples": int(len(local)), "coarse8": observed, "permutation_p": p,
            "pairwise_id": pairwise_identification(matrix, local)}, values


def pooled_score(cells, n_perm=5000, seed=1) -> dict:
    """Pool (values, matrix, local) cells of several sessions: mean coarse8, labels shuffled within each session."""
    obs = float(np.concatenate([c[0] for c in cells]).mean())
    rng = np.random.default_rng(seed); n = sum(len(c[0]) for c in cells); ge = 0
    for _ in range(n_perm):
        tot = sum(m[np.arange(len(loc)), rng.permutation(loc)].sum() for _, m, loc in cells)
        ge += tot / n >= obs
    pair = float(np.concatenate([((m[np.arange(len(l)), l][:, None] > m).sum(1) / (len(l) - 1)) for _, m, l in cells]).mean())
    return {"n_images": n, "coarse8": obs, "permutation_p": (ge + 1) / (n_perm + 1), "pairwise_id": pair}


def paired_bootstrap(a, b, n_boot=10000, seed=2) -> dict:
    d = np.asarray(a) - np.asarray(b)
    boots = d[np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)]}


def corr_cols(P, T) -> np.ndarray:
    P = P - P.mean(0); T = T - T.mean(0)
    return (P * T).sum(0) / (np.linalg.norm(P, axis=0) * np.linalg.norm(T, axis=0) + 1e-12)
