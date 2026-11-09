"""Linear readouts: per-person ZCA whitening and ridge regression (kernel form for few examples, primal for THINGS)."""
import numpy as np
import torch

from .metrics import coarse8

ALPHAS = (1e2, 1e3, 1e4, 1e5, 1e6)          # lab ridge penalties (on ZCA-whitened, flattened 25-31 x 250 EEG)


def zca_by_subject(eeg: torch.Tensor, subjects: np.ndarray) -> torch.Tensor:
    """Per-subject ZCA from shrinkage covariances (Ledoit-Wolf per epoch, averaged, eigenvalue-floored). Label-free."""
    from sklearn.covariance import LedoitWolf
    out = eeg.clone()
    for s in np.unique(subjects):
        rows = np.flatnonzero(subjects == s)
        cov = np.mean([LedoitWolf().fit(eeg[i].numpy().T).covariance_ for i in rows], axis=0)
        eigval, eigvec = np.linalg.eigh(cov)
        eigval = np.clip(eigval, eigval.max() * 1e-6, None)               # the average-reference null direction
        whiten = eigvec @ np.diag(eigval ** -0.5) @ eigvec.T
        out[rows] = torch.einsum("cd,ndt->nct", torch.from_numpy(whiten).float(), eeg[rows])
    return out


def dual_ridge_predict(Xtr, Ytr, Xte, alpha):
    """Kernel-form ridge with intercept: exact and cheap for n << d."""
    mu_x, mu_y = Xtr.mean(0, keepdims=True), Ytr.mean(0, keepdims=True)
    Xc = Xtr - mu_x
    coef = np.linalg.solve(Xc @ Xc.T + alpha * np.eye(len(Xc)), Ytr - mu_y)
    return mu_y + (Xte - mu_x) @ Xc.T @ coef


def aggregate(eeg: torch.Tensor, targets, subjects, level):
    """-> (X flattened float64, image per example, subject per example) at 'single_trial', 'subject_image' or 'image'."""
    X = eeg.flatten(1).numpy().astype(np.float64)
    if level == "single_trial":
        return X, targets, subjects
    if level == "subject_image":
        keys = sorted({(s, int(t)) for s, t in zip(subjects, targets)})
        rows = [np.flatnonzero((subjects == s) & (targets == t)) for s, t in keys]
        return np.stack([X[r].mean(0) for r in rows]), np.array([t for _, t in keys]), np.array([s for s, _ in keys])
    if level == "image":
        uniq = np.array(sorted(set(targets.tolist())))
        return np.stack([X[targets == t].mean(0) for t in uniq]), uniq, np.array(["pooled"] * len(uniq))
    raise ValueError(level)


def nested_cv(X, img, true32, uniq, outer_folds=8, inner_folds=4, seed=0, X_eval=None):
    """Out-of-fold ridge predictions, folds by image; alpha tuned on coarse8 in inner image folds.
    X_eval (rows aligned with X): predict these inputs for the held-out rows instead of X (e.g. fewer-repeat averages)."""
    rng = np.random.default_rng(seed)
    images = np.array(sorted(set(img.tolist())))
    perm = rng.permutation(len(images))
    fold_of_image = {int(images[perm[i]]): i % outer_folds for i in range(len(images))}
    fold = np.array([fold_of_image[int(t)] for t in img])
    position = {int(t): i for i, t in enumerate(uniq)}
    Y = np.stack([true32[position[int(t)]].flatten().numpy() for t in img])
    preds = np.zeros((len(X), Y.shape[1]))
    for k in range(outer_folds):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        tr_images = np.array(sorted(set(img[tr].tolist())))
        inner_of = {int(tr_images[i]): i % inner_folds for i in range(len(tr_images))}
        ifold = np.array([inner_of[int(t)] for t in img[tr]])
        scores = []
        for alpha in ALPHAS:
            vals = []
            for j in range(inner_folds):
                itr, ite = tr[ifold != j], tr[ifold == j]
                pred = dual_ridge_predict(X[itr], Y[itr], X[ite], alpha)
                loc = np.array([position[int(t)] for t in img[ite]])
                vals.append(coarse8(torch.from_numpy(pred.reshape(-1, 3, 32, 32)).float(), true32, loc)[0].mean())
            scores.append(float(np.mean(vals)))
        preds[te] = dual_ridge_predict(X[tr], Y[tr], (X if X_eval is None else X_eval)[te], ALPHAS[int(np.argmax(scores))])
    return preds


PRIOR_ALPHAS = ALPHAS + (1e7, 1e8, 1e9)      # large alpha -> the prior readout alone


def nested_prior_cv(X, img, true32, uniq, f_T, outer_folds=8, inner_folds=4, seed=0):
    """nested_cv with shrinkage toward a prior readout f_T: argmin ||Y - Xb||^2 + alpha ||b - b_T||^2, i.e. f_T plus a ridge
    on the residuals Y - f_T(X). f_T takes and returns torch tensors. -> (out-of-fold predictions, alpha per outer fold)."""
    images = np.array(sorted(set(img.tolist()))); perm = np.random.default_rng(seed).permutation(len(images))
    fold_of = {int(images[perm[i]]): i % outer_folds for i in range(len(images))}; fold = np.array([fold_of[int(t)] for t in img])
    position = {int(t): i for i, t in enumerate(uniq)}
    Y = np.stack([true32[position[int(t)]].flatten().numpy() for t in img]).astype(np.float64)
    F0 = f_T(torch.from_numpy(X).float()).numpy().astype(np.float64)
    preds, chosen = np.zeros_like(Y), []
    for k in range(outer_folds):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        tr_images = np.array(sorted(set(img[tr].tolist())))
        inner_of = {int(tr_images[i]): i % inner_folds for i in range(len(tr_images))}
        ifold = np.array([inner_of[int(t)] for t in img[tr]]); scores = []
        for alpha in PRIOR_ALPHAS:
            vals = []
            for j in range(inner_folds):
                itr, ite = tr[ifold != j], tr[ifold == j]
                pred = F0[ite] + dual_ridge_predict(X[itr], Y[itr] - F0[itr], X[ite], alpha)
                loc = np.array([position[int(t)] for t in img[ite]])
                vals.append(coarse8(torch.from_numpy(pred.reshape(-1, 3, 32, 32)).float(), true32, loc)[0].mean())
            scores.append(float(np.mean(vals)))
        alpha = PRIOR_ALPHAS[int(np.argmax(scores))]; chosen.append(alpha)
        preds[te] = F0[te] + dual_ridge_predict(X[tr], Y[tr] - F0[tr], X[te], alpha)
    return preds, chosen


def fit_ridge(X, Y, img, true32, local, seed=0):
    """alpha by 4-fold image CV (coarse8), then the fit on all rows -> (predict(X), alpha, inner scores)."""
    images = np.unique(img); perm = np.random.default_rng(seed).permutation(len(images))
    fold_of = {int(images[perm[i]]): i % 4 for i in range(len(images))}; fold = np.array([fold_of[int(t)] for t in img])
    scores = []
    for alpha in ALPHAS:
        vals = []
        for k in range(4):
            tr, te = fold != k, fold == k
            pred = dual_ridge_predict(X[tr], Y[tr], X[te], alpha)
            vals.append(coarse8(torch.from_numpy(pred.reshape(-1, 3, 32, 32)).float(), true32, local[te])[0].mean())
        scores.append(float(np.mean(vals)))
    alpha = ALPHAS[int(np.argmax(scores))]
    return (lambda Xn: dual_ridge_predict(X, Y, Xn, alpha)), alpha, scores


def fit_primal_ridge(X: torch.Tensor, Y: torch.Tensor, alpha: float, device="cuda"):
    """Primal ridge for many examples (THINGS). The returned predictor centres its input by the input's own mean."""
    mx, my = X.mean(0, keepdim=True), Y.mean(0, keepdim=True)
    Xc = (X - mx).to(device, torch.float64)
    A = Xc.T @ Xc; A.diagonal().add_(alpha * len(Xc))                         # penalty scales with n
    beta = torch.linalg.solve(A, Xc.T @ (Y - my).to(device, torch.float64))

    def predict(Xn: torch.Tensor) -> torch.Tensor:
        Xn = (Xn - Xn.mean(0, keepdim=True)).to(device, torch.float64)
        return (Xn @ beta + my.to(device, torch.float64)).float().cpu()
    return predict
