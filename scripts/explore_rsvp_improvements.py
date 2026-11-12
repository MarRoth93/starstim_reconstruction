#!/usr/bin/env python3
"""EXPLORATORY: what could improve the decoding of a rapid-stream block (E21)? Many variants on one session: read the
table as a screen for ideas worth a confirmatory test on new data, not as results.

    python scripts/explore_rsvp_improvements.py --session E21 [--offset-ms 0]

Written on 2026-11-10, before the clock correction: the screen that was reported then ran with time zero at the image
marker (--offset-ms 0). Without the option the loader's corrected timing is used.

Images with a target = image file present, or the image is a THINGS-EEG2 stimulus with a cached target (so the script also
runs on the THINGS-EEG2 images only while the image disk is offline). Everything is scored per image against that set
(coarse8 on the 8x8 colour layout, 8 outer image folds as in run_rsvp_reconstruction.py, alpha in 4 inner folds).

A  Shared stimuli. E21 images that are also THINGS-EEG2 training images give paired responses (E21 vs the THINGS group mean,
   40 trials). Label-free diagnostics: lag between the grand-average responses and between the image-specific patterns,
   whether an E21 pattern identifies the THINGS pattern of the same image, channel-by-channel correspondence.
B  THINGS decoder on E21: zero-shot as a function of a time shift of the E21 epochs, and after a spatial (31 x 31) alignment
   E21 -> THINGS space fitted on the paired responses of the training-fold images only (no pixel labels; lambda by inner
   folds on held-out EEG prediction), shrunk toward the identity.
C  Within-person ridge variants: time window, 50 Hz pooling, training on single trials instead of image means,
   low-dimensional targets (PCA of THINGS layouts, or the directions the THINGS decoder predicts best).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starstim_recon import data_things  # noqa: E402
from starstim_recon.data_sessions import load_rsvp  # noqa: E402
from starstim_recon.channels import STARSTIM_31  # noqa: E402
from starstim_recon.images import pixel32  # noqa: E402
from starstim_recon.metrics import coarse8, paired_bootstrap, score  # noqa: E402
from starstim_recon.paths import RESULTS  # noqa: E402
from starstim_recon.readout import ALPHAS, fit_primal_ridge, zca_by_subject  # noqa: E402

DEV = "cuda"
FS = 250
ZCA_SCALE = 1.1
THINGS_ALPHA = 1e-3
WIDE = tuple(10.0 ** k for k in range(0, 8))
LAGS = list(range(-15, 26))                                    # samples: -60 .. +100 ms (E21 later than THINGS = positive)


def pool8(x32):
    return F.adaptive_avg_pool2d(torch.as_tensor(x32).float(), 8).flatten(1).numpy().astype(np.float64)


def shift(E, lag):
    """E'(t) = E(t + lag): undoes a response that arrives `lag` samples later than the reference."""
    if lag == 0:
        return E
    if lag > 0:
        return torch.cat([E[..., lag:], E[..., -1:].expand(*E.shape[:-1], lag)], -1)
    return torch.cat([E[..., :1].expand(*E.shape[:-1], -lag), E[..., :lag]], -1)


def corr(a, b):
    a = a.flatten() - a.mean(); b = b.flatten() - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def make_folds(n, k, seed=0):
    perm = np.random.default_rng(seed).permutation(n); fold = np.empty(n, int); fold[perm] = np.arange(n) % k
    return fold


def ridge_paths(Xtr, Ytr, Xte, alphas):
    """Kernel ridge with intercept for several alphas from one eigendecomposition (torch, float64)."""
    mx, my = Xtr.mean(0, keepdim=True), Ytr.mean(0, keepdim=True)
    Xc = Xtr - mx; s, U = torch.linalg.eigh(Xc @ Xc.T); UtY = U.T @ (Ytr - my); Kte = (Xte - mx) @ Xc.T
    return [my + Kte @ (U @ (UtY / (s.clamp_min(0) + a)[:, None])) for a in alphas]


def cv_predict(Xrows, img_rows, Xeval, Y, score_fn, alphas, fold, n_inner=4):
    """Out-of-fold predictions for every image. Xrows / img_rows: training rows and their image; Xeval: one row per image."""
    preds = torch.zeros_like(Y); chosen = []
    for k in range(fold.max() + 1):
        tr_imgs, te_imgs = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        inner = np.arange(len(tr_imgs)) % n_inner; sc = np.zeros(len(alphas))
        for j in range(n_inner):
            itr, ite = tr_imgs[inner != j], tr_imgs[inner == j]
            rows = torch.as_tensor(np.isin(img_rows, itr))
            for a, p in enumerate(ridge_paths(Xrows[rows], Y[img_rows[rows.numpy()]], Xeval[ite], alphas)):
                sc[a] += score_fn(p, ite)
        best = int(np.argmax(sc)); chosen.append(alphas[best])
        rows = torch.as_tensor(np.isin(img_rows, tr_imgs))
        preds[te_imgs] = ridge_paths(Xrows[rows], Y[img_rows[rows.numpy()]], Xeval[te_imgs], [alphas[best]])[0]
    return preds, chosen


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--session", default="E21"); ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--offset-ms", type=float, nargs="+", default=None)
    args = ap.parse_args(); out = {}

    # ---------------------------------------------------------------- data
    R = load_rsvp(args.session, require_images=False, offset_ms=args.offset_ms)
    trig_all = R["tr"].trigger.values.astype(int); uniq_all = np.array(sorted(set(trig_all)))
    tr_paths, te_paths = data_things.image_paths("train"), data_things.image_paths("test")
    tr_idx = {p.name: i for i, p in enumerate(tr_paths)}; te_idx = {p.name: i for i, p in enumerate(te_paths)}
    T32 = {"train": data_things.targets("train"), "test": data_things.targets("test")}
    Y8, keep_img, things_train_i = [], [], []
    for u in uniq_all:
        f = R["files"][int(u)]
        if f.exists():
            y = pool8(pixel32(f)[None])[0]
        elif f.name in tr_idx:
            y = pool8(T32["train"][tr_idx[f.name]][None])[0]
        elif f.name in te_idx:
            y = pool8(T32["test"][te_idx[f.name]][None])[0]
        else:
            continue
        Y8.append(y); keep_img.append(u); things_train_i.append(tr_idx.get(f.name, -1))
    uniq = np.array(keep_img); things_train_i = np.array(things_train_i); nS = len(uniq)
    Y8 = torch.as_tensor(np.stack(Y8), device=DEV); TRUE8 = Y8.view(-1, 3, 8, 8).float().cpu()
    sel = np.isin(trig_all, uniq); trig = trig_all[sel]; img_rows = np.searchsorted(uniq, trig)
    x = zca_by_subject(R["ep"], np.array(["session"] * len(R["ep"])))[sel]            # (n, 31, 250), unscaled ZCA
    shared = np.flatnonzero(things_train_i >= 0)
    print(f"{args.session}: {len(trig)} trials, {nS} of {len(uniq_all)} images have a target; {len(shared)} are THINGS-EEG2 "
          f"training images (paired responses)")
    out["n_images"], out["n_shared"] = nS, int(len(shared))
    fold = make_folds(nS, 8); groups = np.zeros(nS); local = np.arange(nS)

    def c8(p192, idx):
        return float(coarse8(torch.as_tensor(p192).float().cpu().view(-1, 3, 8, 8), TRUE8, idx)[0].mean())

    def final(name, p192, ref=None):
        r, v = score(torch.as_tensor(p192).float().cpu().view(-1, 3, 8, 8), TRUE8, local, groups, args.n_perm)
        if ref is not None:
            r["vs_baseline"] = paired_bootstrap(v, ref)
        out.setdefault("results", {})[name] = r
        d = r.get("vs_baseline"); extra = f"  vs baseline {d['mean_diff']:+.4f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]" if d else ""
        print(f"  {name:34s} coarse8 {r['coarse8']:+.4f}  p {r['permutation_p']:.4f}  pair id {r['pairwise_id']:.3f}{extra}")
        return v

    Eimg = torch.stack([x[img_rows == i].mean(0) for i in range(nS)]) * ZCA_SCALE     # (nS, 31, 250), THINGS cache scale

    # ---------------------------------------------------------------- THINGS side
    names = {R["files"][int(u)].name for u in uniq_all}
    keep = data_things.train_indices(exclude_basenames=names)
    T_all = data_things.eeg_means("train")                                              # (16540, 7750)
    YT8 = torch.as_tensor(pool8(T32["train"]))
    f_T = fit_primal_ridge(T_all[keep], YT8[keep].float(), THINGS_ALPHA)                # THINGS decoder to the 8x8 layout
    G_T = T_all.mean(0).view(31, 250); T_sh = T_all[things_train_i[shared]].view(-1, 31, 250)
    things_pred = lambda E: f_T(E.flatten(1).float())                                   # noqa: E731

    # ---------------------------------------------------------------- A  shared stimuli (no pixel labels)
    print("\nA  paired responses, E21 vs THINGS group mean (label-free)")
    E_sh = Eimg[shared]; Es = E_sh - E_sh.mean(0); Ts = T_sh - T_sh.mean(0); G_E = Eimg.mean(0)
    w_ga = slice(0, int(0.6 * FS)); w_im = slice(int(0.05 * FS), int(0.4 * FS))
    ga = [corr(shift(G_E, l)[:, w_ga], G_T[:, w_ga]) for l in LAGS]
    im = [corr(shift(Es, l)[..., w_im], Ts[..., w_im]) for l in LAGS]
    l_ga, l_im = LAGS[int(np.argmax(ga))], LAGS[int(np.argmax(im))]
    print(f"  grand-average response: r at lag 0 = {ga[LAGS.index(0)]:+.3f}; best lag {l_ga * 4:+d} ms (r {max(ga):+.3f})")
    print(f"  image-specific patterns ({len(shared)} images): r at lag 0 = {im[LAGS.index(0)]:+.4f}; best lag {l_im * 4:+d} ms (r {max(im):+.4f})")
    rng = np.random.default_rng(0)

    def identify(l):
        P = shift(Es, l)[..., w_im].flatten(1); Q = Ts[..., w_im].flatten(1)
        P = P - P.mean(1, keepdim=True); Q = Q - Q.mean(1, keepdim=True)
        M = ((P / P.norm(dim=1, keepdim=True)) @ (Q / Q.norm(dim=1, keepdim=True)).T).numpy()
        acc = lambda m: float(1 - (m > m.diagonal()[:, None]).sum(1).mean() / (len(m) - 1))   # noqa: E731
        null = np.array([acc(M[:, rng.permutation(len(M))]) for _ in range(2000)])
        return acc(M), float((np.sum(null >= acc(M)) + 1) / 2001)
    for l in sorted({0, l_ga, l_im}):
        a, p = identify(l); print(f"  E21 pattern identifies the THINGS pattern of the same image, lag {l * 4:+d} ms: rank accuracy {a:.3f} (chance .5, p {p:.4f})")
        out.setdefault("paired_identification", {})[str(l * 4)] = {"rank_accuracy": a, "p": p}
    Ea = shift(Es, l_ga)[..., w_im]; Tb = Ts[..., w_im]
    Ea = Ea - Ea.mean((0, 2), keepdim=True); Tb = Tb - Tb.mean((0, 2), keepdim=True)
    C = torch.einsum("ict,idt->cd", Ea, Tb) / (Ea.pow(2).sum((0, 2)).sqrt()[:, None] * Tb.pow(2).sum((0, 2)).sqrt()[None])
    diag = C.diagonal().numpy(); best = C.abs().argmax(1).numpy()
    print(f"  channel correspondence at lag {l_ga * 4:+d} ms: mean r(same channel) {diag.mean():+.3f}, negative for {int((diag < 0).sum())} of 31; "
          f"same channel is the best match for {int((best == np.arange(31)).sum())} of 31")
    print("    strongest same-channel r:", ", ".join(f"{STARSTIM_31[i]} {diag[i]:+.2f}" for i in np.argsort(-np.abs(diag))[:8]))
    out["lag_ms"] = {"grand_average": l_ga * 4, "image_specific": l_im * 4, "ga_curve": dict(zip([l * 4 for l in LAGS], np.round(ga, 4).tolist())),
                     "image_curve": dict(zip([l * 4 for l in LAGS], np.round(im, 4).tolist()))}
    out["channel_same_r"] = dict(zip(STARSTIM_31, np.round(diag, 3).tolist()))

    # ---------------------------------------------------------------- baseline first (reference for the paired tests)
    print("\nC  within-person ridge variants")
    Xtrial = torch.as_tensor(x.flatten(1).numpy(), dtype=torch.float64, device=DEV)

    def feats(t0, t1, pool):
        a, b = int(t0 * FS), int(t1 * FS); n = (b - a) // pool
        f = x[..., a:a + n * pool].reshape(len(x), 31, n, pool).mean(-1).flatten(1)
        ft = torch.as_tensor(f.numpy(), dtype=torch.float64, device=DEV)
        fm = torch.stack([ft[torch.as_tensor(img_rows == i)].mean(0) for i in range(nS)])
        return ft, fm
    arange = np.arange(nS)
    _, Xm_full = feats(0, 1.0, 1)
    p, ch = cv_predict(Xm_full, arange, Xm_full, Y8, c8, ALPHAS, fold)
    base = final("baseline 0-1s 250Hz means (repo)", p); out["baseline_alphas"] = ch
    within_pred = p

    # ---------------------------------------------------------------- B  THINGS decoder on E21
    print("\nB  THINGS decoder on E21")
    zs = {l: c8(things_pred(shift(Eimg, l)), local) for l in LAGS[::2] + [l_ga, l_im]}
    print("  zero-shot coarse8 by time shift (ms):", " ".join(f"{l * 4:+d}:{zs[l]:+.3f}" for l in LAGS[::2]))
    out["zeroshot_by_lag_ms"] = {str(l * 4): round(v, 4) for l, v in zs.items()}
    final("THINGS zero-shot, lag 0", things_pred(Eimg), base)
    if l_ga != 0:
        final(f"THINGS zero-shot, lag {l_ga * 4:+d} ms (grand avg)", things_pred(shift(Eimg, l_ga)), base)

    def fit_A(E, T, lam):                                       # T ~ A E over images x time, shrunk toward the identity
        E2 = E.permute(1, 0, 2).flatten(1).double(); T2 = T.permute(1, 0, 2).flatten(1).double(); I = torch.eye(31, dtype=torch.float64)
        return ((T2 @ E2.T + lam * I) @ torch.linalg.inv(E2 @ E2.T + lam * I)).float()

    def aligned(lag):
        Esh = shift(Eimg, lag); pred = torch.zeros(nS, 192); lams = []
        for k in range(8):
            tr_s = np.flatnonzero((fold != k) & (things_train_i >= 0)); te = np.flatnonzero(fold == k)
            Etr = Esh[tr_s] - Esh[tr_s].mean(0); Ttr = T_all[things_train_i[tr_s]].view(-1, 31, 250); Ttr = Ttr - Ttr.mean(0)
            s0 = float(Etr.pow(2).sum() / 31); grid = [s0 * g for g in (1e-2, 1e-1, 1, 10, 100)]; inner = np.arange(len(tr_s)) % 4; sc = []
            for lam in grid:
                r = [corr(torch.einsum("cd,idt->ict", fit_A(Etr[inner != j], Ttr[inner != j], lam), Etr[inner == j]), Ttr[inner == j]) for j in range(4)]
                sc.append(np.mean(r))
            lam = grid[int(np.argmax(sc))]; lams.append(lam / s0)
            A = fit_A(Etr, Ttr, lam)
            pred[te] = things_pred(torch.einsum("cd,idt->ict", A, Esh[te]))
        return pred, lams
    for lag in sorted({0, l_ga}):
        p, lams = aligned(lag)
        v = final(f"THINGS + spatial alignment, lag {lag * 4:+d} ms", p, base)
        out.setdefault("alignment_lambda_rel", {})[str(lag * 4)] = lams
        if lag == l_ga:
            z = lambda q: (q - q.mean(0)) / q.std()               # noqa: E731
            final("ensemble: aligned THINGS + within-person", z(p.double()) + z(within_pred.cpu()), base)

    # ---------------------------------------------------------------- C  continued
    print("\nC  within-person ridge variants (continued)")
    p, _ = cv_predict(Xm_full, arange, Xm_full, Y8, c8, WIDE, fold); final("0-1s 250Hz means, wide alpha grid", p, base)
    for t1 in (0.25, 0.4, 0.6):
        _, fm = feats(0, t1, 1); p, _ = cv_predict(fm, arange, fm, Y8, c8, WIDE, fold); final(f"0-{t1}s 250Hz means", p, base)
    ft, fm = feats(0, 0.4, 5); p, _ = cv_predict(fm, arange, fm, Y8, c8, WIDE, fold); final("0-0.4s 50Hz means", p, base)
    p, _ = cv_predict(Xtrial, img_rows, Xm_full, Y8, c8, WIDE, fold); final("0-1s 250Hz single-trial training", p, base)
    p, _ = cv_predict(ft, img_rows, fm, Y8, c8, WIDE, fold); final("0-0.4s 50Hz single-trial training", p, base)

    muT = YT8[keep].mean(0, keepdim=True).double()
    _, _, Vp = torch.pca_lowrank(YT8[keep].double() - muT, q=32, center=False, niter=6)
    Pt = f_T(data_things.eeg_means("test").float()).double(); Yt = torch.as_tensor(pool8(T32["test"]))
    _, _, Vh = torch.linalg.svd((Pt - Pt.mean(0)).T @ (Yt - Yt.mean(0)))
    for label, V in (("PCA of THINGS layouts", Vp), ("THINGS-decodable directions", Vh.T[:, :32])):
        for k in (8, 16, 32):
            Vk = V[:, :k].to(DEV); mu = muT.to(DEV)
            back = lambda q, idx, Vk=Vk, mu=mu: c8(q @ Vk.T + mu, idx)   # noqa: E731
            p, _ = cv_predict(fm, arange, fm, (Y8 - mu) @ Vk, back, WIDE, fold)
            final(f"0-0.4s 50Hz means, {label} k={k}", p @ Vk.T + mu, base)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"explore_rsvp_{args.session}.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote", RESULTS / f"explore_rsvp_{args.session}.json")


if __name__ == "__main__":
    main()
