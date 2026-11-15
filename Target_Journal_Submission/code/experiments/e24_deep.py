"""E24: deep-learning uncertainty baselines for the ECG studies (comparators, no hypothesis).

Protocol: research_design/experimental_protocol.md, row E24 (committed in 3de341c before this script was written
or run). A 1-D residual CNN in the style of the ResNet baseline of Wang et al. (2017) for time series
classification, reduced to fit the CPU budget: three residual blocks with 32/64/64 channels, each three
convolutions with kernel sizes 8/5/3, batch norm and ReLU, a 1x1 shortcut, global average pooling, dropout 0.2
and a linear output layer. Trained on the clean PTB-XL training records (folds 1-8, as in the E20 primary
analysis) with per-lead standardisation from training statistics, Adam (learning rate 1e-3), batch 64,
20 epochs, cross-entropy, no early stopping. Five networks with seeds 0-4.

Reported: Deep-single (network 0, dropout off), Deep-ensemble (mean of the five networks' probabilities),
MC-dropout (network 0, dropout on, mean of 20 stochastic passes).
Outputs: results/e24/ptbxl.npz (PTB-XL test folds 9-10) and results/e24/cinc.npz (CinC 2011 set A).
Usage: python experiments/e24_deep.py [train|predict|analyze]   (default: all three)
analyze: E20 metrics of the deep baselines on the PTB-XL test folds -> results/e24/summary.json
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ptbxl                                                              # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e24"
SEEDS, EPOCHS, BATCH, LR, MC_PASSES = range(5), 20, 64, 1e-3, 20
torch.set_num_threads(int(__import__("os").environ.get("E24_THREADS", "4")))


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        layers, c = [], cin
        for k in (8, 5, 3):
            layers += [nn.Conv1d(c, cout, k, padding="same"), nn.BatchNorm1d(cout)]
            if k != 3:
                layers.append(nn.ReLU())
            c = cout
        self.body = nn.Sequential(*layers)
        self.short = nn.Sequential(nn.Conv1d(cin, cout, 1), nn.BatchNorm1d(cout))
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.body(x) + self.short(x))


class ResNet1d(nn.Module):
    def __init__(self, n_in=12, n_out=5, widths=(32, 64, 64), p_drop=0.2):
        super().__init__()
        blocks, c = [], n_in
        for w in widths:
            blocks.append(Block(c, w)); c = w
        self.blocks = nn.Sequential(*blocks)
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(c, n_out)

    def forward(self, x):
        return self.fc(self.drop(self.blocks(x).mean(-1)))


def data():
    X, y, noisy, fold, ids = ptbxl.load()
    tr = (fold <= 8) & ~noisy
    mu = X[tr].mean(axis=(0, 2), keepdims=True); sd = X[tr].std(axis=(0, 2), keepdims=True) + 1e-6
    return X, y, noisy, fold, ids, tr, mu.astype(np.float32), sd.astype(np.float32)


def train(seed):
    path = OUT / f"net_s{seed}.pt"
    if path.exists():
        return
    X, y, noisy, fold, ids, tr, mu, sd = data()
    Xtr = torch.from_numpy((X[tr] - mu) / sd); ytr = torch.from_numpy(y[tr])
    del X
    torch.manual_seed(seed); np.random.seed(seed)
    net = ResNet1d(); opt = torch.optim.Adam(net.parameters(), lr=LR); lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(ytr), generator=g); tot, t0 = 0.0, time.time()
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad(); loss = lossf(net(Xtr[b]), ytr[b]); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        print(f"seed {seed} epoch {ep + 1} loss {tot / len(ytr):.4f} {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), path)


@torch.no_grad()
def proba(net, X, dropout=False, passes=1, seed=0):
    net.eval()
    if dropout:
        net.drop.train()
    torch.manual_seed(seed)
    P = 0
    for _ in range(passes):
        out = []
        for i in range(0, len(X), 256):
            out.append(torch.softmax(net(torch.from_numpy(X[i:i + 256])), 1).numpy())
        P = P + np.concatenate(out)
    return P / passes


def predict():
    import e23_cinc
    X, y, noisy, fold, ids, tr, mu, sd = data()
    te = fold >= 9
    sets = {"ptbxl": ((X[te] - mu) / sd, dict(y=y[te], noisy=noisy[te], ecg_id=ids[te]))}
    Xc, unacc, rec = e23_cinc.load()
    sets["cinc"] = ((Xc - mu) / sd, dict(unacc=unacc, rec=rec))
    nets = []
    for s in SEEDS:
        n = ResNet1d(); n.load_state_dict(torch.load(OUT / f"net_s{s}.pt")); nets.append(n)
    for name, (Z, meta) in sets.items():
        Z = Z.astype(np.float32)
        single = [proba(n, Z) for n in nets]
        out = dict(meta, **{"P_Deep-single": single[0], "P_Deep-ensemble": np.mean(single, 0),
                            "P_MC-dropout": proba(nets[0], Z, dropout=True, passes=MC_PASSES)})
        for s, P in zip(SEEDS, single):
            out[f"P_net{s}"] = P
        np.savez_compressed(OUT / f"{name}.npz", **out)
        print(name, {k: v.shape for k, v in out.items() if k.startswith("P_")}, flush=True)


def analyze():
    import json
    from e20_ptbxl import describe
    z = dict(np.load(OUT / "ptbxl.npz")); y, noisy = z["y"], z["noisy"].astype(bool)
    summ = {m: describe(z[f"P_{m}"], y, noisy) for m in ("Deep-single", "Deep-ensemble", "MC-dropout")}
    summ["single_nets"] = [describe(z[f"P_net{s}"], y, noisy) for s in SEEDS]
    summ["n_params"] = sum(p.numel() for p in ResNet1d().parameters())
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    for m in ("Deep-single", "Deep-ensemble", "MC-dropout"):
        d = summ[m]
        print(f"{m:14s} acc {d['acc']:.3f} nll_noisy {d['nll_noisy']:.3f} det {d['detect_auroc']:.3f} "
              f"caught@20 {d['caught@20']:.3f} err@20 {d['auto_err@20']:.3f}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("train", "all"):
        for s in SEEDS:
            train(s)
    if what in ("predict", "all"):
        predict()
    if what in ("analyze", "all"):
        analyze()
