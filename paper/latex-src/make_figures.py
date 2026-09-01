"""Monte Carlo study of Section 12 and Figures 1-4.

DGP (Section 12.1): Theta ~ N(0, tau2); root j: E_j = Theta + eps_j;
report i on root j: R_ij = E_j + eta_ij. tau2 = sig2 = nu2 = 1.
60,000 realizations per cell, seed 20260818.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(20260818)
REPS = 60_000
TAU2 = SIG2 = NU2 = 1.0
Z95 = 1.959963984540054
NS = [1, 2, 4, 8, 16, 32]

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "figure.dpi": 150,
})
C_BAYES = "#1f4e79"   # dark blue
C_NAIVE = "#b03a2e"   # brick red

# ---------------- k = 1: report multiplication with one root ----------------
res = {}
for n in NS:
    theta = rng.normal(0.0, np.sqrt(TAU2), REPS)
    eps = rng.normal(0.0, np.sqrt(SIG2), REPS)
    eta_bar = rng.normal(0.0, np.sqrt(NU2 / n), REPS)
    rbar = theta + eps + eta_bar

    # dependence-aware Bayes (correct covariance)
    J = n / (NU2 + n * SIG2)
    var_b = 1.0 / (1.0 / TAU2 + J)
    mu_b = J * rbar * var_b
    # naive: n conditionally independent reports of variance sig2+nu2
    Jn = n / (SIG2 + NU2)
    var_n = 1.0 / (1.0 / TAU2 + Jn)
    mu_n = Jn * rbar * var_n

    def stats(mu, var):
        err = theta - mu
        cov = np.mean(np.abs(err) <= Z95 * np.sqrt(var))
        nll = np.mean(0.5 * np.log(2 * np.pi * var) + err**2 / (2 * var))
        rmse = np.sqrt(np.mean(err**2))
        return cov, nll, rmse / np.sqrt(var)

    res[n] = (*stats(mu_b, var_b), *stats(mu_n, var_n))

print("n | Bayes cov | Naive cov | Bayes NLL | Naive NLL | C_bayes | C_naive")
for n in NS:
    cb, lb, Cb, cn, ln_, Cn = res[n]
    print(f"{n:2d} |   {cb:.3f}   |   {cn:.3f}   |  {lb:.3f}   |  {ln_:.3f}   | {Cb:.3f}  | {Cn:.3f}")

# Figure 1: calibration ratio vs n (k = 1)
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(NS, [res[n][2] for n in NS], "o-", color=C_BAYES, lw=1.4, ms=4,
        label="Dependence-aware Bayes")
ax.plot(NS, [res[n][5] for n in NS], "s--", color=C_NAIVE, lw=1.4, ms=4,
        label="Naive independent pooling")
ax.axhline(1.0, color="gray", lw=0.8, ls=":")
ax.set_xscale("log", base=2)
ax.set_xticks(NS); ax.set_xticklabels(NS)
ax.set_xlabel(r"Number of reports $n$ (one primitive root, $k=1$)")
ax.set_ylabel(r"Calibration ratio $C=\mathrm{RMSE}/\mathrm{posterior\ s.d.}$")
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig("figures/fig1_calibration.pdf"); plt.close(fig)

# Figure 4: mean negative log score vs n (k = 1)
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(NS, [res[n][1] for n in NS], "o-", color=C_BAYES, lw=1.4, ms=4,
        label="Dependence-aware Bayes")
ax.plot(NS, [res[n][4] for n in NS], "s--", color=C_NAIVE, lw=1.4, ms=4,
        label="Naive independent pooling")
ax.set_xscale("log", base=2)
ax.set_xticks(NS); ax.set_xticklabels(NS)
ax.set_xlabel(r"Number of reports $n$ (one primitive root, $k=1$)")
ax.set_ylabel("Mean negative log score")
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig("figures/fig4_logscore.pdf"); plt.close(fig)

# ---------- Figure 2: information at n = 16 as root count k varies ----------
KS = [1, 2, 4, 8, 16]
info_true = []
for k in KS:
    m = 16 // k
    Jk = k * m / (NU2 + m * SIG2)
    info_true.append(0.5 * np.log(1 + TAU2 * Jk))
info_naive = 0.5 * np.log(1 + TAU2 * 16 / (SIG2 + NU2))

print("\nk | I(Theta;R) nats (n = 16)")
for k, i in zip(KS, info_true):
    print(f"{k:2d} | {i:.3f}")
print(f"naive implied: {info_naive:.3f}")

fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(KS, info_true, "o-", color=C_BAYES, lw=1.4, ms=4,
        label=r"Actual $I(\Theta;R)$")
ax.axhline(info_naive, color=C_NAIVE, lw=1.4, ls="--",
           label="Implied by naive pooling")
ax.set_xscale("log", base=2)
ax.set_xticks(KS); ax.set_xticklabels(KS)
ax.set_xlabel(r"Number of independent primitive roots $k$ (fixed $n=16$)")
ax.set_ylabel(r"Information about $\Theta$ (nats)")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig("figures/fig2_roots_info.pdf"); plt.close(fig)

# ------------- Figure 3: precision scaling, shared vs independent -----------
m = np.arange(1, 33)
J_shared = m / (NU2 + m * SIG2)
J_ind = m / (SIG2 + NU2)
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(m, J_ind, "--", color=C_BAYES, lw=1.4, label="Independent roots")
ax.plot(m, J_shared, "-", color=C_NAIVE, lw=1.4, label="Shared root")
ax.axhline(1.0 / SIG2, color="gray", lw=0.8, ls=":",
           label=r"Source ceiling $1/\sigma^2$")
ax.set_xlabel(r"Number of reports $m$")
ax.set_ylabel(r"Likelihood precision $J_m$")
ax.set_xlim(1, 32)
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig("figures/fig3_precision.pdf"); plt.close(fig)

print("\nfigures written")
