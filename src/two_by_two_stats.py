"""Final statistical round on the 2x2 evaluation set. $0 -- no new API
calls, all on already-collected data. Four diagnostics requested before
freezing:

1. Factorial contrasts (representation, ancestry, interaction) on inferred
   cluster count, paired bootstrap by world.
2. Confirmatory Delta_BC = E[cos(B)] - E[cos(C)] on the 200 eval worlds
   (pilot gave 0.242 [0.216, 0.270] on 30 pilot worlds; not used to select
   or tune anything).
3. Paired contrasts (coverage, NLL, calibration ratio) between C-A and
   D-B for the balanced dedup aggregator: same evidence, same estimates,
   representation-only difference.
4. Exact oracle sweep over every distinct observed cosine-similarity value
   in cells B and C (not just the 14-point grid), reporting
   min_t max(FMR_B(t), FSR_C(t)) and min_t sqrt(FMR_B(t)^2+FSR_C(t)^2).

Usage:
    python src/two_by_two_stats.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import worldgen
import aggregate
import embed
from two_by_two import load_pool, build_cells, CELLS, CELL_DESIGN

ROOT = Path(__file__).resolve().parents[1]


def load_everything():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    ab_fit = json.loads((ROOT / "results" / "processed" / "calibration_fit.json").read_text())
    tt_fit = json.loads((ROOT / "results" / "processed" / "two_by_two_calibration_fit.json").read_text())
    pool = load_pool(cfg, "two_by_two_eval")
    cells_by_world = build_cells(pool)
    master_seed = cfg["master_seed"]
    theta_by_world = {wid: worldgen.generate_world(master_seed, wid, cfg).theta for wid in cells_by_world}
    return cfg, ab_fit, tt_fit, cells_by_world, theta_by_world


def cluster_count(values, rationales, threshold):
    vecs = embed.embed_texts(rationales)
    clusters = aggregate._cluster_by_threshold(vecs, threshold)
    return len(clusters)


def bootstrap_ci(values, n_boot=5000, seed=0):
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    return boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


# ---------------------------------------------------------------------------
# 1. Factorial contrasts on cluster count
# ---------------------------------------------------------------------------

def diagnostic_1_factorial_contrasts(cells_by_world, threshold):
    worlds = sorted(w for w, c in cells_by_world.items() if all(x in c for x in "ABCD"))
    counts = {c: [] for c in "ABCD"}
    for wid in worlds:
        for c in "ABCD":
            values = [v for v, _ in cells_by_world[wid][c]]
            rationales = [r for _, r in cells_by_world[wid][c]]
            counts[c].append(cluster_count(values, rationales, threshold))

    n = len(worlds)
    rep_effect = [((counts["C"][i] + counts["D"][i]) - (counts["A"][i] + counts["B"][i])) / 2 for i in range(n)]
    anc_effect = [((counts["B"][i] + counts["D"][i]) - (counts["A"][i] + counts["C"][i])) / 2 for i in range(n)]
    interaction = [(counts["D"][i] - counts["C"][i]) - (counts["B"][i] - counts["A"][i]) for i in range(n)]

    def summarize(effect, label):
        mean = sum(effect) / n
        lo, hi = bootstrap_ci(effect)
        print(f"{label}: {mean:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n_worlds={n})")
        return {"mean": mean, "ci95": [lo, hi]}

    print(f"\n--- Diagnostic 1: factorial contrasts on cluster count (threshold={threshold}) ---")
    print(f"cell means: A={sum(counts['A'])/n:.3f} B={sum(counts['B'])/n:.3f} "
          f"C={sum(counts['C'])/n:.3f} D={sum(counts['D'])/n:.3f}")
    out = {
        "cell_means": {c: sum(v) / n for c, v in counts.items()},
        "representation_effect": summarize(rep_effect, "Representation effect (C+D)/2 - (A+B)/2"),
        "ancestry_effect": summarize(anc_effect, "Ancestry effect (B+D)/2 - (A+C)/2"),
        "interaction_effect": summarize(interaction, "Interaction (D-C)-(B-A)"),
        "n_worlds": n,
    }
    return out


# ---------------------------------------------------------------------------
# 2. Confirmatory Delta_BC on evaluation set
# ---------------------------------------------------------------------------

def _mean_pairwise_cosine(texts):
    vecs = embed.embed_texts(texts)
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(embed.cosine_similarity(vecs[i], vecs[j]))
    return sims


def diagnostic_2_confirmatory_delta_bc(cells_by_world):
    b_means, c_means = [], []
    for wid, cells in cells_by_world.items():
        if "B" in cells:
            sims = _mean_pairwise_cosine([t for _, t in cells["B"]])
            if sims:
                b_means.append(sum(sims) / len(sims))
        if "C" in cells:
            sims = _mean_pairwise_cosine([t for _, t in cells["C"]])
            if sims:
                c_means.append(sum(sims) / len(sims))

    mean_b, mean_c = sum(b_means) / len(b_means), sum(c_means) / len(c_means)
    delta = mean_b - mean_c

    rng = random.Random(1)
    boots = []
    for _ in range(5000):
        bb = [b_means[rng.randrange(len(b_means))] for _ in b_means]
        cc = [c_means[rng.randrange(len(c_means))] for _ in c_means]
        boots.append(sum(bb) / len(bb) - sum(cc) / len(cc))
    boots.sort()
    ci = [boots[int(0.025 * 5000)], boots[int(0.975 * 5000) - 1]]

    print("\n--- Diagnostic 2: confirmatory Delta_BC on 200 evaluation worlds ---")
    print(f"mean cos(B)={mean_b:.4f}  mean cos(C)={mean_c:.4f}  "
          f"Delta_BC={delta:.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    print("(pilot, 30 worlds: Delta_BC=0.242, CI [0.216, 0.270] -- not reused/updated here)")
    return {"mean_cos_B": mean_b, "mean_cos_C": mean_c, "delta_bc": delta, "delta_bc_ci95": ci,
            "n_worlds_b": len(b_means), "n_worlds_c": len(c_means)}


# ---------------------------------------------------------------------------
# 3. Paired downstream contrasts (C-A, D-B), dedup_balanced
# ---------------------------------------------------------------------------

def diagnostic_3_paired_downstream(cfg, ab_fit, tt_fit, cells_by_world, theta_by_world):
    prior_mean, prior_sd = cfg["dgp"]["prior_mean"], cfg["dgp"]["prior_sd"]
    sigma_r2 = ab_fit["sigma_r2"]
    t = tt_fit["balanced_dedup_threshold"]
    z95 = 1.959963984540054

    worlds = sorted(w for w, c in cells_by_world.items() if all(x in c for x in "ABCD"))
    per_cell = {c: {"err": [], "sd": [], "nll": [], "covered": []} for c in "ABCD"}

    for wid in worlds:
        theta = theta_by_world[wid]
        for c in "ABCD":
            values = [v for v, _ in cells_by_world[wid][c]]
            rationales = [r for _, r in cells_by_world[wid][c]]
            post, _ = aggregate.dedup_pool(values, rationales, prior_mean, prior_sd, sigma_r2, t)
            err = post.mean - theta
            per_cell[c]["err"].append(err)
            per_cell[c]["sd"].append(post.sd)
            per_cell[c]["nll"].append(aggregate.nll(theta, post))
            per_cell[c]["covered"].append(1.0 if abs(err) <= z95 * post.sd else 0.0)

    n = len(worlds)

    def paired_ratio_bootstrap(cell_hi, cell_lo, n_boot=5000, seed=2):
        rng = random.Random(seed)
        boots = []
        idx = list(range(n))
        for _ in range(n_boot):
            sample = [rng.randrange(n) for _ in idx]
            rmse_hi = math.sqrt(sum(per_cell[cell_hi]["err"][i] ** 2 for i in sample) / n)
            sd_hi = sum(per_cell[cell_hi]["sd"][i] for i in sample) / n
            rmse_lo = math.sqrt(sum(per_cell[cell_lo]["err"][i] ** 2 for i in sample) / n)
            sd_lo = sum(per_cell[cell_lo]["sd"][i] for i in sample) / n
            boots.append((rmse_hi / sd_hi) - (rmse_lo / sd_lo))
        boots.sort()
        return boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]

    def report_pair(hi, lo, label):
        d_cov = [per_cell[hi]["covered"][i] - per_cell[lo]["covered"][i] for i in range(n)]
        d_nll = [per_cell[hi]["nll"][i] - per_cell[lo]["nll"][i] for i in range(n)]

        rmse_hi = math.sqrt(sum(e ** 2 for e in per_cell[hi]["err"]) / n)
        sd_hi = sum(per_cell[hi]["sd"]) / n
        rmse_lo = math.sqrt(sum(e ** 2 for e in per_cell[lo]["err"]) / n)
        sd_lo = sum(per_cell[lo]["sd"]) / n
        c_hi, c_lo = rmse_hi / sd_hi, rmse_lo / sd_lo
        d_c = c_hi - c_lo
        d_c_ci = paired_ratio_bootstrap(hi, lo)

        mean_d_cov, cov_ci = sum(d_cov) / n, bootstrap_ci(d_cov)
        mean_d_nll, nll_ci = sum(d_nll) / n, bootstrap_ci(d_nll)

        print(f"\n{label}: same evidence & estimates, representation only differs ({hi} vs {lo})")
        print(f"  Delta coverage = {mean_d_cov:+.3f}  95% CI [{cov_ci[0]:+.3f}, {cov_ci[1]:+.3f}]")
        print(f"  Delta NLL      = {mean_d_nll:+.3f}  95% CI [{nll_ci[0]:+.3f}, {nll_ci[1]:+.3f}]")
        print(f"  Delta calib. C = {d_c:+.3f}  95% CI [{d_c_ci[0]:+.3f}, {d_c_ci[1]:+.3f}]  "
              f"(C_{hi}={c_hi:.3f}, C_{lo}={c_lo:.3f})")
        return {
            "delta_coverage": {"mean": mean_d_cov, "ci95": list(cov_ci)},
            "delta_nll": {"mean": mean_d_nll, "ci95": list(nll_ci)},
            "delta_calibration_ratio": {"value": d_c, "ci95": list(d_c_ci)},
        }

    print(f"\n--- Diagnostic 3: paired downstream contrasts, dedup (balanced t={t}) ---")
    out = {
        "C_minus_A": report_pair("C", "A", "C - A (shared root: dissimilar vs similar representation)"),
        "D_minus_B": report_pair("D", "B", "D - B (independent roots: dissimilar vs similar representation)"),
        "n_worlds": n,
    }
    return out


# ---------------------------------------------------------------------------
# 4. Exact oracle sweep over observed similarity values
# ---------------------------------------------------------------------------

def diagnostic_4_exact_oracle_sweep(cells_by_world):
    b_rationales_by_world = {wid: [r for _, r in cells["B"]] for wid, cells in cells_by_world.items() if "B" in cells}
    c_rationales_by_world = {wid: [r for _, r in cells["C"]] for wid, cells in cells_by_world.items() if "C" in cells}

    all_sims = set()
    for texts in list(b_rationales_by_world.values()) + list(c_rationales_by_world.values()):
        vecs = embed.embed_texts(texts)
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                all_sims.add(round(embed.cosine_similarity(vecs[i], vecs[j]), 10))
    candidate_thresholds = sorted(all_sims)
    # a cut point strictly between consecutive distinct values (and past the max)
    cuts = [candidate_thresholds[0] - 1e-6]
    for i in range(len(candidate_thresholds) - 1):
        cuts.append((candidate_thresholds[i] + candidate_thresholds[i + 1]) / 2)
    cuts.append(candidate_thresholds[-1] + 1e-6)

    def rate_at(t, rationales_by_world, want_same_root):
        flags = []
        for wid, texts in rationales_by_world.items():
            vecs = embed.embed_texts(texts)
            clusters = aggregate._cluster_by_threshold(vecs, t)
            cluster_of = {}
            for ci, idxs in enumerate(clusters):
                for i in idxs:
                    cluster_of[i] = ci
            n = len(texts)
            for i in range(n):
                for j in range(i + 1, n):
                    same_cluster = cluster_of[i] == cluster_of[j]
                    if want_same_root:  # cell C: true relation is same-root; false split = different cluster
                        flags.append(not same_cluster)
                    else:  # cell B: true relation is different-root; false merge = same cluster
                        flags.append(same_cluster)
        return sum(flags) / len(flags) if flags else float("nan")

    rows = []
    for t in cuts:
        fmr_b = rate_at(t, b_rationales_by_world, want_same_root=False)
        fsr_c = rate_at(t, c_rationales_by_world, want_same_root=True)
        rows.append({"threshold": t, "fmr_b": fmr_b, "fsr_c": fsr_c,
                      "max": max(fmr_b, fsr_c), "l2": math.sqrt(fmr_b ** 2 + fsr_c ** 2)})

    best_minimax = min(rows, key=lambda r: r["max"])
    best_l2 = min(rows, key=lambda r: r["l2"])

    print(f"\n--- Diagnostic 4: exact oracle sweep over {len(candidate_thresholds)} "
          f"distinct observed cosine values ({len(cuts)} cut points) ---")
    print(f"min_t max(FMR_B, FSR_C): t={best_minimax['threshold']:.6f}  "
          f"FMR_B={best_minimax['fmr_b']:.4f}  FSR_C={best_minimax['fsr_c']:.4f}  "
          f"max={best_minimax['max']:.4f}")
    print(f"min_t sqrt(FMR_B^2+FSR_C^2): t={best_l2['threshold']:.6f}  "
          f"FMR_B={best_l2['fmr_b']:.4f}  FSR_C={best_l2['fsr_c']:.4f}  "
          f"l2={best_l2['l2']:.4f}")
    return {"n_candidate_thresholds": len(cuts), "best_minimax": best_minimax, "best_l2": best_l2,
            "sweep": rows}


def main():
    cfg, ab_fit, tt_fit, cells_by_world, theta_by_world = load_everything()
    threshold = tt_fit["balanced_dedup_threshold"]

    d1 = diagnostic_1_factorial_contrasts(cells_by_world, threshold)
    d2 = diagnostic_2_confirmatory_delta_bc(cells_by_world)
    d3 = diagnostic_3_paired_downstream(cfg, ab_fit, tt_fit, cells_by_world, theta_by_world)
    d4 = diagnostic_4_exact_oracle_sweep(cells_by_world)

    out = {"factorial_contrasts": d1, "confirmatory_delta_bc": d2,
           "paired_downstream_contrasts": d3, "exact_oracle_sweep": {
               k: v for k, v in d4.items() if k != "sweep"  # sweep is large; omit from summary, keep separately
           }}
    out_path = ROOT / "results" / "processed" / "two_by_two_final_stats.json"
    out_path.write_text(json.dumps(out, indent=2))
    sweep_path = ROOT / "results" / "processed" / "two_by_two_exact_oracle_sweep.json"
    sweep_path.write_text(json.dumps(d4["sweep"], indent=2))
    print(f"\nwrote {out_path}")
    print(f"wrote {sweep_path}")


if __name__ == "__main__":
    main()
