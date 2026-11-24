"""
Collects every finished evaluation and training run into tables, figures and a short
text summary. Safe to run at any time, it only reports what has actually finished.

    python analyze.py config.json
Outputs go to <project>/analysis/.
"""

import csv
import glob
import json
import math
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CFG = json.load(open(sys.argv[1]))
P = CFG["project"]
OUT = os.path.join(P, "analysis")
os.makedirs(OUT, exist_ok=True)
CONDS = {c["name"]: c for c in CFG["conditions"]}
COND_ORDER = [c["name"] for c in CFG["conditions"]]

METHOD_LABEL = {
    "pretrained": "Pretrained diffusion policy",
    "pretrained_raw": "Pretrained (raw weights)",
    "dppo": "DPPO",
    "robust": "Robust DPPO",
}
COLORS = {"pretrained": "#7f7f7f", "dppo": "#1f77b4", "robust": "#d62728", "pretrained_raw": "#bcbd22"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def split_model(model):
    if model in ("pretrained", "pretrained_raw"):
        return model, None
    kind, seed = model.rsplit("_s", 1)
    return kind, int(seed)


# 1) Evaluation results
rows = []
for path in sorted(glob.glob(os.path.join(P, "runs", "eval", "*", "*", "result.npz"))):
    cond_name = os.path.basename(os.path.dirname(path))
    model = os.path.basename(os.path.dirname(os.path.dirname(path)))
    if cond_name not in CONDS:
        continue
    r = np.load(path)
    n = int(r["num_episode"])
    k = int(round(float(r["eval_success_rate"]) * n))
    lo, hi = wilson(k, n)
    method, seed = split_model(model)
    c = CONDS[cond_name]
    rows.append(dict(model=model, method=method, seed=seed if seed is not None else "", condition=cond_name,
                     type=c["type"], level=c["level"], episodes=n, successes=k, success_rate=k / max(n, 1),
                     ci95_low=lo, ci95_high=hi, source=os.path.relpath(path, P)))

with open(os.path.join(OUT, "results_all_evaluations.csv"), "w", newline="") as f:
    cols = ["model", "method", "seed", "condition", "type", "level", "episodes", "successes", "success_rate",
            "ci95_low", "ci95_high", "source"]
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

# pool seeds per method and condition
pooled = defaultdict(lambda: dict(k=0, n=0, seeds=[], rates=[]))
for r in rows:
    if r["method"] == "pretrained_raw":
        continue
    key = (r["method"], r["condition"])
    pooled[key]["k"] += r["successes"]
    pooled[key]["n"] += r["episodes"]
    pooled[key]["seeds"].append(r["seed"])
    pooled[key]["rates"].append(r["success_rate"])
methods = [m for m in ("pretrained", "dppo", "robust") if any(k[0] == m for k in pooled)]


def rate(method, cond):
    d = pooled.get((method, cond))
    if not d or d["n"] == 0:
        return None
    return d["k"] / d["n"]


def cell(method, cond):
    d = pooled.get((method, cond))
    if not d or d["n"] == 0:
        return "not run"
    lo, hi = wilson(d["k"], d["n"])
    s = f"{100 * d['k'] / d['n']:.0f}% ({d['k']}/{d['n']}) [{100 * lo:.0f}-{100 * hi:.0f}]"
    if len(d["rates"]) > 1:
        s += f", seeds {np.mean(d['rates']) * 100:.0f}+/-{np.std(d['rates'], ddof=1) * 100:.0f}"
    return s


def diff_ci(m1, m2, cond):
    """Difference m1 - m2 in percentage points with a normal-approximation 95% interval."""
    a, b = pooled.get((m1, cond)), pooled.get((m2, cond))
    if not a or not b or a["n"] == 0 or b["n"] == 0:
        return None
    p1, p2 = a["k"] / a["n"], b["k"] / b["n"]
    se = math.sqrt(p1 * (1 - p1) / a["n"] + p2 * (1 - p2) / b["n"])
    return 100 * (p1 - p2), 100 * 1.96 * se


# main table (markdown)
md = ["| Condition | " + " | ".join(METHOD_LABEL[m] for m in methods) + " |",
      "|---|" + "---|" * len(methods)]
for cn in COND_ORDER:
    md.append(f"| {CONDS[cn]['label']} | " + " | ".join(cell(m, cn) for m in methods) + " |")
main_table = "\n".join(md)
with open(os.path.join(OUT, "table_main.md"), "w") as f:
    f.write("Success rate (successes/episodes) [95% Wilson interval]\n\n" + main_table + "\n")

# 2) Figures
figs = []
if rows:
    # nominal bar chart
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    xs, labels = [], []
    for i, m in enumerate(methods):
        d = pooled.get((m, "nominal"))
        if not d:
            continue
        p = d["k"] / d["n"]
        lo, hi = wilson(d["k"], d["n"])
        ax.bar(i, 100 * p, color=COLORS[m], width=0.6)
        ax.errorbar(i, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], color="k", capsize=4)
        ax.text(i, 100 * hi + 2, f"{100 * p:.0f}%", ha="center", fontsize=9)
        xs.append(i)
        labels.append(METHOD_LABEL[m])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Robomimic Square, nominal conditions (95% CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_nominal_success.png"), dpi=200)
    fig.savefig(os.path.join(OUT, "fig_nominal_success.pdf"))
    plt.close(fig)
    figs.append("fig_nominal_success.png")

    # robustness curves, one panel per perturbation type
    types = []
    for cn in COND_ORDER:
        t = CONDS[cn]["type"]
        if t != "Nominal" and t not in types:
            types.append(t)
    fig, axes = plt.subplots(1, len(types), figsize=(3.3 * len(types), 3.3), squeeze=False)
    for ax, t in zip(axes[0], types):
        levels = [(CONDS["nominal"]["level_for"].get(t), "nominal")] + [
            (CONDS[cn]["level"], cn) for cn in COND_ORDER if CONDS[cn]["type"] == t]
        levels.sort(key=lambda x: x[0])
        for m in methods:
            pts = [(lv, rate(m, cn), pooled.get((m, cn))) for lv, cn in levels if rate(m, cn) is not None]
            if not pts:
                continue
            x = [p[0] for p in pts]
            y = [100 * p[1] for p in pts]
            err = np.array([[100 * (p[1] - wilson(p[2]["k"], p[2]["n"])[0]), 100 * (wilson(p[2]["k"], p[2]["n"])[1] - p[1])] for p in pts]).T
            ax.errorbar(x, y, yerr=err, marker="o", capsize=3, color=COLORS[m], label=METHOD_LABEL[m])
        ax.set_title(t, fontsize=10)
        ax.set_xlabel(CONDS[[cn for cn in COND_ORDER if CONDS[cn]["type"] == t][0]]["unit"])
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Success rate (%)")
    axes[0][-1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_robustness.png"), dpi=200)
    fig.savefig(os.path.join(OUT, "fig_robustness.pdf"))
    plt.close(fig)
    figs.append("fig_robustness.png")

# learning curves from in-training evaluations
train_runs = {}
for d in sorted(glob.glob(os.path.join(P, "runs", "train", "*"))):
    name = os.path.basename(d)
    hist = {}
    for pk in sorted(glob.glob(os.path.join(d, "result_seg*.pkl"))) + glob.glob(os.path.join(d, "result.pkl")):
        try:
            for r in pickle.load(open(pk, "rb")):
                hist.setdefault(r["itr"], {}).update(r)
        except Exception:
            pass
    done = os.path.exists(os.path.join(d, "DONE.json"))
    secs = json.load(open(os.path.join(d, "DONE.json"))).get("total_seconds") if done else None
    evals = sorted((r["itr"], r["step"], r["eval_success_rate"]) for r in hist.values() if "eval_success_rate" in r)
    last_step = max((r["step"] for r in hist.values()), default=0)
    train_runs[name] = dict(done=done, seconds=secs, evals=evals, last_itr=max(hist) if hist else None, env_steps=last_step)

if any(v["evals"] for v in train_runs.values()):
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for name, v in train_runs.items():
        if not v["evals"]:
            continue
        kind = name.rsplit("_s", 1)[0]
        x = [s / 1e6 for _, s, _ in v["evals"]]
        y = [100 * e for _, _, e in v["evals"]]
        style = "-" if name.endswith(f"_s{CFG['seeds_main'][0]}") else "--"
        lab = f"{METHOD_LABEL[kind]} (seed {name.rsplit('_s', 1)[1]})" + (" - evaluated with randomization on" if kind == "robust" else "")
        ax.plot(x, y, style, marker="o", color=COLORS[kind], label=lab)
    ax.set_xlabel("Environment steps used for fine-tuning (millions)")
    ax.set_ylabel("In-training eval success (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("Fine-tuning curves", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_learning_curves.png"), dpi=200)
    fig.savefig(os.path.join(OUT, "fig_learning_curves.pdf"))
    plt.close(fig)
    figs.append("fig_learning_curves.png")

# 3) Plain-language summary with the numbers filled in
perturbed = [cn for cn in COND_ORDER if cn != "nominal"]


def avg_perturbed(m):
    vals = [rate(m, cn) for cn in perturbed]
    vals = [v for v in vals if v is not None]
    return (100 * np.mean(vals), len(vals)) if vals else (None, 0)


def worst(m):
    vals = [(rate(m, cn), cn) for cn in perturbed if rate(m, cn) is not None]
    return min(vals) if vals else None


lines = ["# Results summary (auto-generated from finished runs only)", ""]
hw = CFG.get("hardware", {})
lines += [f"Hardware: {hw.get('gpu', '?')}, {hw.get('cpus', '?')} vCPUs, {hw.get('ram_gb', '?')} GB RAM (Google Colab).",
          f"Evaluation: {CFG['eval']['n_envs'] * CFG['eval']['episodes_per_env']} episodes per condition, eval seed {CFG['eval']['seed']}, "
          "same initial states for every method; 95% Wilson intervals.", ""]
lines += ["## Main table", "", main_table, ""]

raw = [r for r in rows if r["model"] == "pretrained_raw" and r["condition"] == "nominal"]
ema = pooled.get(("pretrained", "nominal"))
if raw or ema:
    lines += ["## Sanity check: raw vs EMA weights", ""]
    if raw:
        r = raw[0]
        lines.append(f"- Pretrained checkpoint, raw weights (what DPPO's eval script loads): {100 * r['success_rate']:.0f}% "
                     f"({r['successes']}/{r['episodes']}).")
    if ema:
        lines.append(f"- Same checkpoint, EMA weights (the policy DPPO actually starts from): {100 * ema['k'] / ema['n']:.0f}% "
                     f"({ema['k']}/{ema['n']}). This is the baseline used in the main table.")
    lines.append("")

lines += ["## Training runs", "", "| run | finished | iterations logged | env steps | wall time | in-training eval success (first -> last) |", "|---|---|---|---|---|---|"]
for name, v in train_runs.items():
    ev = v["evals"]
    curve = f"{100 * ev[0][2]:.0f}% -> {100 * ev[-1][2]:.0f}%" if ev else "-"
    wt = f"{v['seconds'] / 3600:.1f} h" if v["seconds"] else "-"
    lines.append(f"| {name} | {'yes' if v['done'] else 'no'} | {v['last_itr']} | {v['env_steps']:,} | {wt} | {curve} |")
lines.append("")

facts = []
dn = diff_ci("dppo", "pretrained", "nominal")
if dn:
    facts.append(f"DPPO vs pretrained, nominal: {dn[0]:+.0f} percentage points (+/- {dn[1]:.0f}, 95%).")
for m in methods:
    a, k = avg_perturbed(m)
    wv = worst(m)
    if a is not None:
        facts.append(f"{METHOD_LABEL[m]}: mean success over {k} perturbed conditions {a:.0f}%"
                     + (f"; worst case {100 * wv[0]:.0f}% ({CONDS[wv[1]]['label']})." if wv else "."))
for cn in perturbed:
    d = diff_ci("dppo", "pretrained", cn)
    if d:
        facts.append(f"DPPO - pretrained under {CONDS[cn]['label']}: {d[0]:+.0f} pp (+/- {d[1]:.0f}).")
for cn in COND_ORDER:
    d = diff_ci("robust", "dppo", cn)
    if d:
        facts.append(f"Robust DPPO - DPPO under {CONDS[cn]['label']}: {d[0]:+.0f} pp (+/- {d[1]:.0f}).")
lines += ["## Key numbers (check each one against the table before using it)", ""] + [f"- {x}" for x in facts] + [""]
lines += ["## Figures", ""] + [f"- {f}" for f in figs] + [""]
lines += ["## Files", "", "- results_all_evaluations.csv: one row per evaluation run, with its source file", "- table_main.md: the main table", ""]

with open(os.path.join(OUT, "RESULTS_SUMMARY.md"), "w") as f:
    f.write("\n".join(lines))
with open(os.path.join(OUT, "key_numbers.json"), "w") as f:
    json.dump(dict(pooled={f"{m}|{c}": v for (m, c), v in pooled.items()}, train_runs=train_runs, facts=facts), f, indent=1, default=str)
print("\n".join(lines))
