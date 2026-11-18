import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIMENSIONS = ["quality", "diversity", "controlability"]

# Composite score: weighted geometric mean of the three dimensions.
# A geometric mean punishes a generator that collapses on one dimension (e.g. very diverse but unplayable),
# which an arithmetic mean would hide. The weights are the argument made in the report for Mario levels.
def composite(scores, weights, eps=1e-6):
    w = np.array([weights[d] for d in DIMENSIONS], dtype=float)
    v = np.array([max(scores[d], eps) for d in DIMENSIONS])
    return float(np.exp((w * np.log(v)).sum() / w.sum()))

def weighted_mean(scores, weights):
    return sum(weights[d] * scores[d] for d in DIMENSIONS) / sum(weights.values())

def load_histories(results, name, key):
    curves = []
    for path in sorted(glob.glob(os.path.join(results, name, "run_*.json"))):
        with open(path) as f:
            history = json.load(f)["history"]
        # --single runs store the shared history only once
        if history:
            curves.append([h.get(key, np.nan) for h in history])
    if not curves:
        return None
    # early-stopped runs keep their final value until the longest run ends
    length = max(len(c) for c in curves)
    return np.array([c + [c[-1]] * (length - len(c)) for c in curves], dtype=float)

def plot_radar(summary, names, path):
    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(1, len(names) + 1, subplot_kw={"polar": True}, figsize=(4 * (len(names) + 1), 4))
    for ax, name in zip(axes, names):
        values = [summary[name][d] for d in DIMENSIONS] + [summary[name][DIMENSIONS[0]]]
        passes = [summary[name][d + "_pass"] for d in DIMENSIONS] + [summary[name][DIMENSIONS[0] + "_pass"]]
        ax.plot(angles, values, label="mean value")
        ax.fill(angles, values, alpha=0.25)
        ax.plot(angles, passes, linestyle="--", label="pass rate")
        ax.set_title(name)
    for name in names:
        values = [summary[name][d] for d in DIMENSIONS] + [summary[name][DIMENSIONS[0]]]
        axes[-1].plot(angles, values, label=name)
        axes[-1].fill(angles, values, alpha=0.15)
    axes[-1].set_title("all generators")
    for ax in axes:
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(DIMENSIONS)
        ax.set_ylim(0, 1)
    axes[0].legend(loc="lower left", bbox_to_anchor=(-0.3, -0.25), fontsize="small")
    axes[-1].legend(loc="lower right", bbox_to_anchor=(1.3, -0.25), fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# each generator optimises its own fitness, so each gets its own panel instead of sharing one axis
FITNESS_LABELS = {
    "genetic": ("best: (Q + C·[Q≥1] + D·[C≥1]) / 3", "mean fitness of the population"),
    "random_search": ("best: (Q + C·[Q≥1] + D·[C≥1]) / 3", "mean fitness of the last batch"),
    "map_elites": ("target elite: (Q + C·[Q≥1]) / 2, one run, first target", "mean quality of the archive"),
    "q": ("best greedy level so far: (Q + C·[Q≥1]) / 2", "mean fitness of the ε-greedy episode"),
}

def plot_episode_reward(results, name, path):
    # the brief asks for Episode Reward evolution plots for the RL generator
    # entry 0 is logged at reset, before any episode, so it is left out
    curves = load_histories(results, name, "episode_reward")[:, 1:]
    fig, ax = plt.subplots(figsize=(7, 4))
    mean, std = np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)
    x = np.arange(1, len(mean) + 1)
    ax.plot(x, mean, label=f"mean over {len(curves)} runs")
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("episode")
    ax.set_ylabel("episode reward (return)")
    ax.set_title(f"{name}: episode reward (= reward scale × final fitness)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_fitness(results, names, path):
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), squeeze=False)
    for ax, name in zip(axes[0], names):
        best_label, mean_label = FITNESS_LABELS.get(name, ("best fitness", "mean fitness"))
        n_runs = 0
        for key, style, label in [("best", "-", best_label), ("mean", ":", mean_label)]:
            curves = load_histories(results, name, key)
            if curves is None or np.isnan(curves).all():
                continue
            n_runs = len(curves)
            mean, std = np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)
            x = np.arange(len(mean))
            line, = ax.plot(x, mean, style, label=label)
            if key == "best" and n_runs > 1:
                ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.2, color=line.get_color())
        ax.set_title(f"{name} ({n_runs} run{'s' if n_runs != 1 else ''}{', mean ± std' if n_runs > 1 else ''})")
        ax.set_xlabel("generation / iteration")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize="small", loc="best")
    axes[0][0].set_ylabel("fitness")
    fig.suptitle("Fitness evolution (each generator's own fitness: not on a common scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_archive(results, name, path):
    # one enemies x jumps grid per coins value; colour = elite quality, empty cells blank, x = evaluated targets
    with open(os.path.join(results, name, "archive", "archive.json")) as f:
        archive = json.load(f)
    n_enemies, n_jumps, n_coins = archive["bins"]
    grids = np.full((n_coins, n_jumps, n_enemies), np.nan)
    for elite in archive["elites"]:
        e, j, c = elite["cell"]
        grids[c, j, e] = elite["quality"]
    targets = []
    for p in glob.glob(os.path.join(results, name, "run_*.json")):
        with open(p) as f:
            control = json.load(f)["control"]
        targets.append((control["enemies"], control["jumps"], control["coins"]))

    fig, axes = plt.subplots(1, n_coins, figsize=(3 * n_coins, 3.6), sharey=True)
    for c, ax in enumerate(axes):
        image = ax.imshow(grids[c], origin="lower", vmin=0, vmax=1, cmap="viridis")
        for e, j, tc in targets:
            if tc == c:
                ax.plot(e, j, "rx")
        ax.set_title(f"coins = {c}")
        ax.set_xlabel("enemies")
    axes[0].set_ylabel("jumps")
    filled = np.count_nonzero(~np.isnan(grids))
    fig.suptitle(f"{name} archive: {filled}/{grids.size} cells filled (colour = quality, x = evaluated targets)")
    fig.colorbar(image, ax=axes, shrink=0.8)
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_coverage(results, name, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in [("coverage", "cells filled"), ("playable_coverage", "cells with a playable level")]:
        curves = load_histories(results, name, key)
        if curves is not None:
            ax.plot(curves[0], label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("fraction of the archive")
    ax.set_title(f"{name} archive coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Composite score, radar charts and fitness plots from evaluate.py results.")
    parser.add_argument("results", type=str, help="the evaluate.py output folder")
    parser.add_argument("--wq", type=float, default=0.5, help="weight of quality")
    parser.add_argument("--wd", type=float, default=0.3, help="weight of diversity")
    parser.add_argument("--wc", type=float, default=0.2, help="weight of controlability")
    args = parser.parse_args()

    with open(os.path.join(args.results, "summary.json")) as f:
        summary = json.load(f)
    names = [n for n in summary if not n.startswith("_")]
    weights = {"quality": args.wq, "diversity": args.wd, "controlability": args.wc}

    plots = os.path.join(args.results, "plots")
    os.makedirs(plots, exist_ok=True)

    rows = []
    for name in names:
        s = summary[name]
        # the same composite on pass rates: the benchmark's continuous controlability gives ~0.99 to near misses,
        # so the fraction of levels that actually meet each criterion is the stricter summary
        passes = {d: s[d + "_pass"] for d in DIMENSIONS}
        rows.append((name, s["quality"], s["diversity"], s["controlability"],
                     weighted_mean(s, weights), composite(s, weights), composite(passes, weights)))
    rows.sort(key=lambda r: r[5], reverse=True)

    header = "generator,quality,diversity,controlability,weighted_mean,composite_geo,composite_geo_pass"
    lines = [header] + [f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},{r[6]:.4f}" for r in rows]
    with open(os.path.join(plots, "composite.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"weights {weights}")
    print("\n".join(lines))
    print(f"best overall: {rows[0][0]}")

    plot_radar(summary, names, os.path.join(plots, "radar.png"))
    plot_fitness(args.results, names, os.path.join(plots, "fitness.png"))
    for name in names:
        rewards = load_histories(args.results, name, "episode_reward")
        if rewards is not None and not np.isnan(rewards).all():
            plot_episode_reward(args.results, name, os.path.join(plots, f"episode_reward_{name}.png"))
        if os.path.exists(os.path.join(args.results, name, "archive", "archive.json")):
            plot_archive(args.results, name, os.path.join(plots, f"archive_{name}.png"))
            plot_coverage(args.results, name, os.path.join(plots, f"coverage_{name}.png"))
    print(f"plots saved in {plots}")
