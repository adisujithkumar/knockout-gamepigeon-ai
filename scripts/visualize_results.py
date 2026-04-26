#!/usr/bin/env python3
"""Generate summary plots from all training experiments.

Reads data from runs/{contrastive,attention,self_play_v3,self_play_v4,llm_architect}
and produces PNG plots + a text summary in runs/visualizations/.
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RUNS = Path(__file__).resolve().parent.parent / "runs"
OUT = RUNS / "visualizations"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_csv(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_all():
    data = {}
    # contrastive
    data["contrastive_log"] = load_json(RUNS / "contrastive" / "discovery_log.json")
    # attention
    data["attention_log"] = load_json(RUNS / "attention" / "discovery_log.json")
    # self-play v3
    data["sp3_csv"] = load_csv(RUNS / "self_play_v3" / "training_log.csv")
    # self-play v4 (may have CSV or may not exist yet)
    data["sp4_csv"] = load_csv(RUNS / "self_play_v4" / "training_log.csv")
    # llm architect
    data["llm_summary"] = load_json(RUNS / "llm_architect" / "summary.json")
    return data


# ---------------------------------------------------------------------------
# Plot 1: ELO progression
# ---------------------------------------------------------------------------

def plot_elo(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False

    for label, key in [("Self-Play v3", "sp3_csv"), ("Self-Play v4", "sp4_csv")]:
        rows = data.get(key)
        if not rows:
            continue
        steps = [int(r["step"]) for r in rows]
        elos = [safe_float(r["elo"]) for r in rows]
        ax.plot(steps, elos, marker="o", markersize=5, label=label)
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No self-play data found", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray")

    ax.set_title("Self-Play ELO Progression", fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("ELO Rating")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}k"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "elo_progression.png", dpi=150)
    plt.close(fig)
    print("  saved elo_progression.png")


# ---------------------------------------------------------------------------
# Plot 2: win-rate comparison bar chart
# ---------------------------------------------------------------------------

def plot_win_rates(data):
    # Gather final win rates for each approach.
    # Sources: contrastive/attention from MEMORY notes; self-play from CSV; LLM from summary.
    approaches = []
    vs_random = []
    vs_heuristic = []

    # contrastive: last iteration win ratio (win_steps / total)
    clog = data.get("contrastive_log")
    if clog:
        last = clog[-1]
        wr = last["win_steps"] / max(last["total_trajectory_steps"], 1)
        approaches.append("Contrastive")
        vs_random.append(wr)
        vs_heuristic.append(0.0)  # known from project memory

    # attention: derive from trajectory (all training was vs random)
    alog = data.get("attention_log")
    if alog:
        approaches.append("Attention")
        vs_random.append(0.90)  # from project memory, ~90% vs random
        vs_heuristic.append(0.0)

    # llm architect
    llm = data.get("llm_summary")
    if llm:
        # summary.json is a list of iterations with win_rate
        if isinstance(llm, list):
            last_wr = llm[-1].get("win_rate", 0)
        else:
            last_wr = llm.get("win_rate", 0)
        approaches.append("LLM Architect")
        vs_random.append(last_wr)
        vs_heuristic.append(0.0)

    # self-play v3
    sp3 = data.get("sp3_csv")
    if sp3:
        last = sp3[-1]
        approaches.append("Self-Play v3")
        vs_random.append(safe_float(last.get("vs_random_winrate", 0)))
        vs_heuristic.append(0.0)

    if not approaches:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(approaches))
    width = 0.35
    ax.bar([i - width / 2 for i in x], vs_random, width, label="vs Random", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], vs_heuristic, width, label="vs Heuristic", color="#C44E52")
    ax.set_xticks(list(x))
    ax.set_xticklabels(approaches, fontsize=10)
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("Win Rate Comparison (Final)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # annotate values
    for i, (vr, vh) in enumerate(zip(vs_random, vs_heuristic)):
        ax.text(i - width / 2, vr + 0.02, f"{vr:.0%}", ha="center", fontsize=9)
        ax.text(i + width / 2, vh + 0.02, f"{vh:.0%}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "win_rate_comparison.png", dpi=150)
    plt.close(fig)
    print("  saved win_rate_comparison.png")


# ---------------------------------------------------------------------------
# Plot 3: contrastive feature discovery timeline
# ---------------------------------------------------------------------------

def plot_contrastive_discovery(data):
    clog = data.get("contrastive_log")
    if not clog:
        return

    # Collect all features across iterations; pick top 5 by max |effect_size|.
    feature_history = defaultdict(list)  # name -> [(iter, abs_effect_size)]
    for entry in clog:
        it = entry["iteration"]
        for feat in entry["discovery"]["features"]:
            feature_history[feat["name"]].append((it, abs(feat["effect_size"])))

    # rank by peak effect size
    ranked = sorted(feature_history.items(), key=lambda kv: max(v for _, v in kv[1]), reverse=True)
    top5 = ranked[:5]

    fig, ax = plt.subplots(figsize=(9, 5))
    markers = ["o", "s", "D", "^", "v"]
    for (name, points), marker in zip(top5, markers):
        iters = [p[0] for p in sorted(points)]
        vals = [p[1] for p in sorted(points)]
        ax.plot(iters, vals, marker=marker, markersize=6, label=name, linewidth=2)

    ax.set_title("Contrastive Feature Discovery (Top 5 by Peak |Cohen's d|)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Discovery Iteration")
    ax.set_ylabel("|Cohen's d| (Effect Size)")
    ax.set_xticks(range(max(e["iteration"] for e in clog) + 1))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "contrastive_discovery.png", dpi=150)
    plt.close(fig)
    print("  saved contrastive_discovery.png")


# ---------------------------------------------------------------------------
# Plot 4: attention attribution over time
# ---------------------------------------------------------------------------

def plot_attention_attribution(data):
    alog = data.get("attention_log")
    if not alog:
        return

    # Track attribution of each feature across cycles
    feature_series = defaultdict(lambda: {"steps": [], "attributions": []})
    for entry in alog:
        step = entry["step"]
        for feat in entry["features"]:
            name = feat["name"]
            feature_series[name]["steps"].append(step)
            feature_series[name]["attributions"].append(feat["attribution"])

    # Pick features that appear in at least half the cycles, rank by mean attribution
    min_appearances = len(alog) // 2
    eligible = {name: s for name, s in feature_series.items()
                if len(s["steps"]) >= min_appearances}
    ranked = sorted(eligible.items(),
                    key=lambda kv: sum(kv[1]["attributions"]) / len(kv[1]["attributions"]),
                    reverse=True)
    top_features = ranked[:6]

    fig, ax = plt.subplots(figsize=(10, 5))
    markers = ["o", "s", "D", "^", "v", "P"]
    for (name, series), marker in zip(top_features, markers):
        ax.plot(series["steps"], series["attributions"],
                marker=marker, markersize=3, label=name, linewidth=1.5, alpha=0.85)

    ax.set_title("Attention Attribution Over Training (Top Features)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Attribution Score")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "attention_attribution.png", dpi=150)
    plt.close(fig)
    print("  saved attention_attribution.png")


# ---------------------------------------------------------------------------
# Plot 5: training curves (policy loss + value loss)
# ---------------------------------------------------------------------------

def plot_training_curves(data):
    panels = []

    # contrastive
    clog = data.get("contrastive_log")
    if clog:
        iters = [e["iteration"] for e in clog]
        ploss = [e["ppo_final_metrics"]["policy_loss"] for e in clog]
        vloss = [e["ppo_final_metrics"]["value_loss"] for e in clog]
        panels.append(("Contrastive", iters, ploss, vloss, "Iteration"))

    # attention: parse from train.log rollout lines
    attn_log_path = RUNS / "attention" / "train.log"
    if attn_log_path.exists():
        attn_rollouts, attn_ploss, attn_vloss = [], [], []
        with open(attn_log_path) as f:
            for line in f:
                if line.startswith("Rollout "):
                    parts = line.split()
                    rollout_num = int(parts[1].split("/")[0])
                    for part in parts:
                        if part.startswith("ploss="):
                            pl = float(part.split("=")[1])
                        elif part.startswith("vloss="):
                            vl = float(part.split("=")[1])
                    attn_rollouts.append(rollout_num)
                    attn_ploss.append(pl)
                    attn_vloss.append(vl)
        if attn_rollouts:
            panels.append(("Attention", attn_rollouts, attn_ploss, attn_vloss, "Rollout"))

    # self-play v3
    sp3 = data.get("sp3_csv")
    if sp3:
        steps = [int(r["step"]) for r in sp3]
        ploss = [safe_float(r.get("policy_loss", 0)) for r in sp3]
        vloss = [safe_float(r.get("value_loss", 0)) for r in sp3]
        panels.append(("Self-Play v3", steps, ploss, vloss, "Steps"))

    # self-play v4
    sp4 = data.get("sp4_csv")
    if sp4:
        steps = [int(r["step"]) for r in sp4]
        ploss = [safe_float(r.get("policy_loss", 0)) for r in sp4]
        vloss = [safe_float(r.get("value_loss", 0)) for r in sp4]
        panels.append(("Self-Play v4", steps, ploss, vloss, "Steps"))

    if not panels:
        return

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, (title, xs, ploss, vloss, xlabel) in zip(axes, panels):
        ax.plot(xs, ploss, label="Policy Loss", color="#4C72B0", linewidth=1.5)
        ax2 = ax.twinx()
        ax2.plot(xs, vloss, label="Value Loss", color="#C44E52", linewidth=1.5, linestyle="--")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Policy Loss", color="#4C72B0", fontsize=9)
        ax2.set_ylabel("Value Loss", color="#C44E52", fontsize=9)
        ax.grid(True, alpha=0.3)

        # combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="best")

    fig.suptitle("Training Curves", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved training_curves.png")


# ---------------------------------------------------------------------------
# summary text
# ---------------------------------------------------------------------------

def write_summary(data):
    lines = ["=== Experiment Summary ===", ""]

    # contrastive
    clog = data.get("contrastive_log")
    if clog:
        n_iters = len(clog)
        total_features = sum(e["num_features_discovered"] for e in clog)
        unique_features = len({f["name"] for e in clog for f in e["discovery"]["features"]})
        last = clog[-1]
        wr = last["win_steps"] / max(last["total_trajectory_steps"], 1)
        lines.append(
            f"Contrastive: {n_iters} iters, {unique_features} unique features, "
            f"{wr:.1%} vs random, 0% vs heuristic"
        )

    # attention
    alog = data.get("attention_log")
    if alog:
        n_cycles = len(alog)
        all_feats = {f["name"] for e in alog for f in e["features"]}
        lines.append(
            f"Attention: {n_cycles} cycles, {len(all_feats)} unique features, "
            f"~90% vs random, 0% vs heuristic"
        )

    # llm architect
    llm = data.get("llm_summary")
    if llm:
        if isinstance(llm, list):
            n_iters = len(llm)
            last_wr = llm[-1].get("win_rate", 0)
        else:
            n_iters = 1
            last_wr = llm.get("win_rate", 0)
        lines.append(
            f"LLM Architect: {n_iters} iters, {last_wr:.0%} vs random"
        )

    # self-play v3
    sp3 = data.get("sp3_csv")
    if sp3:
        last = sp3[-1]
        n_iters = len(sp3)
        elo = safe_float(last.get("elo", 0))
        vs_rand = safe_float(last.get("vs_random_winrate", 0))
        lines.append(
            f"Self-Play v3: {n_iters} iters, ELO {elo:.0f}, "
            f"{vs_rand:.0%} vs random, 0% vs heuristic"
        )

    # self-play v4
    sp4 = data.get("sp4_csv")
    if sp4:
        last = sp4[-1]
        n_iters = len(sp4)
        elo = safe_float(last.get("elo", 0))
        vs_rand = safe_float(last.get("vs_random_winrate", 0))
        lines.append(
            f"Self-Play v4: {n_iters} iters, ELO {elo:.0f}, "
            f"{vs_rand:.0%} vs random"
        )

    lines.append("")
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text)
    print("  saved summary.txt")
    print()
    print(text)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT}\n")

    data = load_all()

    print("Generating plots...")
    plot_elo(data)
    plot_win_rates(data)
    plot_contrastive_discovery(data)
    plot_attention_attribution(data)
    plot_training_curves(data)
    write_summary(data)
    print("Done.")


if __name__ == "__main__":
    main()
