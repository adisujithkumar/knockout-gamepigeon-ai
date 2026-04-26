#!/usr/bin/env python3
"""Web dashboard for Knockout experiment results and checkpoint management."""

import csv
import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

from flask import Flask, send_from_directory

app = Flask(__name__)
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

# ---------------------------------------------------------------------------
# Shared HTML
# ---------------------------------------------------------------------------

STYLE = """
<style>
  body { background: #1a1a2e; color: #e0e0e0; font-family: 'Consolas', monospace; padding: 20px; }
  table { border-collapse: collapse; width: 100%; margin: 20px 0; }
  th, td { border: 1px solid #333; padding: 8px 12px; text-align: left; }
  th { background: #16213e; }
  tr:nth-child(even) { background: #1a1a2e; }
  tr:nth-child(odd) { background: #0f0f23; }
  a { color: #4fc3f7; }
  .metric-good { color: #4caf50; font-weight: bold; }
  .metric-bad { color: #f44336; font-weight: bold; }
  .metric-neutral { color: #ffd54f; }
  pre { background: #0f0f23; padding: 15px; border-radius: 5px; overflow-x: auto; }
  code { color: #a5d6a7; }
  img { max-width: 100%; border: 1px solid #333; margin: 10px 0; }
  nav { margin-bottom: 20px; }
  nav a { margin-right: 15px; text-decoration: none; font-size: 1.1em; }
  h1, h2, h3 { color: #bb86fc; }
  .btn { display: inline-block; background: #16213e; color: #4fc3f7; border: 1px solid #4fc3f7;
         padding: 4px 10px; border-radius: 4px; text-decoration: none; font-family: inherit;
         font-size: 0.85em; cursor: pointer; }
  .btn:hover { background: #1a3a5c; }
  .card { background: #0f0f23; border: 1px solid #333; border-radius: 6px; padding: 15px;
          margin: 10px 0; }
  .gallery { display: flex; flex-wrap: wrap; gap: 20px; }
  .gallery img { max-width: 48%; }
  .cmd { background: #0f0f23; color: #4fc3f7; padding: 8px 12px; border-radius: 4px;
         font-family: inherit; display: inline-block; margin: 4px 0; }
  .sort-links a { margin-right: 10px; font-size: 0.9em; }
</style>
"""

NAV = """
<nav>
  <a href="/">Overview</a>
  <a href="/checkpoints">Checkpoints</a>
  <a href="/status">Live Status</a>
  <a href="/plots">Plots</a>
</nav>
"""


def page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title} - Knockout Dashboard</title>{STYLE}</head>
<body>{NAV}<h1>{title}</h1>{body}</body></html>"""


# ---------------------------------------------------------------------------
# Helpers — data loading
# ---------------------------------------------------------------------------

def _color_pct(val: float, good_thresh: float = 0.8, bad_thresh: float = 0.3) -> str:
    """Return CSS class for a percentage metric."""
    if val >= good_thresh:
        return "metric-good"
    if val <= bad_thresh:
        return "metric-bad"
    return "metric-neutral"


def _sizeof_fmt(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def _load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_csv(path: Path) -> list[dict]:
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _tail(path: Path, n: int = 20) -> str:
    try:
        lines = open(path).readlines()
        return "".join(lines[-n:])
    except Exception:
        return "(no log file)"


def _collect_experiments() -> list[dict]:
    """Build summary rows for all known experiments."""
    experiments = []

    # --- contrastive ---
    data = _load_json(RUNS_DIR / "contrastive" / "discovery_log.json")
    if data and isinstance(data, list):
        last = data[-1]
        total_feats = sum(e.get("num_features_discovered", 0) for e in data)
        experiments.append(dict(
            name="contrastive", status="complete", iterations=len(data),
            elo="-",
            vs_random="99.7%", vs_heuristic="0%",
            detail=f"{total_feats} features discovered",
        ))

    # --- attention ---
    data = _load_json(RUNS_DIR / "attention" / "discovery_log.json")
    if data and isinstance(data, list):
        experiments.append(dict(
            name="attention", status="complete", iterations=len(data),
            elo="-",
            vs_random="~90%", vs_heuristic="0%",
            detail=f"{len(data)} attribution cycles",
        ))

    # --- llm_architect ---
    data = _load_json(RUNS_DIR / "llm_architect" / "summary.json")
    if data and isinstance(data, list):
        best_wr = max((e.get("win_rate", 0) for e in data), default=0)
        experiments.append(dict(
            name="llm_architect", status="complete", iterations=len(data),
            elo="-",
            vs_random=f"{best_wr*100:.0f}%", vs_heuristic="0%",
            detail="LLM-designed reward",
        ))

    # --- llm_cli ---
    data = _load_json(RUNS_DIR / "llm_cli" / "summary.json")
    if data and isinstance(data, list):
        last = data[-1]
        best_wr = max(
            (e.get("stats", {}).get("win_rate", 0) for e in data), default=0
        )
        experiments.append(dict(
            name="llm_cli", status="complete", iterations=len(data),
            elo="-",
            vs_random=f"{best_wr*100:.1f}%", vs_heuristic="-",
            detail="CLI reward architect",
        ))

    # --- llm_cli_v2 ---
    if (RUNS_DIR / "llm_cli_v2").is_dir():
        iters = sorted(glob(str(RUNS_DIR / "llm_cli_v2" / "iteration_*")))
        experiments.append(dict(
            name="llm_cli_v2",
            status="in-progress" if len(iters) <= 1 else "complete",
            iterations=len(iters), elo="-",
            vs_random="-", vs_heuristic="-",
            detail="CLI reward v2",
        ))

    # --- self_play variants ---
    for sp in ("self_play", "self_play_v2", "self_play_v3", "self_play_v4"):
        sp_dir = RUNS_DIR / sp
        if not sp_dir.is_dir():
            continue
        csv_rows = _load_csv(sp_dir / "training_log.csv")
        pool_idx = _load_json(sp_dir / "pool_index.json")
        if csv_rows:
            last = csv_rows[-1]
            elo_val = last.get("elo", "-")
            try:
                elo_val = f"{float(elo_val):.1f}"
            except (ValueError, TypeError):
                pass
            vs_rand = last.get("vs_random_winrate", "-")
            try:
                vs_rand = f"{float(vs_rand)*100:.0f}%"
            except (ValueError, TypeError):
                pass
            experiments.append(dict(
                name=sp, status="complete", iterations=len(csv_rows),
                elo=elo_val, vs_random=vs_rand, vs_heuristic="0%",
                detail=f"pool size {last.get('pool_size', '?')}",
            ))
        else:
            # directory exists but no CSV yet — probably in-progress
            has_pid = (sp_dir / "train.pid").exists()
            pool_size = len(glob(str(sp_dir / "pool" / "*.pt")))
            experiments.append(dict(
                name=sp,
                status="running" if has_pid else "started",
                iterations=0, elo="-",
                vs_random="-", vs_heuristic="-",
                detail=f"pool size {pool_size}",
            ))

    return experiments


def _collect_checkpoints() -> list[dict]:
    """Find all .pt checkpoint files across runs/."""
    patterns = [
        str(RUNS_DIR / "*" / "checkpoints" / "*.pt"),
        str(RUNS_DIR / "*" / "pool" / "*.pt"),
        str(RUNS_DIR / "*" / "iteration_*" / "agent_checkpoint.pt"),
        str(RUNS_DIR / "*" / "final_agent.pt"),
    ]
    results = []
    for pat in patterns:
        for fp in glob(pat):
            p = Path(fp)
            stat = p.stat()
            # determine approach from path
            rel = p.relative_to(RUNS_DIR)
            approach = rel.parts[0]
            # extract step/iteration from filename
            name = p.stem
            step = ""
            if "step_" in name:
                step = name.split("step_")[-1].lstrip("0") or "0"
            elif "iter_" in name:
                step = "iter " + (name.split("iter_")[-1].lstrip("0") or "0")
            elif "iteration_" in str(p):
                for part in p.parts:
                    if part.startswith("iteration_"):
                        step = "iter " + (part.split("_")[-1].lstrip("0") or "0")
            elif name == "final_agent":
                step = "final"
            # try to get ELO from pool_index
            elo = "-"
            pool_idx = _load_json(RUNS_DIR / approach / "pool_index.json")
            if pool_idx:
                for entry in pool_idx:
                    if p.name in entry.get("path", ""):
                        try:
                            elo = f"{float(entry['elo']):.1f}"
                        except (ValueError, TypeError, KeyError):
                            pass
            results.append(dict(
                path=str(p),
                relpath=str(rel),
                approach=approach,
                step=step,
                elo=elo,
                size=_sizeof_fmt(stat.st_size),
                size_bytes=stat.st_size,
                mtime=datetime.fromtimestamp(stat.st_mtime),
            ))
    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def overview():
    experiments = _collect_experiments()
    rows = ""
    for e in experiments:
        status_cls = "metric-good" if e["status"] == "complete" else "metric-neutral"
        vs_r_cls = _color_pct(
            float(e["vs_random"].rstrip("%")) / 100
            if e["vs_random"] not in ("-", "~90%") else 0.9,
            good_thresh=0.8, bad_thresh=0.3,
        )
        rows += f"""<tr>
  <td><a href="/experiment/{e['name']}">{e['name']}</a></td>
  <td class="{status_cls}">{e['status']}</td>
  <td>{e['iterations']}</td>
  <td>{e['elo']}</td>
  <td class="{vs_r_cls}">{e['vs_random']}</td>
  <td>{e['vs_heuristic']}</td>
  <td>{e['detail']}</td>
</tr>"""

    body = f"""
<table>
<tr><th>Approach</th><th>Status</th><th>Iterations</th><th>ELO</th>
    <th>vs Random</th><th>vs Heuristic</th><th>Detail</th></tr>
{rows}
</table>
<h2>Quick Summary</h2>
<div class="card"><pre>{_tail(RUNS_DIR / "visualizations" / "summary.txt", 20)}</pre></div>
"""
    return page("Experiment Overview", body)


@app.route("/experiment/<name>")
def experiment_detail(name: str):
    exp_dir = RUNS_DIR / name
    if not exp_dir.is_dir():
        return page("Not Found", f"<p>No experiment directory: {name}</p>"), 404

    sections: list[str] = []

    # ---- Training metrics from CSV ----
    csv_rows = _load_csv(exp_dir / "training_log.csv")
    if csv_rows:
        headers = list(csv_rows[0].keys())
        hdr = "".join(f"<th>{h}</th>" for h in headers)
        trows = ""
        for r in csv_rows:
            cells = ""
            for h in headers:
                v = r.get(h, "")
                try:
                    fv = float(v)
                    if "winrate" in h or "win_rate" in h:
                        cls = _color_pct(fv)
                        cells += f'<td class="{cls}">{fv:.3f}</td>'
                    elif "elo" in h:
                        cells += f"<td>{fv:.1f}</td>"
                    elif "loss" in h or "entropy" in h:
                        cells += f"<td>{fv:.4f}</td>"
                    else:
                        cells += f"<td>{v}</td>"
                except (ValueError, TypeError):
                    cells += f"<td>{v}</td>"
            trows += f"<tr>{cells}</tr>"
        sections.append(
            f"<h2>Training Log (CSV)</h2><table><tr>{hdr}</tr>{trows}</table>"
        )

    # ---- Discovery log (contrastive) ----
    if name == "contrastive":
        data = _load_json(exp_dir / "discovery_log.json")
        if data:
            sections.append("<h2>Feature Discovery</h2>")
            for entry in data:
                it = entry.get("iteration", "?")
                feats = entry.get("discovery", {}).get("features", [])
                nf = entry.get("num_features_discovered", 0)
                ws = entry.get("win_steps", 0)
                ls = entry.get("loss_steps", 0)
                total = ws + ls
                wr = f"{ws/total*100:.1f}%" if total > 0 else "-"
                sections.append(
                    f"<h3>Iteration {it} &mdash; {nf} features, win rate {wr}</h3>"
                )
                if feats:
                    rows_html = ""
                    for f in feats:
                        d = f.get("direction", 0)
                        d_str = "+" if d > 0 else "-"
                        d_cls = "metric-good" if d > 0 else "metric-bad"
                        es = abs(f.get("effect_size", 0))
                        rows_html += (
                            f'<tr><td>{f.get("name","?")}</td>'
                            f'<td>{f.get("index","")}</td>'
                            f"<td>{es:.3f}</td>"
                            f'<td class="{d_cls}">{d_str}</td></tr>'
                        )
                    sections.append(
                        "<table><tr><th>Feature</th><th>Index</th>"
                        f"<th>Cohen's d</th><th>Direction</th></tr>{rows_html}</table>"
                    )

    # ---- Discovery log (attention) ----
    if name == "attention":
        data = _load_json(exp_dir / "discovery_log.json")
        if data:
            sections.append("<h2>Attribution Scores Over Time</h2>")
            # show first 5 and last 5 cycles
            show = data[:5] + ([{"_sep": True}] if len(data) > 10 else []) + data[-5:]
            for entry in show:
                if entry.get("_sep"):
                    sections.append(f"<p class='metric-neutral'>... {len(data)-10} more cycles ...</p>")
                    continue
                step = entry.get("step", "?")
                feats = entry.get("features", [])
                sections.append(f"<h3>Step {step}</h3>")
                if feats:
                    rows_html = ""
                    for f in feats[:8]:
                        sign_cls = "metric-good" if f.get("sign", 1) > 0 else "metric-bad"
                        rows_html += (
                            f'<tr><td>{f.get("name","?")}</td>'
                            f'<td>{f.get("weight",0):.4f}</td>'
                            f'<td>{f.get("attribution",0):.4f}</td>'
                            f'<td class="{sign_cls}">{"+" if f.get("sign",1)>0 else "-"}</td></tr>'
                        )
                    sections.append(
                        "<table><tr><th>Feature</th><th>Weight</th>"
                        f"<th>Attribution</th><th>Sign</th></tr>{rows_html}</table>"
                    )

    # ---- LLM architect / cli — reward function code ----
    if name.startswith("llm"):
        summary = _load_json(exp_dir / "summary.json")
        iters = sorted(glob(str(exp_dir / "iteration_*")))
        for it_dir in iters:
            it_path = Path(it_dir)
            it_name = it_path.name
            rf = it_path / "reward_function.py"
            met = _load_json(it_path / "metrics.json")

            sections.append(f"<h2>{it_name}</h2>")

            # metrics
            if met:
                sections.append("<div class='card'>")
                if isinstance(met, dict):
                    # flatten nested dicts
                    flat = {}
                    for k, v in met.items():
                        if isinstance(v, dict):
                            for k2, v2 in v.items():
                                flat[f"{k}.{k2}"] = v2
                        else:
                            flat[k] = v
                    for k, v in flat.items():
                        if isinstance(v, float):
                            sections.append(f"<b>{k}:</b> {v:.4f}<br>")
                        else:
                            sections.append(f"<b>{k}:</b> {v}<br>")
                sections.append("</div>")

            # reward code
            if rf.exists():
                code = rf.read_text()
                sections.append(f"<h3>Reward Function</h3><pre><code>{code}</code></pre>")

    # ---- Relevant visualizations ----
    vis_dir = RUNS_DIR / "visualizations"
    if vis_dir.is_dir():
        images = []
        for png in sorted(vis_dir.glob("*.png")):
            # match relevant plots
            if name in png.stem or (
                name == "contrastive" and "contrastive" in png.stem
            ) or (
                name == "attention" and "attention" in png.stem
            ) or (
                "self_play" in name and ("elo" in png.stem or "training" in png.stem)
            ):
                images.append(png.name)
        if images:
            sections.append("<h2>Visualizations</h2><div class='gallery'>")
            for img in images:
                sections.append(
                    f'<img src="/plots/image/{img}" alt="{img}"><br>'
                )
            sections.append("</div>")

    # ---- Checkpoints for this experiment ----
    ckpts = [c for c in _collect_checkpoints() if c["approach"] == name]
    if ckpts:
        ckpts.sort(key=lambda c: c["mtime"])
        rows_html = ""
        for c in ckpts:
            rows_html += (
                f"<tr><td>{c['relpath']}</td><td>{c['step']}</td>"
                f"<td>{c['elo']}</td><td>{c['size']}</td>"
                f"<td>{c['mtime'].strftime('%Y-%m-%d %H:%M')}</td></tr>"
            )
        sections.append(
            "<h2>Checkpoints</h2>"
            "<table><tr><th>Path</th><th>Step</th><th>ELO</th>"
            f"<th>Size</th><th>Date</th></tr>{rows_html}</table>"
        )

    # ---- Train log tail ----
    log = _tail(exp_dir / "train.log", 30)
    sections.append(f"<h2>Train Log (last 30 lines)</h2><pre>{log}</pre>")

    return page(f"Experiment: {name}", "\n".join(sections))


@app.route("/checkpoints")
def checkpoints():
    sort_by = "date"  # default
    ckpts = _collect_checkpoints()

    # provide sort links
    header = """<div class="sort-links">Sort by:
        <a href="/checkpoints?sort=date">Date</a>
        <a href="/checkpoints?sort=elo">ELO</a>
        <a href="/checkpoints?sort=approach">Approach</a>
        <a href="/checkpoints?sort=size">Size</a>
    </div>"""

    from flask import request
    sort_by = request.args.get("sort", "date")
    if sort_by == "elo":
        def elo_key(c):
            try:
                return -float(c["elo"])
            except (ValueError, TypeError):
                return 0
        ckpts.sort(key=elo_key)
    elif sort_by == "approach":
        ckpts.sort(key=lambda c: (c["approach"], c["mtime"]))
    elif sort_by == "size":
        ckpts.sort(key=lambda c: -c["size_bytes"])
    else:
        ckpts.sort(key=lambda c: c["mtime"], reverse=True)

    rows = ""
    for c in ckpts:
        cmd = f"python scripts/play.py --opponent-type ppo --opponent-checkpoint {c['relpath']}"
        rows += f"""<tr>
  <td>{c['relpath']}</td>
  <td>{c['approach']}</td>
  <td>{c['step']}</td>
  <td>{c['elo']}</td>
  <td>{c['size']}</td>
  <td>{c['mtime'].strftime('%Y-%m-%d %H:%M')}</td>
  <td><span class="cmd">{cmd}</span></td>
</tr>"""

    body = f"""{header}
<p>{len(ckpts)} checkpoints found</p>
<table>
<tr><th>Path</th><th>Approach</th><th>Step</th><th>ELO</th>
    <th>Size</th><th>Date</th><th>Play Command</th></tr>
{rows}
</table>"""
    return page("Checkpoint Manager", body)


@app.route("/status")
def status():
    # auto-refresh
    refresh = '<meta http-equiv="refresh" content="30">'

    # find training processes
    try:
        ps = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5
        )
        lines = ps.stdout.strip().split("\n")
        header_line = lines[0] if lines else ""
        train_lines = [
            l for l in lines[1:]
            if any(kw in l for kw in ("train_", "self_play", "train.py", "train_monitored"))
            and "grep" not in l
        ]
    except Exception as e:
        header_line = ""
        train_lines = []

    if train_lines:
        proc_rows = ""
        for l in train_lines:
            parts = l.split(None, 10)
            if len(parts) >= 11:
                proc_rows += (
                    f"<tr><td>{parts[1]}</td><td>{parts[2]}</td>"
                    f"<td>{parts[3]}</td><td>{parts[9]}</td>"
                    f"<td>{parts[10][:80]}</td></tr>"
                )
        proc_table = (
            "<table><tr><th>PID</th><th>CPU%</th><th>MEM%</th>"
            f"<th>Start</th><th>Command</th></tr>{proc_rows}</table>"
        )
    else:
        proc_table = '<p class="metric-neutral">No training processes detected.</p>'

    # PID files
    pid_info = ""
    for pid_file in glob(str(RUNS_DIR / "*" / "train.pid")):
        pf = Path(pid_file)
        try:
            pid = pf.read_text().strip()
            approach = pf.parent.name
            # check if PID is alive
            alive = os.path.isdir(f"/proc/{pid}")
            status_str = (
                '<span class="metric-good">RUNNING</span>'
                if alive
                else '<span class="metric-bad">STOPPED</span>'
            )
            pid_info += f"<p>{approach}: PID {pid} {status_str}</p>"
        except Exception:
            pass

    # recent log tails
    log_sections = ""
    for log_file in sorted(glob(str(RUNS_DIR / "*" / "train.log")))[-5:]:
        lf = Path(log_file)
        approach = lf.parent.name
        tail = _tail(lf, 20)
        log_sections += (
            f"<h3>{approach}/train.log</h3><pre>{tail}</pre>"
        )
    # also check top-level logs
    for log_file in sorted(glob(str(RUNS_DIR / "*.log")))[-3:]:
        lf = Path(log_file)
        tail = _tail(lf, 20)
        log_sections += f"<h3>{lf.name}</h3><pre>{tail}</pre>"

    body = f"""{refresh}
<p class="metric-neutral">Auto-refreshes every 30 seconds</p>
<h2>Training Processes</h2>
{proc_table}
{pid_info}
<h2>Recent Logs</h2>
{log_sections}
"""
    return page("Live Training Status", body)


@app.route("/plots")
def plots():
    vis_dir = RUNS_DIR / "visualizations"
    images = []
    if vis_dir.is_dir():
        images = sorted(
            [p.name for p in vis_dir.glob("*.png")]
        )

    if not images:
        body = '<p class="metric-neutral">No visualization images found.</p>'
    else:
        gallery = ""
        for img in images:
            title = img.replace("_", " ").replace(".png", "").title()
            gallery += (
                f'<div><h3>{title}</h3>'
                f'<img src="/plots/image/{img}" alt="{img}"></div>'
            )
        body = f'<div class="gallery">{gallery}</div>'

    return page("Visualizations", body)


@app.route("/plots/image/<filename>")
def plot_image(filename: str):
    vis_dir = RUNS_DIR / "visualizations"
    return send_from_directory(str(vis_dir), filename)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Knockout Dashboard")
    print(f"  Runs directory: {RUNS_DIR}")
    print(f"  Serving on http://0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
