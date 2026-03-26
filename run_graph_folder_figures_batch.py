#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image, ImageOps

IDENTITY = "Harry Triantafyllidis"
TRACKS = ["TRACK_ground_truth", "TRACK_partial_observation"]
OUTPUT_ROOT_NAME = "graph_folder_figures_batch"
LOG_FILENAME = "graph_folder_figures_batch_log.txt"
LATEX_FILENAME = "statistics.tex"
DEFAULT_MAX_GRAPHS = "100%"


class ProgressBar:
    def __init__(self, total: int, prefix: str = "TOTAL") -> None:
        self.total = max(int(total), 1)
        self.current = 0
        self.prefix = prefix

    def show(self, note: str = "") -> None:
        self._render(note)

    def advance(self, note: str = "") -> None:
        self.current = min(self.current + 1, self.total)
        self._render(note)

    def _render(self, note: str = "") -> None:
        width = 32
        filled = int(width * self.current / self.total)
        bar = "#" * filled + "-" * (width - filled)
        msg = f"{self.prefix} [{bar}] {self.current}/{self.total}"
        if note:
            msg += f" | {note}"
        print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run graph_folder_figures.py in compare mode for each track and build summary grids.")
    parser.add_argument("--max_graphs", type=str, default=DEFAULT_MAX_GRAPHS,
                        help='Pass-through value for graph_folder_figures.py --max_graphs. Supports counts, percentages like "25%%", or "all". Default: 100%%')
    return parser.parse_args()


def script_root() -> Path:
    return Path(__file__).resolve().parent


def figure_caption_from_png_name(name: str) -> str:
    lname = name.lower()
    if lname.startswith("figure_microgrid_"):
        return "Dataset-level graph summary grid: node and edge counts, density, degree behaviour, edge-weight behaviour, and categorical attribute summaries where available."
    if lname.startswith("figure_distributions_"):
        return "Structural distribution grid across graphs: clustering, transitivity, assortativity, reciprocity, component structure, and distance-related summaries where available."
    if lname.startswith("figure_communities_and_centrality_"):
        return "Community and centrality grid: community count and modularity summaries together with centrality distributions across nodes."
    if lname.startswith("figure_flow_sankey_"):
        return "Aggregated flow Sankey grid showing directed flow between node categories, using edge weights when available and counts otherwise."
    if lname.startswith("figure_timeline_nodes_edges_"):
        return "Timeline grid for graph size over parsed day index, including node and edge trajectories and day-to-day change summaries."
    if lname.startswith("figure_state_percentages_"):
        return "Timeline grid for node-state composition over parsed day index, averaged across graphs that share the same day."
    if lname == "figure_train_vs_test_shift.png":
        return "Histogram-based train-versus-test shift grid for key graph metrics."
    if lname == "figure_train_vs_test_ecdf.png":
        return "ECDF-based train-versus-test shift grid for selected graph metrics."
    if lname == "figure_timeline_diff_test_minus_train.png":
        return "Per-day difference grid between test and train graph-size trajectories."
    return "Automatically generated graph-folder summary figure."


def collect_pngs(out_dir: Path) -> List[Path]:
    priority_prefixes = [
        "figure_microgrid_",
        "figure_distributions_",
        "figure_communities_and_centrality_",
        "figure_flow_sankey_",
        "figure_timeline_nodes_edges_",
        "figure_state_percentages_",
        "figure_train_vs_test_shift",
        "figure_train_vs_test_ecdf",
        "figure_timeline_diff_test_minus_train",
    ]
    pngs = sorted([p for p in out_dir.glob("*.png") if p.is_file()], key=lambda p: p.name.lower())
    def key(p: Path) -> tuple[int, str]:
        name = p.name.lower()
        for i, prefix in enumerate(priority_prefixes):
            if name.startswith(prefix):
                return (i, name)
        return (999, name)
    return sorted(pngs, key=key)


def make_summary_grid(pngs: List[Path], out_path: Path, title: str) -> None:
    if not pngs:
        return
    images = [Image.open(p).convert("RGB") for p in pngs]
    try:
        n = len(images)
        cols = 2 if n > 1 else 1
        rows = math.ceil(n / cols)
        cell_w, cell_h = 1800, 1200
        margin = 80
        header_h = 160
        grid_w = cols * cell_w + (cols + 1) * margin
        grid_h = rows * cell_h + (rows + 1) * margin + header_h
        canvas = Image.new("RGB", (grid_w, grid_h), "white")
        for idx, img in enumerate(images):
            thumb = ImageOps.contain(img, (cell_w, cell_h))
            row, col = divmod(idx, cols)
            x = margin + col * (cell_w + margin) + (cell_w - thumb.width) // 2
            y = header_h + margin + row * (cell_h + margin) + (cell_h - thumb.height) // 2
            canvas.paste(thumb, (x, y))
        canvas.save(out_path, dpi=(600, 600))
    finally:
        for img in images:
            img.close()


def _reader(pipe, track_name: str, q: queue.Queue, log_lines: List[str]) -> None:
    try:
        for line in iter(pipe.readline, ''):
            text = line.rstrip()
            if text:
                print(f"[{track_name}] {text}", flush=True)
                q.put(text)
                log_lines.append(f"[{track_name}] {text}")
    finally:
        pipe.close()


def run_one_track(track_dir: Path, max_graphs: str, per_track_workers: int, log_lines: List[str]) -> Dict[str, Any]:
    base_dir = script_root()
    driver_script = base_dir / "graph_folder_figures.py"
    track_name = track_dir.name
    train_root = track_dir / "kept_graphml" / "step4_baseline_train"
    test_root = track_dir / "kept_graphml" / "frozen_test_once"
    out_root = base_dir / OUTPUT_ROOT_NAME / track_name / "baseline_train_vs_frozen_test"
    out_root.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "track": track_name,
        "status": "failed",
        "out_root": out_root,
        "summary_grid": None,
        "note": "",
    }
    if not train_root.exists() or not test_root.exists():
        result["note"] = "Missing pooled kept_graphml train/test roots"
        return result

    cmd = [
        sys.executable,
        str(driver_script),
        "--graph_dir", str(train_root),
        "--compare_dir", str(test_root),
        "--out_dir", str(out_root),
        "--identity", IDENTITY,
        "--title", f"{track_name} baseline train vs frozen test",
        "--label", "train",
        "--compare_label", "test",
        "--workers", str(per_track_workers),
        "--max_graphs", str(max_graphs),
    ]
    log_lines.append(f"=== TRACK {track_name} ===")
    log_lines.append(f"RUN: {' '.join(cmd)}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    q: queue.Queue = queue.Queue()
    t = threading.Thread(target=_reader, args=(proc.stdout, track_name, q, log_lines), daemon=True)
    t.start()
    rc = proc.wait()
    t.join()

    if rc != 0:
        result["note"] = f"graph_folder_figures.py exited with code {rc}"
        return result

    pngs = collect_pngs(out_root)
    summary = (base_dir / OUTPUT_ROOT_NAME / track_name / f"{track_name}__baseline_train_vs_frozen_test__summary_grid.png")
    make_summary_grid(pngs, summary, track_name)
    result["status"] = "ran"
    result["summary_grid"] = summary
    result["note"] = f"Generated {len(pngs)} PNG(s)"
    return result


def write_statistics_tex(base_dir: Path, results: List[Dict[str, Any]]) -> Path:
    out_path = base_dir / LATEX_FILENAME
    lines: List[str] = []
    lines.append("% Auto-generated by run_graph_folder_figures_batch.py")
    lines.append("% Suggested preamble: \\usepackage{graphicx}")
    lines.append("")
    for res in results:
        lines.append("\\clearpage")
        lines.append("\\begin{figure}[p]")
        lines.append("  \\centering")
        if res["status"] == "ran" and res["summary_grid"] is not None:
            rel = Path(res["summary_grid"]).relative_to(base_dir).as_posix()
            lines.append(f"  \\includegraphics[width=0.98\\textwidth,height=0.93\\textheight,keepaspectratio]{{{rel}}}")
            lines.append(f"  \\caption{{{res['track'].replace('_', ' ')} baseline train vs frozen test summary grid.}}")
        else:
            lines.append(f"  % Track failed: {res['track']} | {res['note']}")
            lines.append("  \\caption{Track run failed; see graph\\_folder\\_figures\\_batch\\_log.txt for details.}")
        lines.append("\\end{figure}")
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    args = parse_args()
    base_dir = script_root()
    output_root = base_dir / OUTPUT_ROOT_NAME
    output_root.mkdir(parents=True, exist_ok=True)
    results_root = base_dir / "experiments_results"
    log_path = base_dir / LOG_FILENAME

    track_dirs = [results_root / t for t in TRACKS if (results_root / t).exists()]
    if not track_dirs:
        print(f"ERROR: no track directories found under {results_root}", file=sys.stderr)
        return 1

    total_cpus = os.cpu_count() or 1
    per_track_workers = max(1, (total_cpus - 1) // 2)
    log_lines: List[str] = [
        f"Repository root: {base_dir}",
        f"Python executable: {sys.executable}",
        f"Tracks found: {', '.join(p.name for p in track_dirs)}",
        "Parallel track jobs: enabled",
        "",
    ]

    progress = ProgressBar(total=len(track_dirs), prefix="TOTAL")
    progress.show(f"starting | workers/track={per_track_workers} | max_graphs={args.max_graphs}")

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(run_one_track, track_dir, args.max_graphs, per_track_workers, log_lines): track_dir.name for track_dir in track_dirs}
        for fut in as_completed(futs):
            track_name = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"track": track_name, "status": "failed", "summary_grid": None, "note": str(e)}
            results.append(res)
            progress.advance(f"{track_name} | {res['status']}")

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    tex_path = write_statistics_tex(base_dir, sorted(results, key=lambda r: r["track"]))
    print(f"Wrote log: {log_path}")
    print(f"Wrote LaTeX: {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
