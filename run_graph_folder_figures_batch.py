#!/usr/bin/env python3
"""
run_graph_folder_figures_batch.py

Batch runner for graph_folder_figures.py.

Compatible with the current experiments_pb.py layout, including both:
1. selective GraphML retention via --keep_step_train_graphml, and
2. full GraphML retention via --keep_graphml.

Assumptions
-----------
- This script resides in the repository root.
- In the same directory there is:
    - graph_folder_figures.py
    - experiments_results/
- experiments_results contains track folders such as:
    TRACK_ground_truth/
    TRACK_partial_observation/

What this script does
---------------------
1. For each track, it first looks under:
      TRACK_.../kept_graphml/
   which is the layout produced by experiments_pb.py when
   --keep_step_train_graphml is used.
2. It discovers dataset-level GraphML roots for preserved train datasets and
   runs graph_folder_figures.py on each one.
3. It runs graph_folder_figures.py once on the preserved frozen test GraphML
   for each track.
4. If kept_graphml is absent or incomplete, it falls back to older / broader
   locations such as archived train_folder folders under:
      work/repro_artifacts_steps_1_7/
   and the older frozen test location:
      work/synthetic_amr_graphs_test_frozen/
5. It writes:
   - graph_folder_figures_batch_log.txt
   - statistics.txt

statistics.txt contains LaTeX code with one figure per page, using the PNGs
produced by graph_folder_figures.py.

Important
---------
- With --keep_step_train_graphml, the expected preserved raw GraphML lives
  under each track's kept_graphml/ folder.
- With --keep_graphml, additional raw GraphML may still exist in working or
  archived folders; this script can fall back to those locations.
- If a candidate folder does not contain usable GraphML, it is skipped and the
  reason is recorded explicitly in the log and in statistics.txt.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


IDENTITY = "Harry Triantafyllidis"
TRACK_PREFIX = "TRACK_"
REPRO_SUBDIR = Path("work") / "repro_artifacts_steps_1_7"
LEGACY_FROZEN_TEST_SUBDIR = Path("work") / "synthetic_amr_graphs_test_frozen"
KEPT_GRAPHML_SUBDIR = Path("kept_graphml")
OUTPUT_ROOT_NAME = "graph_folder_figures_batch"
LOG_FILENAME = "graph_folder_figures_batch_log.txt"
LATEX_FILENAME = "statistics.txt"


@dataclass
class RunRecord:
    track: str
    kind: str  # train | frozen_test
    source_dir: Path
    out_dir: Path
    status: str  # ran | skipped_no_graphml | missing_source | failed
    pngs: List[Path] = field(default_factory=list)
    note: str = ""


def script_root() -> Path:
    return Path(__file__).resolve().parent


def has_graphml(folder: Path) -> bool:
    return any(folder.rglob("*.graphml")) if folder.exists() else False


def has_direct_graphml(folder: Path) -> bool:
    return any(folder.glob("*.graphml")) if folder.exists() else False


def safe_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def find_track_dirs(results_root: Path) -> List[Path]:
    return sorted(
        [p for p in results_root.iterdir() if p.is_dir() and p.name.startswith(TRACK_PREFIX)],
        key=lambda p: p.name,
    )


def _is_dataset_graph_root(folder: Path) -> bool:
    """
    Dataset-level root heuristic aligned with experiments_pb.py.

    Accepted shapes:
    - a folder containing sim_* subfolders with GraphML beneath it
    - a non-sim_* folder containing GraphML files directly
    """
    if not folder.exists() or not folder.is_dir():
        return False
    if not has_graphml(folder):
        return False

    try:
        child_dirs = [p for p in folder.iterdir() if p.is_dir()]
    except FileNotFoundError:
        return False

    if any(p.name.startswith("sim_") for p in child_dirs):
        return True

    if has_direct_graphml(folder) and not folder.name.startswith("sim_"):
        return True

    return False


def _discover_dataset_graph_roots(search_root: Path) -> List[Path]:
    candidates: List[Path] = []

    if _is_dataset_graph_root(search_root):
        candidates.append(search_root)

    for p in search_root.rglob("*"):
        if p.is_dir() and _is_dataset_graph_root(p):
            candidates.append(p)

    uniq = sorted(set(candidates), key=lambda x: (-len(x.parts), str(x)))
    kept: List[Path] = []
    for p in uniq:
        if any(k in p.parents for k in kept):
            continue
        kept.append(p)

    return sorted(kept, key=lambda x: x.as_posix())


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def find_legacy_train_folders(repro_root: Path) -> List[Path]:
    return sorted(
        [p for p in repro_root.rglob("train_folder") if p.is_dir()],
        key=lambda p: p.as_posix(),
    )


def find_preserved_train_datasets(track_dir: Path) -> List[Path]:
    kept_root = track_dir / KEPT_GRAPHML_SUBDIR
    discovered: List[Path] = []

    if kept_root.exists() and kept_root.is_dir():
        for child in sorted([p for p in kept_root.iterdir() if p.is_dir()], key=lambda p: p.name):
            if child.name == "frozen_test_once":
                continue
            discovered.extend(_discover_dataset_graph_roots(child))

    repro_root = track_dir / REPRO_SUBDIR
    if repro_root.exists() and repro_root.is_dir():
        for train_folder in find_legacy_train_folders(repro_root):
            if has_graphml(train_folder):
                discovered.extend(_discover_dataset_graph_roots(train_folder))

    return _dedupe_paths(sorted(discovered, key=lambda p: p.as_posix()))


def find_frozen_test_dataset(track_dir: Path) -> Tuple[Path, str]:
    kept_candidate = track_dir / KEPT_GRAPHML_SUBDIR / "frozen_test_once"
    if kept_candidate.exists() and has_graphml(kept_candidate):
        return kept_candidate, "kept_graphml"

    legacy_candidate = track_dir / LEGACY_FROZEN_TEST_SUBDIR
    if legacy_candidate.exists() and has_graphml(legacy_candidate):
        return legacy_candidate, "legacy_workdir"

    return kept_candidate, "missing"


def label_from_relative_path(path: Path) -> str:
    parts = list(path.parts)
    if not parts:
        return "dataset"
    return safe_tag("__".join(parts))


def label_for_train_dataset(track_dir: Path, dataset_dir: Path) -> str:
    try:
        rel = dataset_dir.relative_to(track_dir)
    except ValueError:
        rel = dataset_dir
    return label_from_relative_path(rel)


def run_graph_folder_figures(
    python_exe: str,
    driver_script: Path,
    graph_dir: Path,
    out_dir: Path,
    label: str,
    log_lines: List[str],
) -> subprocess.CompletedProcess[str]:
    cmd = [
        python_exe,
        str(driver_script),
        "--graph_dir",
        str(graph_dir),
        "--out_dir",
        str(out_dir),
        "--identity",
        IDENTITY,
        "--label",
        label,
        "--title",
        label.replace("__", " | "),
    ]

    log_lines.append(f"RUN: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


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

    def order_key(p: Path) -> tuple[int, str]:
        name = p.name.lower()
        for i, prefix in enumerate(priority_prefixes):
            if name.startswith(prefix):
                return (i, name)
        return (999, name)

    return sorted(pngs, key=order_key)


def write_log(log_path: Path, log_lines: Sequence[str]) -> None:
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


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


def write_statistics_txt(base_dir: Path, records: Sequence[RunRecord]) -> Path:
    out_path = base_dir / LATEX_FILENAME
    lines: List[str] = []

    lines.append("% Auto-generated by run_graph_folder_figures_batch.py")
    lines.append("% Suggested preamble:")
    lines.append("% \\usepackage{graphicx}")
    lines.append("% \\usepackage{float}")
    lines.append("% \\usepackage{placeins}")
    lines.append("")
    lines.append("% This file places one PNG per page to keep layouts clean.")
    lines.append("% Paths are written relative to the directory where this script resides.")
    lines.append("")

    grouped = {}
    for rec in records:
        grouped.setdefault(rec.track, []).append(rec)

    for track in sorted(grouped):
        lines.append(f"% ==================== {track} ====================")
        lines.append(f"\\section*{{{latex_escape(track.replace('_', ' '))}}}")
        lines.append("")

        for rec in grouped[track]:
            rel_source = rec.source_dir.relative_to(base_dir).as_posix() if rec.source_dir.exists() else rec.source_dir.as_posix()
            heading = f"{rec.kind}: {rel_source}"
            lines.append(f"\\subsection*{{{latex_escape(heading)}}}")
            lines.append("")

            if rec.status != "ran":
                reason = rec.note if rec.note else rec.status
                lines.append("\\begin{quote}")
                lines.append(
                    f"Skipped: {latex_escape(reason)}. graph\\_folder\\_figures.py requires GraphML input, and this dataset snapshot was not runnable in its archived form."
                )
                lines.append("\\end{quote}")
                lines.append("")
                continue

            for png in rec.pngs:
                rel_png = png.relative_to(base_dir).as_posix()
                caption = figure_caption_from_png_name(png.name)
                label = safe_tag(f"{track}_{rec.kind}_{png.stem}").lower()

                lines.append("\\clearpage")
                lines.append("\\begin{figure}[p]")
                lines.append("  \\centering")
                lines.append(f"  \\includegraphics[width=0.97\\textwidth,height=0.92\\textheight,keepaspectratio]{{{latex_escape(rel_png)}}}")
                lines.append(
                    f"  \\caption{{\\textbf{{{latex_escape(png.stem.replace('_', ' '))}}}. {latex_escape(caption)}}}"
                )
                lines.append(f"  \\label{{fig:{latex_escape(label)}}}")
                lines.append("\\end{figure}")
                lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    base_dir = script_root()
    results_root = base_dir / "experiments_results"
    driver_script = base_dir / "graph_folder_figures.py"
    output_root = base_dir / OUTPUT_ROOT_NAME

    if not results_root.exists():
        print(f"ERROR: Missing folder: {results_root}", file=sys.stderr)
        return 1
    if not driver_script.exists():
        print(f"ERROR: Missing script: {driver_script}", file=sys.stderr)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []
    records: List[RunRecord] = []

    track_dirs = find_track_dirs(results_root)
    if not track_dirs:
        print(f"ERROR: No track folders found under {results_root}", file=sys.stderr)
        return 1

    python_exe = sys.executable
    log_lines.append(f"Repository root: {base_dir}")
    log_lines.append(f"Python executable: {python_exe}")
    log_lines.append(f"Tracks found: {', '.join(p.name for p in track_dirs)}")
    log_lines.append("")

    for track_dir in track_dirs:
        track_name = track_dir.name
        kept_root = track_dir / KEPT_GRAPHML_SUBDIR
        repro_root = track_dir / REPRO_SUBDIR
        track_output_root = output_root / track_name
        track_output_root.mkdir(parents=True, exist_ok=True)

        log_lines.append(f"=== TRACK {track_name} ===")
        log_lines.append(f"kept_graphml root: {kept_root}")
        log_lines.append(f"repro root: {repro_root}")

        train_datasets = find_preserved_train_datasets(track_dir)
        log_lines.append(f"Discovered {len(train_datasets)} candidate train GraphML dataset roots")

        if not train_datasets:
            records.append(
                RunRecord(
                    track=track_name,
                    kind="train",
                    source_dir=track_dir / KEPT_GRAPHML_SUBDIR,
                    out_dir=track_output_root,
                    status="missing_source",
                    note="No preserved train GraphML dataset roots were found under kept_graphml or legacy archive locations",
                )
            )
            log_lines.append("MISSING: no train GraphML dataset roots found")
        else:
            for dataset_dir in train_datasets:
                label = label_for_train_dataset(track_dir, dataset_dir)
                out_dir = track_output_root / label
                out_dir.mkdir(parents=True, exist_ok=True)

                if not has_graphml(dataset_dir):
                    records.append(
                        RunRecord(
                            track=track_name,
                            kind="train",
                            source_dir=dataset_dir,
                            out_dir=out_dir,
                            status="skipped_no_graphml",
                            note="No .graphml files found in discovered train dataset root",
                        )
                    )
                    log_lines.append(f"SKIP (no GraphML): {dataset_dir}")
                    continue

                proc = run_graph_folder_figures(
                    python_exe=python_exe,
                    driver_script=driver_script,
                    graph_dir=dataset_dir,
                    out_dir=out_dir,
                    label=label,
                    log_lines=log_lines,
                )

                if proc.stdout:
                    log_lines.append("STDOUT:")
                    log_lines.append(proc.stdout.rstrip())
                if proc.stderr:
                    log_lines.append("STDERR:")
                    log_lines.append(proc.stderr.rstrip())

                if proc.returncode != 0:
                    records.append(
                        RunRecord(
                            track=track_name,
                            kind="train",
                            source_dir=dataset_dir,
                            out_dir=out_dir,
                            status="failed",
                            note=f"graph_folder_figures.py exited with code {proc.returncode}",
                        )
                    )
                    log_lines.append(f"FAILED [{proc.returncode}]: {dataset_dir}")
                    continue

                pngs = collect_pngs(out_dir)
                records.append(
                    RunRecord(
                        track=track_name,
                        kind="train",
                        source_dir=dataset_dir,
                        out_dir=out_dir,
                        status="ran",
                        pngs=pngs,
                        note=f"Generated {len(pngs)} PNG figure(s)",
                    )
                )
                log_lines.append(f"OK: {dataset_dir} -> {out_dir} ({len(pngs)} PNGs)")

        frozen_test_dir, frozen_source_mode = find_frozen_test_dataset(track_dir)
        frozen_out_dir = track_output_root / "frozen_test"
        frozen_out_dir.mkdir(parents=True, exist_ok=True)
        log_lines.append(f"Frozen test mode: {frozen_source_mode}")

        if frozen_source_mode == "missing":
            records.append(
                RunRecord(
                    track=track_name,
                    kind="frozen_test",
                    source_dir=frozen_test_dir,
                    out_dir=frozen_out_dir,
                    status="missing_source",
                    note="Missing preserved frozen test GraphML under kept_graphml/frozen_test_once and legacy work/synthetic_amr_graphs_test_frozen",
                )
            )
            log_lines.append(f"MISSING: {frozen_test_dir}")
        elif not has_graphml(frozen_test_dir):
            records.append(
                RunRecord(
                    track=track_name,
                    kind="frozen_test",
                    source_dir=frozen_test_dir,
                    out_dir=frozen_out_dir,
                    status="skipped_no_graphml",
                    note="No .graphml files found in frozen test folder",
                )
            )
            log_lines.append(f"SKIP (no GraphML): {frozen_test_dir}")
        else:
            proc = run_graph_folder_figures(
                python_exe=python_exe,
                driver_script=driver_script,
                graph_dir=frozen_test_dir,
                out_dir=frozen_out_dir,
                label=f"{safe_tag(track_name)}__frozen_test",
                log_lines=log_lines,
            )

            if proc.stdout:
                log_lines.append("STDOUT:")
                log_lines.append(proc.stdout.rstrip())
            if proc.stderr:
                log_lines.append("STDERR:")
                log_lines.append(proc.stderr.rstrip())

            if proc.returncode != 0:
                records.append(
                    RunRecord(
                        track=track_name,
                        kind="frozen_test",
                        source_dir=frozen_test_dir,
                        out_dir=frozen_out_dir,
                        status="failed",
                        note=f"graph_folder_figures.py exited with code {proc.returncode}",
                    )
                )
                log_lines.append(f"FAILED [{proc.returncode}]: {frozen_test_dir}")
            else:
                pngs = collect_pngs(frozen_out_dir)
                records.append(
                    RunRecord(
                        track=track_name,
                        kind="frozen_test",
                        source_dir=frozen_test_dir,
                        out_dir=frozen_out_dir,
                        status="ran",
                        pngs=pngs,
                        note=f"Generated {len(pngs)} PNG figure(s)",
                    )
                )
                log_lines.append(f"OK: {frozen_test_dir} -> {frozen_out_dir} ({len(pngs)} PNGs)")

        log_lines.append("")

    log_path = base_dir / LOG_FILENAME
    stats_path = write_statistics_txt(base_dir=base_dir, records=records)
    write_log(log_path, log_lines)

    print(f"Done. Log written to: {log_path}")
    print(f"LaTeX written to: {stats_path}")
    print(f"Figure outputs written under: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
