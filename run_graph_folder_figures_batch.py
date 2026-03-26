#!/usr/bin/env python3
"""
run_graph_folder_figures_batch.py

Batch runner for graph_folder_figures.py.

This version expects the pooled baseline GraphML layout under each track:

    experiments_results/TRACK_.../kept_graphml/step4_baseline_train/
    experiments_results/TRACK_.../kept_graphml/frozen_test_once/

It runs graph_folder_figures.py once per track in compare mode:
- --graph_dir   = pooled baseline train GraphML root
- --compare_dir = pooled frozen test GraphML root

Then it builds one 600 DPI summary grid per track from the generated PNGs and
writes statistics.tex containing only those two high-level summary figures.

The two track-level compare jobs are executed in parallel.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageOps, ImageFile, UnidentifiedImageError, ImageDraw

ImageFile.LOAD_TRUNCATED_IMAGES = True

IDENTITY = "Harry Triantafyllidis"
TRACK_PREFIX = "TRACK_"
KEPT_GRAPHML_SUBDIR = Path("kept_graphml")
TRAIN_ROOT_NAME = "step4_baseline_train"
TEST_ROOT_NAME = "frozen_test_once"
OUTPUT_ROOT_NAME = "graph_folder_figures_batch"
LOG_FILENAME = "graph_folder_figures_batch_log.txt"
LATEX_FILENAME = "statistics.tex"
SUMMARY_SUFFIX = "baseline_train_vs_frozen_test__summary_grid.png"
SUMMARY_DPI = 600
MAX_WORKERS = 2


def compute_per_track_workers() -> int:
    total_cpus = os.cpu_count() or 1
    return max(1, (total_cpus - 1) // 2)



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


@dataclass
class TrackRecord:
    track: str
    train_dir: Path
    test_dir: Path
    out_dir: Path
    status: str
    pngs: List[Path] = field(default_factory=list)
    summary_grid: Optional[Path] = None
    note: str = ""
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


def script_root() -> Path:
    return Path(__file__).resolve().parent


def safe_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


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


def find_track_dirs(results_root: Path) -> List[Path]:
    return sorted(
        [p for p in results_root.iterdir() if p.is_dir() and p.name.startswith(TRACK_PREFIX)],
        key=lambda p: p.name,
    )


def has_graphml(folder: Path) -> bool:
    if not folder.exists():
        return False
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(".graphml"):
                return True
    return False


def collect_pngs(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.rglob("*.png") if p.is_file() and p.name.lower() != ".ds_store"],
        key=lambda p: p.as_posix().lower(),
    )


def desired_png_order(pngs: Sequence[Path]) -> List[Path]:
    preferred = [
        "figure_microgrid_",
        "figure_distributions_",
        "figure_communities_and_centrality_",
        "figure_flow_sankey_",
        "figure_timeline_nodes_edges_",
        "figure_state_percentages_",
        "figure_train_vs_test_shift.png",
        "figure_train_vs_test_ecdf.png",
        "figure_timeline_diff_test_minus_train.png",
    ]

    def key_fn(path: Path) -> Tuple[int, str]:
        lname = path.name.lower()
        for idx, token in enumerate(preferred):
            if lname == token or lname.startswith(token):
                return idx, lname
        return len(preferred), lname

    return sorted(pngs, key=key_fn)


def _open_image_rgb(path: Path) -> Optional[Image.Image]:
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def build_summary_grid(track: str, pngs: Sequence[Path], out_path: Path, dpi: int = SUMMARY_DPI) -> Optional[Path]:
    ordered = desired_png_order(pngs)
    images: List[Tuple[Path, Image.Image]] = []
    for png in ordered:
        img = _open_image_rgb(png)
        if img is not None:
            images.append((png, img))

    if not images:
        return None

    n = len(images)
    cols = 2 if n <= 4 else 3
    rows = math.ceil(n / cols)

    tile_w = 2200
    tile_h = 1600
    pad = 80
    title_h = 150
    label_h = 110

    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = rows * (title_h + tile_h + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, (src_path, img) in enumerate(images):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * tile_w
        y0 = pad + r * (title_h + tile_h + label_h)

        content_box = (tile_w - pad, tile_h - pad)
        fitted = ImageOps.contain(img, content_box)
        inner_x = x0 + (tile_w - fitted.width) // 2
        inner_y = y0 + title_h + (tile_h - fitted.height) // 2

        # panel frame
        draw.rectangle(
            [x0, y0 + title_h, x0 + tile_w, y0 + title_h + tile_h],
            outline="black",
            width=3,
        )
        canvas.paste(fitted, (inner_x, inner_y))

        title = src_path.stem.replace("_", " ")
        label_y = y0 + title_h + tile_h + 20
        draw.text((x0 + 10, y0 + 20), title, fill="black")
        draw.text((x0 + 10, label_y), src_path.name, fill="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(dpi, dpi))
    return out_path


def write_log(log_path: Path, log_lines: Sequence[str]) -> None:
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def write_statistics_tex(base_dir: Path, records: Sequence[TrackRecord]) -> Path:
    out_path = base_dir / LATEX_FILENAME
    lines: List[str] = []
    lines.append("% Auto-generated by run_graph_folder_figures_batch.py")
    lines.append("% Suggested preamble:")
    lines.append("% \\usepackage{graphicx}")
    lines.append("% \\usepackage{float}")
    lines.append("% \\usepackage{placeins}")
    lines.append("")

    for rec in sorted(records, key=lambda x: x.track):
        lines.append(f"% ==================== {rec.track} ====================")
        lines.append(f"\\section*{{{latex_escape(rec.track.replace('_', ' '))}}}")
        lines.append("")
        if rec.status != "ran" or rec.summary_grid is None:
            reason = rec.note if rec.note else rec.status
            lines.append("\\begin{quote}")
            lines.append(f"Skipped: {latex_escape(reason)}.")
            lines.append("\\end{quote}")
            lines.append("")
            continue

        rel_png = rec.summary_grid.relative_to(base_dir).as_posix()
        label = safe_tag(f"{rec.track}_{rec.summary_grid.stem}").lower()
        caption = (
            "High-level summary grid combining all graph-folder figures for the pooled baseline train set, "
            "the pooled frozen test set, and their train-versus-test comparisons."
        )
        lines.append("\\clearpage")
        lines.append("\\begin{figure}[p]")
        lines.append("  \\centering")
        lines.append(
            f"  \\includegraphics[width=0.98\\textwidth,height=0.95\\textheight,keepaspectratio]{{{latex_escape(rel_png)}}}"
        )
        lines.append(f"  \\caption{{{latex_escape(caption)}}}")
        lines.append(f"  \\label{{fig:{latex_escape(label)}}}")
        lines.append("\\end{figure}")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def run_graph_folder_figures_compare(
    python_exe: str,
    driver_script: Path,
    train_dir: Path,
    test_dir: Path,
    out_dir: Path,
    label: str,
    stream_prefix: str,
    workers: int,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        python_exe,
        str(driver_script),
        "--graph_dir",
        str(train_dir),
        "--compare_dir",
        str(test_dir),
        "--out_dir",
        str(out_dir),
        "--workers",
        str(max(1, int(workers))),
        "--identity",
        IDENTITY,
        "--title",
        label,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    captured_lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        captured_lines.append(line)
        text = line.rstrip("\n")
        if text:
            print(f"[{stream_prefix}] {text}", flush=True)
        else:
            print("", flush=True)

    proc.wait()
    combined = "".join(captured_lines)
    return subprocess.CompletedProcess(cmd, proc.returncode, combined, "")


def process_track(base_dir: Path, track_dir: Path, python_exe: str, driver_script: Path, output_root: Path, per_track_workers: int) -> TrackRecord:
    track_name = track_dir.name
    kept_root = track_dir / KEPT_GRAPHML_SUBDIR
    train_dir = kept_root / TRAIN_ROOT_NAME
    test_dir = kept_root / TEST_ROOT_NAME
    out_dir = output_root / track_name / "baseline_train_vs_frozen_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_dir.exists() or not test_dir.exists():
        missing = []
        if not train_dir.exists():
            missing.append(str(train_dir))
        if not test_dir.exists():
            missing.append(str(test_dir))
        return TrackRecord(
            track=track_name,
            train_dir=train_dir,
            test_dir=test_dir,
            out_dir=out_dir,
            status="missing_source",
            note="Missing required pooled kept_graphml root(s): " + "; ".join(missing),
        )

    if not has_graphml(train_dir):
        return TrackRecord(
            track=track_name,
            train_dir=train_dir,
            test_dir=test_dir,
            out_dir=out_dir,
            status="skipped_no_graphml",
            note="No .graphml files found under pooled baseline train root",
        )

    if not has_graphml(test_dir):
        return TrackRecord(
            track=track_name,
            train_dir=train_dir,
            test_dir=test_dir,
            out_dir=out_dir,
            status="skipped_no_graphml",
            note="No .graphml files found under pooled frozen test root",
        )

    proc = run_graph_folder_figures_compare(
        python_exe=python_exe,
        driver_script=driver_script,
        train_dir=train_dir,
        test_dir=test_dir,
        out_dir=out_dir,
        label=f"{track_name} baseline train vs frozen test",
        stream_prefix=track_name,
        workers=per_track_workers,
    )

    record = TrackRecord(
        track=track_name,
        train_dir=train_dir,
        test_dir=test_dir,
        out_dir=out_dir,
        status="failed" if proc.returncode != 0 else "ran",
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        returncode=proc.returncode,
    )

    if proc.returncode != 0:
        record.note = f"graph_folder_figures.py exited with code {proc.returncode}"
        return record

    pngs = collect_pngs(out_dir)
    record.pngs = pngs
    if not pngs:
        record.status = "failed"
        record.note = "graph_folder_figures.py completed but no PNG outputs were found"
        return record

    summary_name = f"{safe_tag(track_name.lower())}__{SUMMARY_SUFFIX}"
    summary_path = output_root / track_name / summary_name
    built = build_summary_grid(track_name, pngs, summary_path, dpi=SUMMARY_DPI)
    if built is None:
        record.status = "failed"
        record.note = "Could not build summary grid from generated PNG outputs"
        return record

    record.summary_grid = built
    record.note = f"Generated {len(pngs)} PNG figure(s) and 1 summary grid"
    return record


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
    track_dirs = find_track_dirs(results_root)
    if not track_dirs:
        print(f"ERROR: No track folders found under {results_root}", file=sys.stderr)
        return 1

    python_exe = sys.executable
    log_lines: List[str] = []
    log_lines.append(f"Repository root: {base_dir}")
    log_lines.append(f"Python executable: {python_exe}")
    log_lines.append(f"Tracks found: {', '.join(p.name for p in track_dirs)}")
    log_lines.append("Parallel track jobs: enabled")
    per_track_workers = compute_per_track_workers()
    log_lines.append(f"Per-track graph_folder_figures workers: {per_track_workers}")
    log_lines.append("")

    progress = ProgressBar(total=len(track_dirs), prefix="TOTAL")
    progress.show(f"starting | workers/track={per_track_workers}")

    records: List[TrackRecord] = []
    future_to_track: Dict[object, str] = {}

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(track_dirs)))) as pool:
        for track_dir in track_dirs:
            future = pool.submit(process_track, base_dir, track_dir, python_exe, driver_script, output_root, per_track_workers)
            future_to_track[future] = track_dir.name

        for future in as_completed(future_to_track):
            track_name = future_to_track[future]
            try:
                rec = future.result()
            except Exception as exc:  # pragma: no cover
                rec = TrackRecord(
                    track=track_name,
                    train_dir=results_root / track_name / KEPT_GRAPHML_SUBDIR / TRAIN_ROOT_NAME,
                    test_dir=results_root / track_name / KEPT_GRAPHML_SUBDIR / TEST_ROOT_NAME,
                    out_dir=output_root / track_name / "baseline_train_vs_frozen_test",
                    status="failed",
                    note=f"Unhandled exception: {exc}",
                )
            records.append(rec)

            log_lines.append(f"=== TRACK {rec.track} ===")
            log_lines.append(f"Train root: {rec.train_dir}")
            log_lines.append(f"Test root: {rec.test_dir}")
            log_lines.append(f"Output root: {rec.out_dir}")
            log_lines.append(f"Status: {rec.status}")
            if rec.note:
                log_lines.append(f"Note: {rec.note}")
            if rec.stdout:
                log_lines.append("STDOUT:")
                log_lines.append(rec.stdout.rstrip())
            if rec.stderr:
                log_lines.append("STDERR:")
                log_lines.append(rec.stderr.rstrip())
            log_lines.append("")

            note = f"{rec.track} | {rec.status}"
            progress.advance(note)

    write_log(base_dir / LOG_FILENAME, log_lines)
    tex_path = write_statistics_tex(base_dir, records)

    print(f"Wrote log: {base_dir / LOG_FILENAME}")
    print(f"Wrote LaTeX: {tex_path}")
    for rec in sorted(records, key=lambda x: x.track):
        if rec.summary_grid is not None:
            print(f"Summary grid [{rec.track}]: {rec.summary_grid}")

    return 0 if all(rec.status == "ran" for rec in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
