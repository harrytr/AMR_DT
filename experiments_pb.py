#!/usr/bin/env python3
"""
experiments_pb.py


python prune_overleaf_package.py --delete-unreferenced-files
python experiments_pb.py --run_both_state_modes --no_train --emit_latex --run_all_T


Experiment orchestration pipeline for the AMR digital-twin prediction framework.

This script coordinates multi-step experimental runs across one or both state
modes, including data generation, graph conversion, model training, evaluation,
figure production, and LaTeX export. It supports running complete pipelines from
scratch as well as resuming from intermediate steps.

Main capabilities
- Runs predefined experiment stages over canonical, ablation, observation-shift,
  and epidemiological-shift settings.
- Supports single-track or dual-track execution across state modes.
- Preserves and reuses a frozen external test benchmark where required by the
  experimental design.
- Optionally archives train/test folders and selected intermediate artefacts.
- Produces structured figures, summary outputs, and LaTeX-ready report material.

Execution control
- Supports step-wise execution with configurable start and stop points.
- Supports repeated runs across multiple temporal horizons.
- Supports optional emission of LaTeX summaries and export bundles for reporting.
- Supports controlled retention of GraphML artefacts for downstream visualisation
  and figure generation.

Outputs
Depending on the selected options, the script can generate:
- training and evaluation artefacts
- archived train/test graph folders
- figures and comparison panels
- metrics summaries
- LaTeX tables and figure blocks
- reproducibility-oriented experiment folders

Examples

1) Run both state modes from scratch through Step 7:
   python experiments_pb.py --run_both_state_modes --emit_latex --run_all_T --archive_train_test_folders

2) Resume both state modes from Step 6.2 through Step 7:
   python experiments_pb.py --run_both_state_modes --start 6.2 --stop 7 --emit_latex --run_all_T

3) Run only the ground-truth track from Step 6.2 through Step 7:

   Windows Command Prompt (cmd.exe):
   set DT_STATE_MODE=ground_truth
   python experiments_pb.py --start 6.2 --stop 7 --emit_latex --run_all_T

   PowerShell:
   $env:DT_STATE_MODE="ground_truth"
   python .\experiments_pb.py --start 6.2 --stop 7 --emit_latex --run_all_T

4) Run only the partial-observation track from Step 6.2 through Step 7:

   Windows Command Prompt (cmd.exe):
   set DT_STATE_MODE=partial_observation
   python experiments_pb.py --start 6.2 --stop 7 --emit_latex --run_all_T

   PowerShell:
   $env:DT_STATE_MODE="partial_observation"
   python .\experiments_pb.py --start 6.2 --stop 7 --emit_latex --run_all_T
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable
# =============================================================================
# CONFIG (EDIT HERE)
# =============================================================================
CONFIG: Dict[str, Any] = {
    "SIM": {
        "num_regions": 1,
        "num_wards": 15,
        "num_patients": 300,
        "num_staff": 900,
        "export_yaml": True,
        "export_gif": False,
        "override_staff_wards_per_staff": True,
        "staff_wards_per_staff": 2,
        "enable_superspreader": False,
        "superspreader_staff": "",
        "superspreader_state": "IR",
        "superspreader_staff_contacts": 50,
        "superspreader_patient_frac_mult": 3.0,
        "superspreader_patient_min_add": 10,
        "superspreader_edge_weight_mult": 1.5,
        "superspreader_start_day": 1,
        "superspreader_end_day": 9999,
        "enable_admit_import_seasonality": False,
        "admit_import_seasonality": "none",
        "admit_import_period_days": 7,
        "admit_import_pmax_cs": 1.0,
        "admit_import_pmax_cr": 1.0,
        "admit_import_amp": 0.5,
        "admit_import_phase_day": 0,
        "admit_import_high_start_day": 1,
        "admit_import_high_end_day": 90,
        "admit_import_high_mult": 1.5,
        "admit_import_low_mult": 1.0,
        "admit_import_shock_min_days": 7,
        "admit_import_shock_max_days": 30,
        "admit_import_shock_mult_min": 1.5,
        "admit_import_shock_mult_max": 5.0,
    },
    "CONVERT": {
        "horizons": "7",
        "workers": 8,
    },
    "MODEL": {
        "use_task_hparams": False,
        "train_model": True,
        "task": "endogenous_importation_majority_h7",
        #"task": "early_outbreak_warning_h14",
        "pred_horizon": 7,
        "early_outbreak_fixed_threshold": 0.55,
        "T": 7,
        "T_list": "7",
        "sliding_step": 1,
        "hidden": 32,
        "heads": 2,
        "dropout": 0.2,
        "transformer_layers": 2,
        "sage_layers": 2,
        "use_cls": False,
        "batch_size": 16,
        "epochs": 50,
        "lr": 1e-5,
        "max_neighbors": 20,
        "neighbor_sampling": True,
        "num_neighbors": "15,10",
        "seed_count": 256,
        "seed_strategy": "random",
        "seed_batch_size": 64,
        "max_sub_batches": 4,
        "attn_top_k": 10,
        "attn_rank_by": "abs_diff",
    },
    "TEST": {
        "test_frac_per_class": 0.5,
        "min_per_class": 10,
        "seed": 1337,
        "balance_tolerance": 1,
        "require_balanced_test": False,
    },
    "STEP1": {
        "n_sims_per_trajectory": 10,
        "num_days": 90,
        "outbreak_expected_label_min_frac": 0.3,
        "outbreak_require_two_class_pooled_baseline": True,
        "trajectories": {
            "endog_high_train": {
                "seed_base": 4100,
                "p_admit_import_cs": 0.005,
                "p_admit_import_cr": 0.005,
                "daily_discharge_frac": 0.02,
                "daily_discharge_min_per_ward": 0,
                "extra_sim_args": [],
            },
            "import_high_train": {
                "seed_base": 5100,
                "p_admit_import_cs": 0.6,
                "p_admit_import_cr": 0.6,
                "daily_discharge_frac": 0.25,
                "daily_discharge_min_per_ward": 1,
                "extra_sim_args": [],
            },
            "endog_high_test": {
                "seed_base": 6100,
                "p_admit_import_cs": 0.005,
                "p_admit_import_cr": 0.005,
                "daily_discharge_frac": 0.02,
                "daily_discharge_min_per_ward": 0,
                "extra_sim_args": [],
            },
            "import_high_test": {
                "seed_base": 7100,
                "p_admit_import_cs": 0.6,
                "p_admit_import_cr": 0.6,
                "daily_discharge_frac": 0.25,
                "daily_discharge_min_per_ward": 1,
                "extra_sim_args": [],
            },
        },
    },
}
# =============================================================================

DEFAULT_RESULTS_PARENT = Path("experiments_results")
DEFAULT_OVERLEAF_DIRNAME = "overleaf_package"

TRAINING_OUT_REL = Path("training_outputs")


def _task_family(task_name: str) -> str:
    t = str(task_name).strip().lower()
    if t.startswith("early_outbreak_warning_h"):
        return "early_outbreak_warning"
    if t.startswith("endogenous_importation_") or t.startswith("endogenous_transmission_"):
        return "mechanism_split"
    return "default"


def _build_task_aligned_step1_trajectories(task_name: str, current_step1_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    family = _task_family(task_name)
    num_days = int(current_step1_cfg.get("num_days", 90))

    if family == "early_outbreak_warning":
        return {
            "outbreak_low_train": {
                "seed_base": 4100,
                "p_admit_import_cs": 0.0,
                "p_admit_import_cr": 0.0,
                "daily_discharge_frac": 0.0,
                "daily_discharge_min_per_ward": 0,
                "extra_sim_args": [
                    "--admit_import_seasonality", "none",
                    "--screen_on_admission", "0",
                    "--persist_observations", "1",
                ],
            },
            "outbreak_high_train": {
                "seed_base": 5100,
                "p_admit_import_cs": 0.55,
                "p_admit_import_cr": 0.55,
                "daily_discharge_frac": 0.45,
                "daily_discharge_min_per_ward": 1,
                "extra_sim_args": [
                    "--superspreader_staff", "s0",
                    "--superspreader_state", "IR",
                    "--superspreader_start_day", "1",
                    "--superspreader_end_day", str(num_days),
                    "--superspreader_patient_frac_mult", "8.0",
                    "--superspreader_patient_min_add", "40",
                    "--superspreader_staff_contacts", "180",
                    "--superspreader_edge_weight_mult", "4.0",
                    "--admit_import_seasonality", "piecewise",
                    "--admit_import_period_days", str(num_days),
                    "--admit_import_high_start_day", "1",
                    "--admit_import_high_end_day", str(num_days),
                    "--admit_import_high_mult", "3.0",
                    "--admit_import_low_mult", "1.5",
                    "--admit_import_pmax_cs", "1.0",
                    "--admit_import_pmax_cr", "1.0",
                    "--screen_on_admission", "1",
                    "--persist_observations", "1",
                ],
            },
            "outbreak_low_test": {
                "seed_base": 6100,
                "p_admit_import_cs": 0.0,
                "p_admit_import_cr": 0.0,
                "daily_discharge_frac": 0.0,
                "daily_discharge_min_per_ward": 0,
                "extra_sim_args": [
                    "--admit_import_seasonality", "none",
                    "--screen_on_admission", "0",
                    "--persist_observations", "1",
                ],
            },
            "outbreak_high_test": {
                "seed_base": 7100,
                "p_admit_import_cs": 0.45,
                "p_admit_import_cr": 0.45,
                "daily_discharge_frac": 0.35,
                "daily_discharge_min_per_ward": 1,
                "extra_sim_args": [
                    "--superspreader_staff", "s0",
                    "--superspreader_state", "IR",
                    "--superspreader_start_day", "1",
                    "--superspreader_end_day", str(num_days),
                    "--superspreader_patient_frac_mult", "7.0",
                    "--superspreader_patient_min_add", "35",
                    "--superspreader_staff_contacts", "140",
                    "--superspreader_edge_weight_mult", "3.5",
                    "--admit_import_seasonality", "piecewise",
                    "--admit_import_period_days", str(num_days),
                    "--admit_import_high_start_day", "1",
                    "--admit_import_high_end_day", str(num_days),
                    "--admit_import_high_mult", "2.5",
                    "--admit_import_low_mult", "1.25",
                    "--admit_import_pmax_cs", "1.0",
                    "--admit_import_pmax_cr", "1.0",
                    "--screen_on_admission", "1",
                    "--persist_observations", "1",
                ],
            },
        }

    return {
        "endog_high_train": {
            "seed_base": 4100,
            "p_admit_import_cs": 0.005,
            "p_admit_import_cr": 0.005,
            "daily_discharge_frac": 0.02,
            "daily_discharge_min_per_ward": 0,
            "extra_sim_args": [],
        },
        "import_high_train": {
            "seed_base": 5100,
            "p_admit_import_cs": 0.60,
            "p_admit_import_cr": 0.60,
            "daily_discharge_frac": 0.25,
            "daily_discharge_min_per_ward": 1,
            "extra_sim_args": [],
        },
        "endog_high_test": {
            "seed_base": 6100,
            "p_admit_import_cs": 0.005,
            "p_admit_import_cr": 0.005,
            "daily_discharge_frac": 0.02,
            "daily_discharge_min_per_ward": 0,
            "extra_sim_args": [],
        },
        "import_high_test": {
            "seed_base": 7100,
            "p_admit_import_cs": 0.60,
            "p_admit_import_cr": 0.60,
            "daily_discharge_frac": 0.25,
            "daily_discharge_min_per_ward": 1,
            "extra_sim_args": [],
        },
    }

def _resolve_canonical_trajectory_sets(task_name: str, current_step1_cfg: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str], List[str]]:
    trajectories = _build_task_aligned_step1_trajectories(task_name, current_step1_cfg)
    names = list(trajectories.keys())
    train_names = [name for name in names if name.endswith("_train")]
    test_names = [name for name in names if name.endswith("_test")]
    return trajectories, names, train_names, test_names


CONFIG["STEP1"]["trajectories"], CANONICAL_TRAJECTORY_NAMES, CANONICAL_TRAIN_TRAJECTORIES, CANONICAL_TEST_TRAJECTORIES = _resolve_canonical_trajectory_sets(
    str(CONFIG["MODEL"].get("task", "")).strip(),
    CONFIG["STEP1"],
)

BASELINE_TRAIN_REL = Path("synthetic_amr_graphs_train")
BASELINE_TEST_REL = Path("synthetic_amr_graphs_test_frozen")
LIVE_TEST_REL = Path("synthetic_amr_graphs_test")

STEP5_ABLATION_REL = Path("step5_ablation")


def _step_dataset_globs(task_name: str, stage: str) -> List[str]:
    family = _task_family(task_name)
    stage_key = str(stage).strip().lower()

    if family == "mechanism_split":
        mapping = {
            "delay": [
                "synthetic_endog_import_step6_delay*_pt_flat",
                "synthetic_endog_import_step6_delay*_pt",
            ],
            "frequency": [
                "synthetic_endog_import_step6_freq*_pt_flat",
                "synthetic_endog_import_step6_freq*_pt",
            ],
            "sweep": [
                "synthetic_endog_import_step7c_sweep_pt_flat",
                "synthetic_endog_import_step7c_sweep_pt",
            ],
        }
        return list(mapping.get(stage_key, []))

    if family == "early_outbreak_warning":
        mapping = {
            "delay": [
                "synthetic_outbreak_step6_delay*_pt_flat",
                "synthetic_outbreak_step6_delay*_pt",
                "synthetic_*outbreak*step6_delay*_pt_flat",
                "synthetic_*outbreak*step6_delay*_pt",
                "synthetic_*step6_delay*_pt_flat",
                "synthetic_*step6_delay*_pt",
            ],
            "frequency": [
                "synthetic_outbreak_step6_freq*_pt_flat",
                "synthetic_outbreak_step6_freq*_pt",
                "synthetic_*outbreak*step6_freq*_pt_flat",
                "synthetic_*outbreak*step6_freq*_pt",
                "synthetic_*step6_freq*_pt_flat",
                "synthetic_*step6_freq*_pt",
            ],
            "sweep": [
                "synthetic_outbreak_step7*_sweep*_pt_flat",
                "synthetic_outbreak_step7*_sweep*_pt",
                "synthetic_*outbreak*step7*_sweep*_pt_flat",
                "synthetic_*outbreak*step7*_sweep*_pt",
                "synthetic_*step7*_sweep*_pt_flat",
                "synthetic_*step7*_sweep*_pt",
            ],
        }
        return list(mapping.get(stage_key, []))

    fallback = {
        "delay": ["synthetic_*step6_delay*_pt_flat", "synthetic_*step6_delay*_pt"],
        "frequency": ["synthetic_*step6_freq*_pt_flat", "synthetic_*step6_freq*_pt"],
        "sweep": ["synthetic_*step7*_sweep*_pt_flat", "synthetic_*step7*_sweep*_pt"],
    }
    return list(fallback.get(stage_key, []))


DATASET_FIGURES_REL = Path("_dataset_graph_figures")


# =============================================================================
# Report writer
# =============================================================================

class Report:
    def __init__(self, path: Path, dry: bool):
        self.path = path
        self.dry = bool(dry)
        self._enabled = not self.dry

        self._active_kind: Optional[str] = None
        self._console_last_render_len: int = 0

        self._step_title: Optional[str] = None
        self._step_current: int = 0
        self._step_total: int = 1
        self._step_extra: str = ""
        self._step_started_at: Optional[float] = None

        self._subtask_title: Optional[str] = None
        self._subtask_current: int = 0
        self._subtask_total: int = 1
        self._subtask_extra: str = ""
        self._subtask_started_at: Optional[float] = None

        if self._enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            existed = self.path.exists() and self.path.stat().st_size > 0
            with self.path.open("a", encoding="utf-8") as f:
                if existed:
                    f.write("\n" + "#" * 80 + "\n")
                f.write(f"SESSION START UTC: {_dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}\n")

    def _log(self, line: str = "") -> None:
        if not self._enabled:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(str(line) + "\n")

    def write(self, line: str = "") -> None:
        self._log(str(line))

    def _progress_bar(self, current: int, total: int, width: int = 32) -> str:
        total = max(1, int(total))
        current = max(0, min(int(current), total))
        filled = int(round(width * current / total))
        return "[" + "#" * filled + "-" * (width - filled) + f"] {current}/{total}"

    def _fmt_elapsed(self, started_at: Optional[float]) -> str:
        if started_at is None:
            return "00:00:00"
        elapsed = max(0, int(time.time() - started_at))
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _render_line(self, msg: str, kind: str) -> None:
      is_tty = sys.stdout.isatty()
  
      if not is_tty:
          print(msg, flush=True)
          self._active_kind = None
          self._console_last_render_len = 0
          return
  
      if self._active_kind is not None and self._active_kind != kind:
          sys.stdout.write("\n")
          sys.stdout.flush()
          self._console_last_render_len = 0
  
      clear = "\r\033[2K"
      pad = ""
      if len(msg) < self._console_last_render_len:
          pad = " " * (self._console_last_render_len - len(msg))
  
      sys.stdout.write(clear + msg + pad)
      sys.stdout.flush()
      self._active_kind = kind
      self._console_last_render_len = len(msg)

    def _close_active_line(self) -> None:
        if self._active_kind is not None:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._active_kind = None
            self._console_last_render_len = 0

    def _render_step_line(self) -> None:
        if self._step_title is None:
            return
        msg = (
            f"{self._step_title} "
            f"{self._progress_bar(self._step_current, self._step_total)} "
            f"| {self._fmt_elapsed(self._step_started_at)}"
        )
        if self._step_extra:
            msg += f" | {self._step_extra}"
        self._render_line(msg, kind="step")

    def _render_subtask_line(self) -> None:
        if self._subtask_title is None:
            return
        msg = (
            f"  {self._subtask_title} "
            f"{self._progress_bar(self._subtask_current, self._subtask_total)} "
            f"| {self._fmt_elapsed(self._subtask_started_at)}"
        )
        if self._subtask_extra:
            msg += f" | {self._subtask_extra}"
        self._render_line(msg, kind="subtask")

    def start_step_console(self, title: str, total: int = 1, extra: str = "") -> None:
        self._step_title = str(title).strip()
        self._step_current = 0
        self._step_total = max(1, int(total))
        self._step_extra = str(extra).strip()
        self._step_started_at = time.time()
        self._render_step_line()

    def update_step_console(
        self,
        current: Optional[int] = None,
        total: Optional[int] = None,
        extra: Optional[str] = None,
    ) -> None:
        if self._step_title is None:
            return
        if current is not None:
            self._step_current = max(0, int(current))
        if total is not None:
            self._step_total = max(1, int(total))
        if extra is not None:
            self._step_extra = str(extra).strip()
        self._render_step_line()

    def tick_step_console(self, extra: Optional[str] = None) -> None:
        if extra is not None:
            self._step_extra = str(extra).strip()
        self._render_step_line()

    def finish_step_console(self, force_complete: bool = True) -> None:
        if self._step_title is None:
            return
        if force_complete:
            self._step_current = self._step_total
            self._render_step_line()
        self._close_active_line()
        self._step_title = None
        self._step_current = 0
        self._step_total = 1
        self._step_extra = ""
        self._step_started_at = None

    def start_subtask_console(self, title: str, total: int = 1, extra: str = "") -> None:
        self._subtask_title = str(title).strip()
        self._subtask_current = 0
        self._subtask_total = max(1, int(total))
        self._subtask_extra = str(extra).strip()
        self._subtask_started_at = time.time()
        self._render_subtask_line()

    def update_subtask_console(
        self,
        current: Optional[int] = None,
        total: Optional[int] = None,
        extra: Optional[str] = None,
    ) -> None:
        if self._subtask_title is None:
            return
        if current is not None:
            self._subtask_current = max(0, int(current))
        if total is not None:
            self._subtask_total = max(1, int(total))
        if extra is not None:
            self._subtask_extra = str(extra).strip()
        self._render_subtask_line()

    def tick_subtask_console(self, extra: Optional[str] = None) -> None:
        if self._subtask_title is None:
            return
        if extra is not None:
            self._subtask_extra = str(extra).strip()
        self._render_subtask_line()

    def finish_subtask_console(self, force_complete: bool = True) -> None:
        if self._subtask_title is None:
            return
        if force_complete:
            self._subtask_current = self._subtask_total
            self._render_subtask_line()
        self._close_active_line()
        self._subtask_title = None
        self._subtask_current = 0
        self._subtask_total = 1
        self._subtask_extra = ""
        self._subtask_started_at = None
        if self._step_title is not None:
            self._render_step_line()

    def section(self, title: str, total: Optional[int] = None) -> None:
        title_s = str(title)
        self.write("")
        self.write("=" * 80)
        self.write(title_s)
        self.write("=" * 80)

        if title_s.startswith("STEP "):
            if self._step_title is not None:
                self.finish_step_console(force_complete=True)
            self.start_step_console(title_s, total=1 if total is None else int(total))

    def kv(self, k: str, v: Any) -> None:
        self.write(f"{k}: {v}")

# =============================================================================
# Progress helpers
# =============================================================================

def _progress_bar(current: int, total: int, width: int = 32) -> str:
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    filled = int(round(width * current / total))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {current}/{total}"


def _report_progress(
    report: Report,
    *,
    prefix: str,
    current: int,
    total: int,
    extra: str = "",
    target: str = "step",
) -> None:
    msg = f"{prefix} {_progress_bar(current, total)}"
    if extra:
        msg += f" | {extra}"
    report.write(msg)

    if target == "step":
        report.update_step_console(current=current, total=total, extra=f"{prefix} | {extra}" if extra else prefix)
    elif target == "subtask":
        report.update_subtask_console(current=current, total=total, extra=f"{prefix} | {extra}" if extra else prefix)

def _format_duration_hms(total_seconds: float) -> str:
    secs = max(0, int(round(float(total_seconds))))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _to_work_dir_relative_path(path_value: Path | str, work_dir: Path) -> str:
    path_obj = Path(path_value)
    if not path_obj.is_absolute():
        return path_obj.as_posix()
    try:
        return path_obj.relative_to(work_dir).as_posix()
    except Exception:
        return path_obj.as_posix()

# =============================================================================
# Shell helpers
# =============================================================================
def _subtask_label_from_cmd(cmd: List[str]) -> str:
    if len(cmd) >= 2 and str(cmd[1]).endswith(".py"):
        return Path(str(cmd[1])).name
    return Path(str(cmd[0])).name



def _normalize_python_script_cmd_for_cwd(cmd: List[str], cwd: Optional[Path]) -> List[str]:
    normalized = [str(x) for x in cmd]
    if len(normalized) < 2 or not normalized[1].endswith(".py"):
        return normalized

    script_name = Path(normalized[1]).name

    if cwd is not None:
        staged_script = cwd / script_name
        if staged_script.is_file():
            normalized[1] = staged_script.name
            return normalized

    repo_script = Path(__file__).resolve().parent / script_name
    if repo_script.is_file():
        normalized[1] = str(repo_script)

    return normalized


def run_cmd(
    cmd: List[str],
    dry: bool,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    report: Optional[Report] = None,
    *,
    stream_output: bool = False,
) -> None:
    del stream_output

    cwd_path = cwd.resolve() if cwd is not None else None
    cmd_norm = _normalize_python_script_cmd_for_cwd(cmd, cwd_path)

    msg = "RUN: " + " ".join(str(x) for x in cmd_norm)
    if cwd_path is not None:
        msg += f" | CWD: {cwd_path}"
    if report is not None:
        report.write(msg)
    else:
        print("\n" + msg)

    if dry:
        if report is not None:
            report.write("  [DRY_RUN] skipped")
        else:
            print("  [DRY_RUN] skipped")
        return

    merged_env = os.environ.copy()
    merged_env["PYTHONUTF8"] = "1"
    merged_env["PYTHONIOENCODING"] = "utf-8"
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})

    label = _subtask_label_from_cmd(cmd)
    if report is not None:
        report.start_subtask_console(f"run_cmd:{label}", total=1, extra="starting")

    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_f, tempfile.TemporaryFile(
        mode="w+t", encoding="utf-8"
    ) as stderr_f:
        proc = subprocess.Popen(
            cmd_norm,
            cwd=str(cwd_path) if cwd_path else None,
            env=merged_env,
            stdout=stdout_f,
            stderr=stderr_f,
            text=True,
        )

        tick = 0
        while proc.poll() is None:
            if report is not None:
                report.tick_subtask_console(extra=f"running {'.' * ((tick % 3) + 1)}")
            time.sleep(0.5)
            tick += 1

        return_code = proc.wait()

        stdout_f.seek(0)
        stderr_f.seek(0)
        stdout_text = stdout_f.read() or ""
        stderr_text = stderr_f.read() or ""

    if report is not None:
        report.finish_subtask_console(force_complete=True)

    if report is not None:
        if stdout_text.strip():
            report.write("--- stdout begin ---")
            for line in stdout_text.splitlines():
                report.write(line)
            report.write("--- stdout end ---")
        if stderr_text.strip():
            report.write("--- stderr begin ---")
            for line in stderr_text.splitlines():
                report.write(line)
            report.write("--- stderr end ---")

    if return_code != 0:
        stdout_tail = stdout_text.strip()[-4000:]
        stderr_tail = stderr_text.strip()[-4000:]
        err_lines = [
            f"Command failed with exit code {return_code}",
            f"CMD: {' '.join(str(x) for x in cmd_norm)}",
            f"CWD: {cwd_path if cwd_path is not None else Path.cwd()}",
        ]
        if stdout_tail:
            err_lines += ["--- stdout (tail) ---", stdout_tail]
        if stderr_tail:
            err_lines += ["--- stderr (tail) ---", stderr_tail]
        err_msg = "\n".join(err_lines)
        if report is not None:
            report.write(err_msg)
        raise subprocess.CalledProcessError(return_code, cmd_norm, output=stdout_text, stderr=stderr_text)

def require_file(p: Path) -> None:
    if not p.is_file():
        raise FileNotFoundError(f"Missing required file: {p.resolve()}")


def ensure_dir(p: Path, dry: bool) -> None:
    if dry:
        return
    p.mkdir(parents=True, exist_ok=True)


def _safe_tag(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "tag"


def _copy_with_collision_guard(src: Path, dst_dir: Path, *, source_hint: Optional[Path] = None) -> str:
    dst_dir.mkdir(parents=True, exist_ok=True)

    desired_name = src.name
    dst = dst_dir / desired_name
    if not dst.exists():
        shutil.copy2(src, dst)
        return desired_name

    hint = source_hint.name if source_hint is not None else src.parent.name
    hint = _safe_tag(hint)
    alt_name = f"{hint}__{src.name}"
    alt_dst = dst_dir / alt_name
    if not alt_dst.exists():
        shutil.copy2(src, alt_dst)
        return alt_name

    raise RuntimeError(
        f"Filename collision while copying into {dst_dir}: '{src.name}' and '{alt_name}' already exist. "
        f"Source file was: {src}"
    )


def _copy_graphml_tree(
    *,
    src_root: Path,
    dst_root: Path,
    dry: bool,
    report: Optional[Report] = None,
    label: str = "graphml_keep",
) -> int:
    if dry:
        if report is not None:
            report.write(f"[DRY_RUN] would keep GraphML tree {src_root} -> {dst_root} ({label})")
        return 0

    if not src_root.exists():
        raise FileNotFoundError(f"Missing GraphML source root: {src_root.resolve()}")

    graphml_files = sorted([p for p in src_root.rglob("*.graphml") if p.is_file()])
    if not graphml_files:
        if report is not None:
            report.write(f"GRAPHML_KEEP skipped empty source={src_root} label={label}")
        return 0

    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    if report is not None:
        report.start_subtask_console(f"graphml_keep:{label}", total=len(graphml_files), extra=str(src_root))

    copied = 0
    manifest_path = dst_root / "manifest.csv"
    try:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["kept_file", "relative_source_path"])
            for src_file in graphml_files:
                rel = src_file.relative_to(src_root)
                dst_file = dst_root / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                copied += 1
                w.writerow([dst_file.relative_to(dst_root).as_posix(), rel.as_posix()])
                if report is not None:
                    report.update_subtask_console(current=copied, total=len(graphml_files), extra=str(src_root))
    finally:
        if report is not None:
            report.finish_subtask_console(force_complete=True)

    if report is not None:
        report.write(f"GRAPHML_KEEP copied={copied} src={src_root} dst={dst_root} label={label}")
    return copied


def keep_graphml_roots(
    *,
    src_roots: List[Path],
    dst_root: Path,
    dry: bool,
    report: Optional[Report] = None,
    label: str = "graphml_keep",
) -> int:
    copied_total = 0
    seen = set()
    ordered_roots: List[Path] = []
    for root in src_roots:
        rp = root.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        ordered_roots.append(root)

    for src_root in ordered_roots:
        target = dst_root / _safe_tag(src_root.name)
        copied_total += _copy_graphml_tree(
            src_root=src_root,
            dst_root=target,
            dry=dry,
            report=report,
            label=f"{label}::{src_root.name}",
        )

    if report is not None:
        report.write(f"GRAPHML_KEEP_TOTAL copied={copied_total} dst={dst_root} label={label}")
    return copied_total


def _has_graphml_files(folder: Path) -> bool:
    return folder.exists() and any(folder.rglob("*.graphml"))


def _purge_graphml_under(
    root: Path,
    *,
    dry: bool,
    report: Optional[Report] = None,
    label: str = "graphml_cleanup",
) -> int:

    if os.environ.get("DT_KEEP_GRAPHML", "0") == "1":
        if report is not None:
            report.write(f"GRAPHML_PURGE skipped root={root} label={label} because DT_KEEP_GRAPHML=1")
        return 0

    files = sorted([p for p in root.rglob("*.graphml") if p.is_file()])
    n = len(files)

    if dry:
        if report is not None and n > 0:
            report.write(f"[DRY_RUN] would delete {n} .graphml files under {root} ({label})")
        return n

    if n == 0:
        return 0

    if report is not None:
        report.start_subtask_console(f"_purge_graphml_under:{label}", total=n, extra=str(root))

    deleted = 0
    try:
        for p in files:
            try:
                p.unlink()
                deleted += 1
                if report is not None:
                    report.update_subtask_console(current=deleted, total=n, extra=str(root))
            except FileNotFoundError:
                continue
    finally:
        if report is not None:
            report.finish_subtask_console(force_complete=True)

    if report is not None and deleted > 0:
        report.write(f"GRAPHML_PURGE deleted={deleted} root={root} label={label}")
    return deleted


def _has_direct_graphml_files(folder: Path) -> bool:
    return folder.exists() and any(folder.glob("*.graphml"))


def _is_dataset_graph_root(folder: Path) -> bool:
    """
    A dataset graph root is a directory that represents a whole dataset and should
    be summarised once with graph_folder_figures.py before GraphML purge.

    Accepted shapes:
      - a folder containing sim_* subfolders with GraphML beneath it
      - a non-sim_* folder containing GraphML files directly
    """
    if not folder.exists() or not folder.is_dir():
        return False
    if not _has_graphml_files(folder):
        return False

    try:
        child_dirs = [p for p in folder.iterdir() if p.is_dir()]
    except FileNotFoundError:
        return False

    if any(p.name.startswith("sim_") for p in child_dirs):
        return True

    if _has_direct_graphml_files(folder) and not folder.name.startswith("sim_"):
        return True

    return False


def _find_dataset_graph_roots(search_root: Path) -> List[Path]:
    """
    Find dataset-level graph roots that should be summarised before purge.
    Keeps deepest valid roots to avoid duplicate summaries.
    """
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

    return sorted(kept, key=lambda x: str(x))


def _rel_to_workdir_safe(path: Path, work_dir: Path) -> Path:
    try:
        return path.relative_to(work_dir)
    except Exception:
        return Path(path.name)


def archive_single_dataset_figures(
    *,
    py: str,
    graph_dir: Path,
    dst_dir: Path,
    dry: bool,
    report: Optional[Report] = None,
    cwd: Optional[Path] = None,
    identity: str = "Harry Triantafyllidis",
    title: str = "Graph dataset summary",
    label: str = "dataset",
    enable_graph_folder_figures: bool = False,
) -> None:
    if not enable_graph_folder_figures:
        msg = f"SKIP dataset_figures: graph_folder_figures disabled by CLI for {graph_dir}"
        if report is not None:
            report.write(msg)
        else:
            print(msg, flush=True)
        return

    if not _has_graphml_files(graph_dir):
        msg = f"SKIP dataset_figures: no .graphml found under {graph_dir}"
        if report is not None:
            report.write(msg)
        else:
            print(msg, flush=True)
        return

    cmd = [
        py,
        "graph_folder_figures.py",
        "--graph_dir", str(graph_dir),
        "--out_dir", str(dst_dir),
        "--identity", str(identity),
        "--title", str(title),
        "--label", str(label),
    ]
    run_cmd(cmd, dry=dry, cwd=cwd, report=report, stream_output=False)


def archive_dataset_graph_figures_before_purge(
    *,
    py: str,
    work_dir: Path,
    search_root: Path,
    archive_root: Path,
    stage_tag: str,
    dry: bool,
    report: Optional[Report] = None,
    cwd: Optional[Path] = None,
    identity: str = "Harry Triantafyllidis",
    enable_graph_folder_figures: bool = False,
) -> List[Path]:
    if not enable_graph_folder_figures:
        if report is not None:
            report.write(f"DATASET_FIGURES disabled by CLI for stage={stage_tag}")
        return []

    roots = _find_dataset_graph_roots(search_root)

    if not roots:
        if report is not None:
            report.write(f"DATASET_FIGURES none found under {search_root} for stage={stage_tag}")
        return []

    out_base = archive_root / DATASET_FIGURES_REL / _safe_tag(stage_tag)

    if report is not None:
        report.start_subtask_console(f"archive_dataset_graph_figures:{stage_tag}", total=len(roots), extra=str(search_root))

    try:
        for idx, ds_root in enumerate(roots, start=1):
            rel = _rel_to_workdir_safe(ds_root, work_dir)
            ds_tag = _safe_tag(rel.as_posix().replace("/", "__"))
            ds_title = f"{stage_tag}: {rel.as_posix()}"
            ds_out = out_base / ds_tag

            if report is not None:
                report.update_subtask_console(
                    current=idx,
                    total=len(roots),
                    extra=rel.as_posix(),
                )

            archive_single_dataset_figures(
                py=py,
                graph_dir=ds_root,
                dst_dir=ds_out,
                dry=dry,
                report=report,
                cwd=cwd,
                identity=identity,
                title=ds_title,
                label=ds_tag,
                enable_graph_folder_figures=enable_graph_folder_figures,
            )
    finally:
        if report is not None:
            report.finish_subtask_console(force_complete=True)

    return roots


# =============================================================================
# Archiving
# =============================================================================

def archive_training_outputs(
    dst: Path,
    *,
    work_dir: Path,
    dry: bool,
    report: Optional[Report] = None,
    src_dir: Optional[Path] = None,
) -> None:
    ensure_dir(dst, dry)

    if src_dir is None:
        src = work_dir / TRAINING_OUT_REL
    else:
        src = src_dir if src_dir.is_absolute() else (work_dir / src_dir)

    if dry:
        if report is not None:
            report.write(f"[DRY_RUN] would archive {src} -> {dst}")
        return

    if not src.exists():
        raise FileNotFoundError(f"Missing training outputs dir: {src.resolve()}")

    files: List[Path] = []
    for pattern in ["*.png", "*.csv", "*.pt", "*.json", "*.txt"]:
        files.extend(sorted(src.glob(pattern)))

    total = len(files)
    copied_n = 0

    if total > 0 and report is not None:
        report.start_subtask_console(f"archive_training_outputs:{dst.name}", total=total, extra=str(src))

    try:
        for f in files:
            shutil.copy2(f, dst / f.name)
            copied_n += 1
            if report is not None:
                report.update_subtask_console(current=copied_n, total=total, extra=str(src))
    finally:
        if total > 0 and report is not None:
            report.finish_subtask_console(force_complete=True)

    if report is not None:
        report.write(f"ARCHIVE training_outputs ({copied_n} files) -> {dst}")


def _archive_dst_for_training_run(
    *,
    archive_root: Path,
    step_tag: str,
    t_val: int,
    h_val: Optional[int],
    run_all_T: bool,
    run_all_horizons: bool,
) -> Path:
    dst = archive_root / _safe_tag(step_tag)

    if run_all_T:
        dst = dst / f"T{int(t_val)}"

    if run_all_horizons and h_val is not None:
        dst = dst / f"h{int(h_val)}"

    return dst


def _list_archived_step_tags(archive_root: Path, prefix: str) -> List[str]:
    if not archive_root.exists():
        return []
    tags: List[str] = []
    for p in sorted(archive_root.iterdir()):
        if p.is_dir() and p.name.startswith(prefix):
            tags.append(p.name)
    return tags


def archive_dataset_folder(
    dst: Path,
    *,
    work_dir: Path,
    folder: Path,
    dry: bool,
    report: Optional[Report] = None,
    label: str = "dataset_folder",
) -> None:
    src_dir = folder if folder.is_absolute() else (work_dir / folder)

    if dry:
        if report is not None:
            report.write(f"[DRY_RUN] would archive {label} {src_dir} -> {dst}")
        return

    if not src_dir.exists():
        raise FileNotFoundError(f"Missing {label}: {src_dir.resolve()}")

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    all_files = [p for p in sorted(src_dir.rglob("*")) if p.is_file()]
    total = len(all_files)
    copied = 0

    if total > 0 and report is not None:
        report.start_subtask_console(f"archive_dataset_folder:{label}", total=total, extra=str(src_dir))

    manifest_path = dst / "manifest.csv"
    try:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["archived_name", "relative_source_path"])

            for src_file in all_files:
                rel = src_file.relative_to(src_dir)

                if src_file.suffix.lower() in {".json", ".csv", ".txt", ".yaml", ".yml", ".graphml"}:
                    out_path = dst / rel
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, out_path)
                    w.writerow([out_path.relative_to(dst).as_posix(), rel.as_posix()])
                    copied += 1
                    if report is not None:
                        report.update_subtask_console(current=copied, total=total, extra=str(src_dir))
                    continue

                if src_file.suffix.lower() == ".pt":
                    short_name = src_file.name
                    out_path = dst / short_name

                    if len(str(out_path)) >= 235:
                        stem = src_file.stem
                        suffix = src_file.suffix
                        short_stem = stem[:80]
                        short_name = f"{copied:06d}__{short_stem}{suffix}"
                        out_path = dst / short_name

                    shutil.copy2(src_file, out_path)
                    w.writerow([short_name, rel.as_posix()])
                    copied += 1
                    if report is not None:
                        report.update_subtask_console(current=copied, total=total, extra=str(src_dir))
                    continue

                out_path = dst / rel.name
                if out_path.exists():
                    out_path = dst / f"{copied:06d}__{rel.name}"
                shutil.copy2(src_file, out_path)
                w.writerow([out_path.name, rel.as_posix()])
                copied += 1
                if report is not None:
                    report.update_subtask_console(current=copied, total=total, extra=str(src_dir))
    finally:
        if total > 0 and report is not None:
            report.finish_subtask_console(force_complete=True)

    if report is not None:
        report.write(f"ARCHIVE {label} ({copied} files) -> {dst}")

def archive_dataset_pair_figures(
    *,
    py: str,
    graph_dir: Path,
    compare_dir: Path,
    dst_dir: Path,
    dry: bool,
    report: Optional[Report] = None,
    cwd: Optional[Path] = None,
    identity: str = "Harry Triantafyllidis",
    title: str = "Train vs test graph dataset summary",
    label: str = "train",
    compare_label: str = "test",
    enable_graph_folder_figures: bool = False,
) -> None:
    if not enable_graph_folder_figures:
        msg = (
            f"SKIP dataset_figures: graph_folder_figures disabled by CLI, "
            f"graph_dir={graph_dir} compare_dir={compare_dir}"
        )
        if report is not None:
            report.write(msg)
        else:
            print(msg, flush=True)
        return

    graph_has_graphml = _has_graphml_files(graph_dir)
    compare_has_graphml = _has_graphml_files(compare_dir)

    if not graph_has_graphml or not compare_has_graphml:
        msg = (
            f"SKIP dataset_figures: graph_folder_figures.py requires .graphml files, "
            f"but got graph_dir={graph_dir} compare_dir={compare_dir}"
        )
        if report is not None:
            report.write(msg)
        else:
            print(msg, flush=True)
        return

    cmd = [
        py,
        "graph_folder_figures.py",
        "--graph_dir", str(graph_dir),
        "--compare_dir", str(compare_dir),
        "--out_dir", str(dst_dir),
        "--identity", str(identity),
        "--title", str(title),
        "--label", str(label),
        "--compare_label", str(compare_label),
    ]
    run_cmd(cmd, dry=dry, cwd=cwd, report=report, stream_output=False)


# =============================================================================
# Overleaf export
# =============================================================================

def _slugify_label(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s if s else "fig"


def _latex_escape(s: str) -> str:
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def _step_sort_key(step_tag: str) -> Tuple[float, str]:
    s = str(step_tag).strip().lower()
    m = re.match(r"step(\d+)(?:[._-]?(\d+))?", s)
    if m:
        major = int(m.group(1))
        minor = int(m.group(2)) if m.group(2) is not None else 0
        return (major + minor / 10.0, s)
    return (9999.0, s)


def _pretty_step_name(step_tag: str) -> str:
    s = str(step_tag).strip().replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()


def _infer_default_h_from_config() -> int:
    task_h = _infer_horizon_from_task_name(str(CONFIG["MODEL"].get("task", "")).strip())
    if task_h is not None:
        return int(task_h)
    return int(CONFIG["MODEL"].get("pred_horizon", 7))


def _extract_t_h_from_rel_parts(parts: Tuple[str, ...]) -> Tuple[int, int]:
    default_t = int(CONFIG["MODEL"]["T"])
    default_h = _infer_default_h_from_config()

    t_val = default_t
    h_val = default_h

    for part in parts:
        mt = re.fullmatch(r"T(\d+)", part)
        if mt:
            t_val = int(mt.group(1))
            continue

        mh = re.fullmatch(r"h(\d+)", part)
        if mh:
            h_val = int(mh.group(1))
            continue

    return t_val, h_val


def _discover_microgrid_pages(
    *,
    run_root: Path,
    figures_dir: Path,
    figure_rel_map: Optional[Dict[Path, str]] = None,
) -> Dict[Tuple[str, int, int], List[Dict[str, str]]]:
    del figures_dir

    figure_rel_map = figure_rel_map or {}
    pages: Dict[Tuple[str, int, int], List[Dict[str, str]]] = {}

    for track_dir in sorted(run_root.glob("TRACK_*")):
        repro_root = track_dir / "work" / "repro_artifacts_steps_1_7"
        if not repro_root.exists():
            continue

        for step_dir in sorted(repro_root.iterdir()):
            if not step_dir.is_dir():
                continue

            for candidate_dir in [step_dir] + sorted([p for p in step_dir.rglob("*") if p.is_dir()]):
                cm_test = candidate_dir / "confusion_matrix_test.png"
                cm_train = candidate_dir / "confusion_matrix.png"
                roc_train = candidate_dir / "roc_curve.png"
                roc_test = candidate_dir / "roc_curve_test.png"

                cm = cm_test if cm_test.exists() else cm_train

                if not cm.exists():
                    continue
                if not roc_train.exists():
                    continue
                if not roc_test.exists():
                    continue

                rel_from_repro = candidate_dir.relative_to(repro_root)
                parts = rel_from_repro.parts
                if len(parts) == 0:
                    continue

                step_tag = parts[0]
                t_val, h_val = _extract_t_h_from_rel_parts(parts[1:])
                key = (track_dir.name, t_val, h_val)

                def _to_fig_path(p: Path) -> str:
                    p_resolved = p.resolve()
                    if p_resolved in figure_rel_map:
                        return figure_rel_map[p_resolved]
                    rel = p.relative_to(run_root)
                    return (Path("figures") / rel).as_posix()

                pages.setdefault(key, []).append(
                    {
                        "step_tag": step_tag,
                        "cm": _to_fig_path(cm),
                        "roc_train": _to_fig_path(roc_train),
                        "roc_test": _to_fig_path(roc_test),
                    }
                )

    for key in list(pages.keys()):
        pages[key] = sorted(
            pages[key],
            key=lambda row: _step_sort_key(row["step_tag"]),
        )

    return pages


def _latex_microgrid_for_page(
    *,
    track_name: str,
    t_val: int,
    h_val: int,
    rows: List[Dict[str, str]],
) -> List[str]:
    lines: List[str] = []

    caption = (
        f"Comparison grid for {_latex_escape(track_name)} at "
        f"$T={int(t_val)}$ and $H={int(h_val)}$. "
        f"Each row corresponds to one experiment configuration."
    )
    label = _slugify_label(f"{track_name}-T{t_val}-H{h_val}")

    lines.append(r"\clearpage")
    lines.append(r"\begin{figure}[p]")
    lines.append(r"  \centering")
    lines.append(r"  \setlength{\tabcolsep}{3pt}")
    lines.append(r"  \renewcommand{\arraystretch}{1.1}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{m{0.18\textwidth}m{0.26\textwidth}m{0.26\textwidth}m{0.26\textwidth}}")
    lines.append(r"    \textbf{Experiment} & \textbf{Confusion matrix} & \textbf{Train ROC} & \textbf{Test ROC} \\")
    lines.append(r"    \hline")

    for row in rows:
        step_name = _latex_escape(_pretty_step_name(row["step_tag"]))
        cm = row["cm"]
        roc_train = row["roc_train"]
        roc_test = row["roc_test"]

        lines.append(
            "    "
            + step_name
            + " & "
            + rf"\includegraphics[width=\linewidth,height=0.18\textheight,keepaspectratio]{{{cm}}}"
            + " & "
            + rf"\includegraphics[width=\linewidth,height=0.18\textheight,keepaspectratio]{{{roc_train}}}"
            + " & "
            + rf"\includegraphics[width=\linewidth,height=0.18\textheight,keepaspectratio]{{{roc_test}}}"
            + r" \\"
        )

    lines.append(r"  \end{tabular}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{fig:{label}}}")
    lines.append(r"\end{figure}")
    lines.append("")

    return lines


def _mean_sd(values: Iterable[float]) -> Tuple[Optional[float], Optional[float], int]:
    vals: List[float] = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        if fv != fv:
            continue
        vals.append(fv)

    n = len(vals)
    if n == 0:
        return None, None, 0
    if n == 1:
        return vals[0], 0.0, 1

    mean_v = sum(vals) / n
    var_v = sum((x - mean_v) ** 2 for x in vals) / max(1, n - 1)
    return mean_v, var_v ** 0.5, n


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if v != v:
        return None
    return v


def _extract_binary_sens_spec_from_confusion_matrix(cm: Any) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(cm, list) or len(cm) != 2:
        return None, None
    try:
        row0 = list(cm[0])
        row1 = list(cm[1])
        if len(row0) != 2 or len(row1) != 2:
            return None, None
        tn = float(row0[0])
        fp = float(row0[1])
        fn = float(row1[0])
        tp = float(row1[1])
    except Exception:
        return None, None

    sensitivity = None if (tp + fn) <= 0 else tp / (tp + fn)
    specificity = None if (tn + fp) <= 0 else tn / (tn + fp)
    return sensitivity, specificity


def _metric_value_from_split(split: Dict[str, Any], metric_name: str) -> Optional[float]:
    if not isinstance(split, dict):
        return None

    metrics = split.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    if metric_name == "auroc":
        for key in ("roc_auc", "roc_auc_macro_ovr"):
            v = _safe_float(metrics.get(key))
            if v is not None:
                return v
        return None

    if metric_name == "accuracy":
        return _safe_float(metrics.get("accuracy"))

    if metric_name == "f1":
        for key in ("f1_macro", "f1_weighted"):
            v = _safe_float(metrics.get(key))
            if v is not None:
                return v
        return None

    if metric_name == "sensitivity":
        sens, _ = _extract_binary_sens_spec_from_confusion_matrix(split.get("confusion_matrix"))
        return sens

    if metric_name == "specificity":
        _, spec = _extract_binary_sens_spec_from_confusion_matrix(split.get("confusion_matrix"))
        return spec

    return None


def _format_mean_sd_latex(mean_v: Optional[float], sd_v: Optional[float], n: int) -> str:
    if mean_v is None:
        return '--'
    if n <= 1 or sd_v is None:
        return f'{mean_v:.3f}'
    return f'{mean_v:.3f} $\\pm$ {sd_v:.3f}'


def _discover_metrics_tables(run_root: Path) -> Dict[Tuple[str, int, int], List[Dict[str, Any]]]:
    tables: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    metric_names = ["auroc", "accuracy", "f1", "sensitivity", "specificity"]

    for track_dir in sorted(run_root.glob("TRACK_*")):
        repro_root = track_dir / "work" / "repro_artifacts_steps_1_7"
        if not repro_root.exists():
            continue

        grouped: Dict[Tuple[str, int, int, str], List[Path]] = {}
        for summary_path in sorted(repro_root.rglob("metrics_summary.json")):
            rel_parent = summary_path.parent.relative_to(repro_root)
            parts = rel_parent.parts
            if len(parts) == 0:
                continue
            step_tag = parts[0]
            t_val, h_val = _extract_t_h_from_rel_parts(parts[1:])
            grouped.setdefault((track_dir.name, t_val, h_val, step_tag), []).append(summary_path)

        for (track_name, t_val, h_val, step_tag), summary_paths in grouped.items():
            train_values: Dict[str, List[float]] = {k: [] for k in metric_names}
            test_values: Dict[str, List[float]] = {k: [] for k in metric_names}

            for summary_path in summary_paths:
                try:
                    payload = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception:
                    continue

                train_split = payload.get("validation", {})
                test_split = payload.get("test", {})

                for metric_name in metric_names:
                    train_v = _metric_value_from_split(train_split, metric_name)
                    test_v = _metric_value_from_split(test_split, metric_name)
                    if train_v is not None:
                        train_values[metric_name].append(train_v)
                    if test_v is not None:
                        test_values[metric_name].append(test_v)

            row = {
                "step_tag": step_tag,
                "n_runs": len(summary_paths),
                "train": {},
                "test": {},
            }

            for metric_name in metric_names:
                train_mean, train_sd, train_n = _mean_sd(train_values[metric_name])
                test_mean, test_sd, test_n = _mean_sd(test_values[metric_name])
                row["train"][metric_name] = {"mean": train_mean, "sd": train_sd, "n": train_n}
                row["test"][metric_name] = {"mean": test_mean, "sd": test_sd, "n": test_n}

            tables.setdefault((track_name, t_val, h_val), []).append(row)

    for key in list(tables.keys()):
        tables[key] = sorted(tables[key], key=lambda row: _step_sort_key(str(row.get("step_tag", ""))))

    return tables


def _metrics_step_display_order(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _rank(step_tag: str) -> Tuple[float, int, str]:
        s = str(step_tag).strip().lower()

        if s == "step4_baseline":
            return (4.0, 0, s)
        if s == "step5_core_node_features_only":
            return (5.0, 0, s)
        if s == "step5_no_edge_weights":
            return (5.0, 1, s)

        m_delay = re.match(r"step6_delay_(.+)$", s)
        if m_delay:
            mm = re.search(r"(\d+)", m_delay.group(1))
            val = int(mm.group(1)) if mm else 999999
            return (6.1, val, s)

        m_freq = re.match(r"step6_freq_(.+)$", s)
        if m_freq:
            mm = re.search(r"(\d+)", m_freq.group(1))
            val = int(mm.group(1)) if mm else 999999
            return (6.2, val, s)

        if s == "step7_sweep":
            return (7.0, 0, s)

        base, key = _step_sort_key(s)
        return (base, 999999, key)

    return sorted(rows, key=lambda row: _rank(str(row.get("step_tag", ""))))


def _extract_step6_condition_value(step_tag: str, prefix: str) -> Optional[int]:
    s = str(step_tag).strip().lower()
    if not s.startswith(prefix):
        return None

    suffix = s[len(prefix):]
    matches = re.findall(r"(\d+)", suffix)
    if not matches:
        return None
    return int(matches[-1])


def _step_label_metrics(step_tag: str) -> str:
    s = str(step_tag).strip().lower()
    main = _step_label_main(s)
    if main is not None:
        return main

    if s.startswith("step6_delay_"):
        delay_val = _extract_step6_condition_value(s, "step6_delay_")
        if delay_val is not None:
            return rf"\shortstack[l]{{Step6.1 Delay {delay_val}}}"
        return r"\shortstack[l]{Step6.1 Delay}"

    if s.startswith("step6_freq_"):
        freq_val = _extract_step6_condition_value(s, "step6_freq_")
        if freq_val is not None:
            return rf"\shortstack[l]{{Step6.2 Freq {freq_val}}}"
        return r"\shortstack[l]{Step6.2 Freq}"

    return _latex_escape(_pretty_step_name(step_tag))


def _latex_metrics_table_for_page(
    *,
    track_name: str,
    t_val: int,
    h_val: int,
    rows: List[Dict[str, Any]],
) -> List[str]:
    lines: List[str] = []
    metric_pairs = [
        ("AUROC", "auroc"),
        ("Acc", "accuracy"),
        ("F1", "f1"),
        ("Sens", "sensitivity"),
        ("Spec", "specificity"),
    ]
    label = _slugify_label(f"{track_name}-metrics-T{t_val}-H{h_val}")
    caption = (
        f"Compact quantitative summary for {_latex_escape(track_name)} at "
        f"$T={int(t_val)}$ and $H={int(h_val)}$. "
        r"Entries report mean performance across archived runs; when more than one run is available, the table shows mean $\pm$ standard deviation."
    )

    ordered_rows = _metrics_step_display_order(rows)

    lines.append(r"\begin{table}[p]")
    lines.append(r"  \centering")
    lines.append(r"  \scriptsize")
    lines.append(r"  \setlength{\tabcolsep}{3.0pt}")
    lines.append(r"  \renewcommand{\arraystretch}{1.12}")
    lines.append(r"  \begin{adjustbox}{max width=\textwidth,center}")
    lines.append(r"    \begin{tabular}{m{0.22\textwidth}ccccc|ccccc|c}")
    lines.append(r"      \textbf{Experiment} & \multicolumn{5}{c|}{\textbf{Train}} & \multicolumn{5}{c|}{\textbf{Test}} & \textbf{$n$} \\")
    lines.append(r"      & \textbf{AUROC} & \textbf{Acc} & \textbf{F1} & \textbf{Sens} & \textbf{Spec} & \textbf{AUROC} & \textbf{Acc} & \textbf{F1} & \textbf{Sens} & \textbf{Spec} & \\")
    lines.append(r"      \hline")

    for row in ordered_rows:
        step_tag = str(row.get("step_tag", "")).strip()
        label_tex = _step_label_metrics(step_tag)
        cells = [f"      {label_tex}"]
        for _, key in metric_pairs:
            metric = row.get("train", {}).get(key, {})
            cells.append(_format_mean_sd_latex(metric.get("mean"), metric.get("sd"), int(metric.get("n", 0))))
        for _, key in metric_pairs:
            metric = row.get("test", {}).get(key, {})
            cells.append(_format_mean_sd_latex(metric.get("mean"), metric.get("sd"), int(metric.get("n", 0))))
        cells.append(str(int(row.get("n_runs", 0))))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"    \end{tabular}")
    lines.append(r"  \end{adjustbox}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{tab:{label}}}")
    lines.append(r"\end{table}")
    lines.append("")
    return lines


def _dataset_png_order_key(name: str) -> Tuple[int, str]:
    order = {
        "figure_microgrid": 1,
        "figure_distributions": 2,
        "figure_communities_and_centrality": 3,
        "figure_flow_sankey": 4,
        "figure_timeline_nodes_edges": 5,
        "figure_state_percentages": 6,
        "figure_train_vs_test_shift": 7,
        "figure_train_vs_test_ecdf": 8,
        "figure_timeline_diff_test_minus_train": 9,
    }
    for prefix, rank in order.items():
        if name.startswith(prefix):
            return (rank, name)
    return (999, name)


def _pretty_dataset_page_name(rel_parts: Tuple[str, ...]) -> str:
    s = " / ".join(rel_parts)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()


def _discover_dataset_figure_sets(
    run_root: Path,
    figure_rel_map: Optional[Dict[Path, str]] = None,
) -> List[Dict[str, Any]]:
    figure_rel_map = figure_rel_map or {}
    out: List[Dict[str, Any]] = []

    for track_dir in sorted(run_root.glob("TRACK_*")):
        dataset_root = track_dir / "work" / "repro_artifacts_steps_1_7" / DATASET_FIGURES_REL
        if not dataset_root.exists():
            continue

        for ds_dir in sorted([p for p in dataset_root.rglob("*") if p.is_dir()]):
            pngs = sorted(
                [p for p in ds_dir.glob("*.png") if p.is_file()],
                key=lambda p: _dataset_png_order_key(p.name),
            )
            if not pngs:
                continue

            rel = ds_dir.relative_to(dataset_root)
            rel_pngs = [
                figure_rel_map.get(p.resolve(), (Path("figures") / p.relative_to(run_root)).as_posix())
                for p in pngs
            ]

            out.append(
                {
                    "track_name": track_dir.name,
                    "dataset_rel_parts": rel.parts,
                    "dataset_title": _pretty_dataset_page_name(rel.parts),
                    "pngs": rel_pngs,
                }
            )

    return out


def _latex_dataset_figures_for_pages(
    *,
    track_name: str,
    dataset_title: str,
    dataset_rel_parts: Tuple[str, ...],
    pngs: List[str],
    chunk_size: int = 4,
) -> List[str]:
    lines: List[str] = []
    page_chunks = [pngs[i:i + chunk_size] for i in range(0, len(pngs), chunk_size)]

    for page_idx, chunk in enumerate(page_chunks, start=1):
        label = _slugify_label(
            f"{track_name}-{'-'.join(dataset_rel_parts)}-page-{page_idx}"
        )
        caption = (
            f"Dataset-level raw-graph summaries for {_latex_escape(track_name)}: "
            f"{_latex_escape(dataset_title)} (page {page_idx} of {len(page_chunks)})."
        )

        lines.append(r"\clearpage")
        lines.append(r"\begin{figure}[p]")
        lines.append(r"  \centering")

        for idx, fig_path in enumerate(chunk, start=1):
            lines.append(r"  \begin{minipage}[t]{0.48\textwidth}")
            lines.append(r"    \centering")
            lines.append(
                rf"    \includegraphics[width=\linewidth,height=0.34\textheight,keepaspectratio]{{{fig_path}}}"
            )
            lines.append(r"  \end{minipage}")
            if idx % 2 == 1 and idx != len(chunk):
                lines.append(r"  \hfill")
            else:
                lines.append(r"  \vspace{0.7em}")

        lines.append(f"  \\caption{{{caption}}}")
        lines.append(f"  \\label{{fig:{label}}}")
        lines.append(r"\end{figure}")
        lines.append("")

    return lines


def _short_overleaf_rel_path(rel: Path) -> Path:
    rel = Path(rel)
    suffix = rel.suffix.lower()
    stem = _safe_tag(rel.stem)[:80] or "figure"
    digest = hashlib.sha1(rel.as_posix().encode("utf-8")).hexdigest()[:12]

    top_parts = list(rel.parts[:3])
    top_slug = _safe_tag("__".join(top_parts))[:60] or "misc"
    filename = f"{top_slug}__{stem}__{digest}{suffix}"
    return Path(top_slug) / filename




def _step_label_main(step_tag: str) -> Optional[str]:
    s = str(step_tag).strip().lower()
    if s == "step4_baseline":
        return "Step4 Baseline"
    if s == "step5_core_node_features_only":
        return r"\shortstack[l]{Step5 Core Node\\Features Only}"
    if s == "step5_no_edge_weights":
        return r"\shortstack[l]{Step5 No Edge\\Weights}"
    if s == "step7_sweep":
        return "Step7 Sweep"
    return None


def _step_label_cross_track(step_tag: str, track_name: str) -> Optional[str]:
    s = str(step_tag).strip().lower()
    track_suffix = "Ground truth" if str(track_name).strip() == "TRACK_ground_truth" else "Partial observation"

    delay_val = _extract_step6_condition_value(s, "step6_delay_")
    if delay_val is not None:
        return rf"\shortstack[l]{{Step6 Delay {delay_val}\\{track_suffix}}}"

    freq_val = _extract_step6_condition_value(s, "step6_freq_")
    if freq_val is not None:
        return rf"\shortstack[l]{{Step6 Freq {freq_val}\\{track_suffix}}}"

    return None


def _latex_row_from_page_entry(row: Dict[str, str], label_tex: str, *, image_height: str) -> str:
    cm = row["cm"]
    roc_train = row["roc_train"]
    roc_test = row["roc_test"]
    return (
        f"      {label_tex} &\n"
        f"      \\includegraphics[width=\\linewidth,height={image_height},keepaspectratio]{{{cm}}} &\n"
        f"      \\includegraphics[width=\\linewidth,height={image_height},keepaspectratio]{{{roc_train}}} &\n"
        f"      \\includegraphics[width=\\linewidth,height={image_height},keepaspectratio]{{{roc_test}}} \\\\"
    )


def _latex_grid_block(
    *,
    caption: str,
    label: str,
    rows_tex: List[str],
    arraystretch: str,
) -> List[str]:
    lines: List[str] = []
    lines.append(r"\begin{figure}[!]")
    lines.append(r"  \centering")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(rf"  \renewcommand{{\arraystretch}}{{{arraystretch}}}")
    lines.append(r"  \small")
    lines.append(r"  \begin{adjustbox}{max totalsize={\textwidth}{0.92\textheight},center}")
    lines.append(r"    \begin{tabular}{m{0.18\textwidth}m{0.26\textwidth}m{0.26\textwidth}m{0.26\textwidth}}")
    lines.append(r"      \textbf{Experiment} & \textbf{Confusion matrix} & \textbf{Train ROC} & \textbf{Test ROC} \\")
    lines.append(r"      \hline")
    lines.extend(rows_tex)
    lines.append(r"    \end{tabular}")
    lines.append(r"  \end{adjustbox}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"\end{figure}")
    lines.append("")
    return lines


def _build_main_track_grid_pages(pages: Dict[Tuple[str, int, int], List[Dict[str, str]]]) -> List[str]:
    lines: List[str] = []
    desired_order = [
        "step4_baseline",
        "step5_core_node_features_only",
        "step5_no_edge_weights",
        "step7_sweep",
    ]
    for track_name, t_val, h_val in sorted(pages.keys(), key=lambda x: (x[0], int(x[1]), int(x[2]))):
        page_rows = pages[(track_name, t_val, h_val)]
        row_map = {str(r.get("step_tag", "")).strip().lower(): r for r in page_rows}
        rows_tex: List[str] = []
        for step_tag in desired_order:
            row = row_map.get(step_tag)
            if row is None:
                continue
            label_tex = _step_label_main(step_tag)
            if label_tex is None:
                continue
            rows_tex.append(_latex_row_from_page_entry(row, label_tex, image_height="0.15\\textheight"))
            rows_tex.append("")

        if not rows_tex:
            continue

        track_label_slug = _slugify_label(track_name.replace("TRACK_", "track-"))
        caption = (
            f"Main comparison grid for {_latex_escape(track_name)} at $T={int(t_val)}$ and $H={int(h_val)}$. "
            r"Step~6 delay/frequency ablations are shown separately in Figures\ref{fig:track-step6-delay-cross-track}, "
            r"\ref{fig:track-step6-frequency-cross-track}."
        )
        label = f"fig:{track_label_slug}-main-t{int(t_val)}-h{int(h_val)}"
        lines.extend(
            _latex_grid_block(
                caption=caption,
                label=label,
                rows_tex=rows_tex,
                arraystretch="1.08",
            )
        )
    return lines


def _build_cross_track_step6_grid_pages(
    pages: Dict[Tuple[str, int, int], List[Dict[str, str]]],
    *,
    mode: str,
) -> List[str]:
    lines: List[str] = []
    if mode not in {"delay", "frequency"}:
        raise ValueError(f"Unsupported mode: {mode}")

    prefix = "step6_delay_" if mode == "delay" else "step6_freq_"
    figure_label = "fig:track-step6-delay-cross-track" if mode == "delay" else "fig:track-step6-frequency-cross-track"
    figure_title = "delay" if mode == "delay" else "screening-frequency"
    figure_title_caption = "delay" if mode == "delay" else "screening frequency"

    combos = sorted(pages.keys(), key=lambda x: (int(x[1]), int(x[2]), x[0]))
    grouped_by_th: Dict[Tuple[int, int], List[Tuple[str, List[Dict[str, str]]]]] = {}
    for track_name, t_val, h_val in combos:
        grouped_by_th.setdefault((int(t_val), int(h_val)), []).append((track_name, pages[(track_name, t_val, h_val)]))

    for (t_val, h_val), per_track_rows in grouped_by_th.items():
        rows_candidates: List[Tuple[int, str, Dict[str, str], str]] = []
        for track_name, page_rows in per_track_rows:
            for row in page_rows:
                step_tag = str(row.get("step_tag", "")).strip().lower()
                if not step_tag.startswith(prefix):
                    continue
                label_tex = _step_label_cross_track(step_tag, track_name)
                if label_tex is None:
                    continue
                m = re.search(r"(\d+)", step_tag)
                rank_num = int(m.group(1)) if m else 999999
                track_rank = 0 if track_name == "TRACK_ground_truth" else 1
                rows_candidates.append((rank_num, f"{rank_num:06d}_{track_rank}_{track_name}_{step_tag}", row, label_tex))

        rows_candidates.sort(key=lambda x: (x[0], x[1]))
        rows_tex = []
        for _, _, row, label_tex in rows_candidates:
            rows_tex.append(_latex_row_from_page_entry(row, label_tex, image_height="0.105\\textheight"))
            rows_tex.append("")

        if not rows_tex:
            continue

        caption = (
            f"Cross-track comparison of Step~6 {figure_title_caption} ablations at $T={int(t_val)}$ and $H={int(h_val)}$. "
            "Rows are grouped by intervention level and state-observation regime."
        )
        label = figure_label if (int(t_val), int(h_val)) == sorted(grouped_by_th.keys())[0] else f"{figure_label}-t{int(t_val)}-h{int(h_val)}"
        lines.extend(
            _latex_grid_block(
                caption=caption,
                label=label,
                rows_tex=rows_tex,
                arraystretch="1.06",
            )
        )
    return lines


def export_overleaf_package(
    *,
    run_root: Path,
    archive_root: Path,
    out_dir: Path,
    dry: bool,
    report: Optional[Report] = None,
) -> Path:
    out_dir = out_dir if out_dir.is_absolute() else (run_root / out_dir)
    figures_dir = out_dir / "figures"
    latex_path = out_dir / "latex.txt"

    if dry:
        if report is not None:
            report.write(f"[DRY_RUN] would export Overleaf package to: {out_dir}")
        return latex_path

    if out_dir.exists():
        shutil.rmtree(out_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".pdf", ".jpg", ".jpeg"}
    copied: List[Path] = []
    figure_rel_map: Dict[Path, str] = {}
    for p in sorted(archive_root.rglob("*")):
        if out_dir in p.parents or p == out_dir:
            continue
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(archive_root)
        short_rel = _short_overleaf_rel_path(rel)
        dst = figures_dir / short_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        copied.append(dst)
        figure_rel_map[p.resolve()] = (Path("figures") / short_rel).as_posix()

    lines: List[str] = []
    lines.append("% Auto-generated by experiments_pb.py")
    lines.append("% Assumes figures are located under: figures/")
    lines.append("% Suggested preamble additions:")
    lines.append("%   \\usepackage{graphicx}")
    lines.append("%   \\usepackage{float}")
    lines.append("%   \\usepackage{array}")
    lines.append("%   \\usepackage{caption}")
    lines.append("%   \\usepackage{adjustbox}")
    lines.append("")

    pages = _discover_microgrid_pages(
        run_root=run_root,
        figures_dir=figures_dir,
        figure_rel_map=figure_rel_map,
    )
    metrics_tables = _discover_metrics_tables(run_root)
    dataset_sets = _discover_dataset_figure_sets(
        run_root=run_root,
        figure_rel_map=figure_rel_map,
    )

    main_lines = _build_main_track_grid_pages(pages)
    delay_lines = _build_cross_track_step6_grid_pages(pages, mode="delay")
    freq_lines = _build_cross_track_step6_grid_pages(pages, mode="frequency")

    if main_lines:
        lines.extend(main_lines)
        for (track_name, t_val, h_val) in sorted(pages.keys(), key=lambda x: (x[0], int(x[1]), int(x[2]))):
            metric_rows = metrics_tables.get((track_name, t_val, h_val), [])
            if metric_rows:
                lines.extend(
                    _latex_metrics_table_for_page(
                        track_name=track_name,
                        t_val=int(t_val),
                        h_val=int(h_val),
                        rows=metric_rows,
                    )
                )
            else:
                lines.append(
                    f"% No metrics-summary table could be assembled for {track_name} T={int(t_val)} H={int(h_val)}."
                )
    else:
        lines.append("% No main track comparison grids could be assembled from the archived outputs.")

    if delay_lines:
        lines.extend(delay_lines)
    else:
        lines.append("% No Step 6 delay comparison grid could be assembled from the archived outputs.")

    if freq_lines:
        lines.extend(freq_lines)
    else:
        lines.append("% No Step 6 frequency comparison grid could be assembled from the archived outputs.")

    if not dataset_sets:
        lines.append("% No dataset-figure pages could be assembled from archived raw-graph summaries.")
    else:
        for item in dataset_sets:
            lines.extend(
                _latex_dataset_figures_for_pages(
                    track_name=str(item["track_name"]),
                    dataset_title=str(item["dataset_title"]),
                    dataset_rel_parts=tuple(item["dataset_rel_parts"]),
                    pngs=list(item["pngs"]),
                )
            )

    latex_path.write_text("\n".join(lines), encoding="utf-8")
    if report is not None:
        report.write(f"OVERLEAF package exported -> {out_dir}")
        report.write(f"OVERLEAF latex.txt -> {latex_path}")
        report.write(f"OVERLEAF copied figures: {len(copied)}")
        report.write(f"OVERLEAF main grids: {1 if main_lines else 0}")
        report.write(f"OVERLEAF metrics tables: {len(metrics_tables)}")
        report.write(f"OVERLEAF step6 delay grids: {1 if delay_lines else 0}")
        report.write(f"OVERLEAF step6 frequency grids: {1 if freq_lines else 0}")
        report.write(f"OVERLEAF dataset-figure sets: {len(dataset_sets)}")
    return latex_path


# =============================================================================
# Task / CLI arg building
# =============================================================================

def step_key_to_float(k: str) -> float:
    if k.endswith("b") and k[:-1].isdigit():
        return float(k[:-1]) + 0.1
    return float(k)


def _parse_int_list(s: str, *, what: str) -> List[int]:
    s = str(s).strip()
    if s == "":
        return []
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    out: List[int] = []
    for p in parts:
        if not re.fullmatch(r"\d+", p):
            raise ValueError(f"Invalid {what} token '{p}'. Expected comma-separated integers.")
        v = int(p)
        if v <= 0:
            raise ValueError(f"Invalid {what} '{v}'. Must be positive.")
        out.append(v)
    return sorted(set(out))


def _parse_horizons_list(s: str) -> List[int]:
    return _parse_int_list(s, what="horizon")


def _parse_T_list(s: str) -> List[int]:
    return _parse_int_list(s, what="T")


def _task_with_horizon(task: str, H: int) -> str:
    t = str(task).strip()
    if re.search(r"_h\d+$", t):
        return re.sub(r"_h\d+$", f"_h{int(H)}", t)
    return f"{t}_h{int(H)}"


def _infer_horizon_from_task_name(task: str) -> Optional[int]:
    m = re.search(r"_h(\d+)$", str(task).strip())
    if not m:
        return None
    try:
        h = int(m.group(1))
    except Exception:
        return None
    return h if h > 0 else None


def _get_max_T_needed_runtime(T_list_runtime: List[int]) -> int:
    vals = [int(CONFIG.get("MODEL", {}).get("T", 7))]
    vals.extend(int(t) for t in T_list_runtime if int(t) > 0)
    return max(vals)


def _build_train_args_from_config(
    model_cfg: Dict[str, Any],
    py: str,
    data_folder: Path,
    *,
    work_dir: Optional[Path] = None,
    task_override: Optional[str] = None,
    out_dir: Optional[str] = None,
    T_override: Optional[int] = None,
    test_folder: Optional[Path] = None,
    train_model_override: Optional[bool] = None,
) -> List[str]:
    task_eff = str(task_override).strip() if task_override is not None else str(model_cfg["task"]).strip()
    T_eff = int(T_override) if T_override is not None else int(model_cfg["T"])

    data_folder_arg = str(data_folder)
    out_dir_arg = str(out_dir) if out_dir is not None else None
    test_folder_arg = str(test_folder) if test_folder is not None else None
    if work_dir is not None:
        data_folder_arg = _to_work_dir_relative_path(data_folder, work_dir)
        if out_dir_arg is not None and str(out_dir_arg).strip() != "":
            out_dir_arg = _to_work_dir_relative_path(Path(out_dir_arg), work_dir)
        if test_folder_arg is not None and str(test_folder_arg).strip() != "":
            test_folder_arg = _to_work_dir_relative_path(Path(test_folder_arg), work_dir)

    args: List[str] = [
        py, "train_amr_dygformer.py",
        "--data_folder", data_folder_arg,
        "--task", task_eff,
        "--T", str(T_eff),
    ]

    if bool(model_cfg.get("use_task_hparams", True)):
        args.append("--use_task_hparams")
    else:
        args += [
            "--sliding_step", str(int(model_cfg["sliding_step"])),
            "--hidden", str(int(model_cfg["hidden"])),
            "--heads", str(int(model_cfg["heads"])),
            "--dropout", str(float(model_cfg["dropout"])),
            "--transformer_layers", str(int(model_cfg["transformer_layers"])),
            "--sage_layers", str(int(model_cfg["sage_layers"])),
            "--batch_size", str(int(model_cfg["batch_size"])),
            "--epochs", str(int(model_cfg["epochs"])),
            "--lr", str(float(model_cfg["lr"])),
        ]
        if bool(model_cfg.get("use_cls", False)):
            args.append("--use_cls")

    ns_flag = "true" if bool(model_cfg.get("neighbor_sampling", False)) else "false"
    args += ["--neighbor_sampling", ns_flag]

    if bool(model_cfg.get("neighbor_sampling", False)):
        args += [
            "--num_neighbors", str(model_cfg["num_neighbors"]),
            "--seed_count", str(int(model_cfg["seed_count"])),
            "--seed_strategy", str(model_cfg["seed_strategy"]),
            "--seed_batch_size", str(int(model_cfg["seed_batch_size"])),
            "--max_sub_batches", str(int(model_cfg["max_sub_batches"])),
        ]
    else:
        args += ["--max_neighbors", str(int(model_cfg["max_neighbors"]))]

    args += [
        "--attn_top_k", str(int(model_cfg["attn_top_k"])),
        "--attn_rank_by", str(model_cfg["attn_rank_by"]),
    ]

    train_model_effective = bool(model_cfg.get("train_model", True)) if train_model_override is None else bool(train_model_override)
    train_flag = "true" if train_model_effective else "false"
    args += ["--train_model", train_flag]

    if out_dir_arg is not None and str(out_dir_arg).strip() != "":
        args += ["--out_dir", str(out_dir_arg)]

    if test_folder_arg is not None and str(test_folder_arg).strip() != "":
        args += ["--test_folder", str(test_folder_arg)]

    return args


def _build_sim_extra_args_global_only(sim_cfg: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts += [
        "--num_regions", str(int(sim_cfg["num_regions"])),
        "--num_patients", str(int(sim_cfg["num_patients"])),
        "--num_staff", str(int(sim_cfg["num_staff"])),
        "--num_wards", str(int(sim_cfg["num_wards"])),
    ]
    if bool(sim_cfg.get("export_yaml", True)):
        parts.append("--export_yaml")
    if not bool(sim_cfg.get("export_gif", True)):
        parts.append("--no_export_gif")
    if bool(sim_cfg.get("override_staff_wards_per_staff", False)):
        parts += ["--staff_wards_per_staff", str(int(sim_cfg["staff_wards_per_staff"]))]

    if bool(sim_cfg.get("enable_superspreader", False)) and str(sim_cfg.get("superspreader_staff", "")).strip():
        parts += [
            "--superspreader_staff", str(sim_cfg["superspreader_staff"]),
            "--superspreader_state", str(sim_cfg["superspreader_state"]),
            "--superspreader_start_day", str(int(sim_cfg["superspreader_start_day"])),
            "--superspreader_end_day", str(int(sim_cfg["superspreader_end_day"])),
            "--superspreader_patient_frac_mult", str(float(sim_cfg["superspreader_patient_frac_mult"])),
            "--superspreader_patient_min_add", str(int(sim_cfg["superspreader_patient_min_add"])),
            "--superspreader_staff_contacts", str(int(sim_cfg["superspreader_staff_contacts"])),
            "--superspreader_edge_weight_mult", str(float(sim_cfg["superspreader_edge_weight_mult"])),
        ]

    if bool(sim_cfg.get("enable_admit_import_seasonality", False)) and str(sim_cfg.get("admit_import_seasonality", "none")) != "none":
        mode = str(sim_cfg["admit_import_seasonality"])
        parts += [
            "--admit_import_seasonality", mode,
            "--admit_import_period_days", str(int(sim_cfg["admit_import_period_days"])),
            "--admit_import_pmax_cs", str(float(sim_cfg["admit_import_pmax_cs"])),
            "--admit_import_pmax_cr", str(float(sim_cfg["admit_import_pmax_cr"])),
        ]
        if mode == "sinusoid":
            parts += [
                "--admit_import_amp", str(float(sim_cfg["admit_import_amp"])),
                "--admit_import_phase_day", str(int(sim_cfg["admit_import_phase_day"])),
            ]
        elif mode == "piecewise":
            parts += [
                "--admit_import_high_start_day", str(int(sim_cfg["admit_import_high_start_day"])),
                "--admit_import_high_end_day", str(int(sim_cfg["admit_import_high_end_day"])),
                "--admit_import_high_mult", str(float(sim_cfg["admit_import_high_mult"])),
                "--admit_import_low_mult", str(float(sim_cfg["admit_import_low_mult"])),
            ]
        elif mode == "shock":
            parts += [
                "--admit_import_shock_min_days", str(int(sim_cfg["admit_import_shock_min_days"])),
                "--admit_import_shock_max_days", str(int(sim_cfg["admit_import_shock_max_days"])),
                "--admit_import_shock_mult_min", str(float(sim_cfg["admit_import_shock_mult_min"])),
                "--admit_import_shock_mult_max", str(float(sim_cfg["admit_import_shock_mult_max"])),
            ]
    return " ".join(parts)


def _build_convert_extra_args(convert_cfg: Dict[str, Any]) -> str:
    parts: List[str] = []

    horizons = str(convert_cfg.get("horizons", "")).strip()
    if horizons:
        parts += ["--horizons", horizons]

    workers = convert_cfg.get("workers", None)
    if workers is not None:
        workers_int = int(workers)
        if workers_int < 0:
            raise ValueError(f"CONFIG['CONVERT']['workers'] must be >= 0, got {workers_int}")
        parts += ["--workers", str(workers_int)]

    return " ".join(parts)


# =============================================================================
# Code staging
# =============================================================================

def _is_python_package_dir(d: Path) -> bool:
    return d.is_dir() and (d / "__init__.py").is_file()


def _copy_tree_excluding(src: Path, dst: Path, exclude_names: set, dry: bool) -> None:
    if dry:
        return
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in exclude_names:
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def stage_code_into_workdir(project_root: Path, work_dir: Path, dry: bool) -> None:
    exclude = {
        ".git", ".github",
        "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        ".venv", "venv", "env", "envs",
        "dist", "build", "node_modules",
        "training_outputs",
        "repro_artifacts_steps_1_7",
        "experiments_results",
    }

    ensure_dir(work_dir, dry)

    py_files = sorted(project_root.glob("*.py"))
    if not py_files:
        raise FileNotFoundError(f"No top-level *.py files found in {project_root.resolve()}")

    for f in py_files:
        dst = work_dir / f.name
        if not dry:
            shutil.copy2(f, dst)

    for d in sorted(project_root.iterdir()):
        if d.name in exclude:
            continue
        if _is_python_package_dir(d):
            _copy_tree_excluding(d, work_dir / d.name, exclude, dry)


def _make_run_dirs(results_parent: Path, timestamped: bool, dry: bool) -> Dict[str, Path]:
    del timestamped
    run_root = results_parent.resolve()
    ensure_dir(run_root, dry)

    report_path = (run_root / "run_report.txt").resolve()
    return {
        "run_root": run_root,
        "report_path": report_path,
    }


def _make_track_dirs(run_root: Path, track_name: str, dry: bool) -> Dict[str, Path]:
    track_root = (run_root / track_name).resolve()
    work_dir = (track_root / "work").resolve()
    archive_root = (work_dir / "repro_artifacts_steps_1_7").resolve()
    graphml_keep_root = (track_root / "kept_graphml").resolve()

    ensure_dir(track_root, dry)
    ensure_dir(work_dir, dry)
    ensure_dir(archive_root, dry)
    ensure_dir(graphml_keep_root, dry)

    return {
        "track_root": track_root,
        "work_dir": work_dir,
        "archive_root": archive_root,
        "graphml_keep_root": graphml_keep_root,
    }


def _default_pt_out_dir_for_track(run_root: Path, track_name: str) -> Path:
    return (run_root / track_name / "work" / "repro_artifacts_steps_1_7" / "pt_copies").resolve()


def _find_all_dirs(work_dir: Path, patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for pat in patterns:
        for candidate in sorted(work_dir.glob(pat)):
            if candidate.is_dir():
                rp = candidate.resolve()
                if rp not in seen:
                    seen.add(rp)
                    out.append(rp)
    return out


def _cleanup_transient_pt_files(
    *,
    work_dir: Path,
    pt_out_dir: Path,
    dry: bool,
    report: Optional[Report] = None,
) -> int:
    """
    Fast cleanup of transient PT-producing outputs while preserving resume-critical assets.

    Preserved on purpose:
      - canonical Step 2 *_pt_flat folders
      - synthetic_amr_graphs_train
      - synthetic_amr_graphs_test_frozen
      - synthetic_amr_graphs_test
      - archived outputs under repro_artifacts_steps_1_7

    Removed as whole transient directories when present:
      - pt_out_dir scratch copies
      - training_outputs* folders in work_dir
      - step5 ablation folder
      - step6 delay generated train folders
      - step6 frequency generated train folders
      - step7 sweep generated train folder

    Fallback:
      - if whole-directory removal fails for any transient root, delete only its .pt files
    """
    transient_roots: List[Path] = []

    transient_roots.append(pt_out_dir.resolve())
    transient_roots.extend(p.resolve() for p in work_dir.glob("training_outputs*") if p.is_dir())

    step5_root = work_dir / STEP5_ABLATION_REL
    if step5_root.exists() and step5_root.is_dir():
        transient_roots.append(step5_root.resolve())

    task_name = str(CONFIG.get("MODEL", {}).get("task", "")).strip()
    transient_roots.extend(_find_all_dirs(work_dir, _step_dataset_globs(task_name, "delay")))
    transient_roots.extend(_find_all_dirs(work_dir, _step_dataset_globs(task_name, "frequency")))
    transient_roots.extend(_find_all_dirs(work_dir, _step_dataset_globs(task_name, "sweep")))

    protected_roots = {
        (work_dir / _trajectory_pt_flat_dir(name)).resolve()
        for name in CANONICAL_TRAJECTORY_NAMES
    }
    protected_roots.update(
        {
            (work_dir / BASELINE_TRAIN_REL).resolve(),
            (work_dir / BASELINE_TEST_REL).resolve(),
            (work_dir / LIVE_TEST_REL).resolve(),
            (work_dir / "repro_artifacts_steps_1_7").resolve(),
        }
    )

    seen_roots = set()
    unique_roots: List[Path] = []
    for root in transient_roots:
        rp = root.resolve()
        if (
            rp.exists()
            and rp.is_dir()
            and rp not in seen_roots
            and rp not in protected_roots
            and not any(protected in rp.parents for protected in protected_roots)
        ):
            seen_roots.add(rp)
            unique_roots.append(rp)

    if dry:
        if report is not None:
            for root in unique_roots:
                report.write(f"[DRY_RUN] PT_CLEANUP_TRANSIENT remove_dir={root}")
        return len(unique_roots)

    removed_dirs = 0
    fallback_deleted_pts = 0

    if unique_roots and report is not None:
        report.start_subtask_console(
            "pt_cleanup_transient",
            total=len(unique_roots),
            extra="remove transient directories",
        )

    try:
        for idx, root in enumerate(unique_roots, start=1):
            try:
                shutil.rmtree(root)
                removed_dirs += 1
                if report is not None:
                    report.update_subtask_console(
                        current=idx,
                        total=len(unique_roots),
                        extra=f"removed {root.name}",
                    )
            except FileNotFoundError:
                if report is not None:
                    report.update_subtask_console(
                        current=idx,
                        total=len(unique_roots),
                        extra=f"already missing {root.name}",
                    )
                continue
            except Exception:
                pts = sorted(p for p in root.rglob("*.pt") if p.is_file())
                for pt_file in pts:
                    try:
                        pt_file.unlink()
                        fallback_deleted_pts += 1
                    except FileNotFoundError:
                        continue
                if report is not None:
                    report.update_subtask_console(
                        current=idx,
                        total=len(unique_roots),
                        extra=f"fallback .pt cleanup in {root.name}",
                    )
    finally:
        if unique_roots and report is not None:
            report.finish_subtask_console(force_complete=True)

    if report is not None:
        for root in unique_roots:
            report.write(f"PT_CLEANUP_TRANSIENT root={root}")
        report.write(
            "PT_CLEANUP_TRANSIENT "
            f"removed_dirs={removed_dirs} "
            f"fallback_deleted_pts={fallback_deleted_pts} "
            f"track_work_dir={work_dir}"
        )

    return removed_dirs + fallback_deleted_pts

# =============================================================================
# Label reading + gates
# =============================================================================

def _task_name_to_label_attr(task_name: str) -> Optional[str]:
    m = re.search(r"_h(\d+)$", str(task_name))
    if not m:
        return None
    h = m.group(1)
    if str(task_name).startswith("endogenous_transmission_majority_h"):
        return f"y_h{h}_trans_majority"
    if str(task_name).startswith("endogenous_importation_majority_h"):
        return f"y_h{h}_endog_majority"
    if str(task_name).startswith("early_outbreak_warning_h"):
        return f"y_h{h}_resistant_frac_cls"
    return None


def _preferred_label_attr_for_task(task_name: str) -> Optional[str]:
    return _task_name_to_label_attr(task_name)


def _read_label_from_pt(
    pt_path: Path,
    preferred_label_attr: Optional[str] = None,
    *,
    strict_preferred: bool = False,
) -> Optional[int]:
    import torch

    obj = torch.load(pt_path, map_location="cpu", weights_only=False)

    def _to_int(x: Any) -> Optional[int]:
        try:
            if hasattr(x, "item"):
                return int(x.item())
            return int(x)
        except Exception:
            return None

    preferred = str(preferred_label_attr).strip() if preferred_label_attr else None

    if strict_preferred and not preferred:
        return None

    if preferred:
        if hasattr(obj, preferred):
            v = _to_int(getattr(obj, preferred))
            if v in (0, 1):
                return v
        if strict_preferred:
            return None

    candidates: List[str] = []
    if hasattr(obj, "y_h7_trans_majority"):
        candidates.append("y_h7_trans_majority")
    if hasattr(obj, "y_h7_endog_majority") and "y_h7_endog_majority" not in candidates:
        candidates.append("y_h7_endog_majority")
    if hasattr(obj, "y"):
        candidates.append("y")

    for name in dir(obj):
        if name.startswith("y_") and name not in candidates:
            candidates.append(name)

    for name in candidates:
        try:
            x = getattr(obj, name)
        except Exception:
            continue
        v = _to_int(x)
        if v in (0, 1):
            return v

    return None


def label_stats_for_pt_folder(
    folder: Path,
    preferred_label_attr: Optional[str] = None,
    *,
    strict_preferred: bool = False,
) -> Dict[str, Any]:
    pts = sorted(folder.glob("*.pt"))
    zeros: List[Path] = []
    ones: List[Path] = []
    unknown: List[Path] = []

    for p in pts:
        lab = _read_label_from_pt(
            p,
            preferred_label_attr=preferred_label_attr,
            strict_preferred=strict_preferred,
        )
        if lab == 0:
            zeros.append(p)
        elif lab == 1:
            ones.append(p)
        else:
            unknown.append(p)

    return {
        "n_total": len(pts),
        "n0": len(zeros),
        "n1": len(ones),
        "n_unknown": len(unknown),
        "files0": zeros,
        "files1": ones,
        "files_unknown": unknown,
    }


def report_label_stats(
    report: Report,
    *,
    folder: Path,
    name: str,
    preferred_label_attr: Optional[str] = None,
    strict_preferred: bool = False,
) -> Dict[str, Any]:
    if not folder.exists():
        report.write(f"[{name}] folder={folder} (missing)")
        return {"n_total": 0, "n0": 0, "n1": 0, "n_unknown": 0}
    stats = label_stats_for_pt_folder(
        folder,
        preferred_label_attr=preferred_label_attr,
        strict_preferred=strict_preferred,
    )
    report.write(f"[{name}] folder={folder}")
    report.write(f"[{name}] n_total={stats['n_total']} n0={stats['n0']} n1={stats['n1']} unknown={stats['n_unknown']}")
    return stats


def assert_nonempty_known_labels_folder(
    folder: Path,
    *,
    name: str,
    report: Optional[Report] = None,
    preferred_label_attr: Optional[str] = None,
    strict_preferred: bool = False,
) -> Dict[str, Any]:
    if not folder.exists():
        raise FileNotFoundError(f"[{name}] folder not found: {folder.resolve()}")

    stats = label_stats_for_pt_folder(
        folder,
        preferred_label_attr=preferred_label_attr,
        strict_preferred=strict_preferred,
    )
    if report is not None:
        report.write(f"[{name}] n_total={stats['n_total']} n0={stats['n0']} n1={stats['n1']} unknown={stats['n_unknown']}")

    if stats["n_total"] == 0:
        raise RuntimeError(f"[{name}] no .pt files found in {folder.resolve()}")

    if stats["n_unknown"] > 0:
        ex = stats["files_unknown"][0]
        raise RuntimeError(
            f"[{name}] unknown label for {stats['n_unknown']} files (example: {ex.name}). Refusing to proceed."
        )

    return stats


def warn_if_single_label(stats: Dict[str, Any], *, name: str, report: Optional[Report] = None) -> None:
    if int(stats.get("n0", 0)) == 0 or int(stats.get("n1", 0)) == 0:
        msg = (
            f"WARNING: [{name}] TRAIN is single-label (n0={stats.get('n0', 0)}, n1={stats.get('n1', 0)}). "
            "Training will run; ROC/AUROC may be degenerate."
        )
        if report is not None:
            report.write(msg)
        else:
            print(msg, flush=True)


def assert_two_class_folder(
    folder: Path,
    *,
    name: str,
    require_balanced: bool,
    balance_tolerance: int = 1,
    report: Optional[Report] = None,
    preferred_label_attr: Optional[str] = None,
    strict_preferred: bool = False,
) -> Dict[str, Any]:
    if not folder.exists():
        raise FileNotFoundError(f"[{name}] folder not found: {folder.resolve()}")

    stats = label_stats_for_pt_folder(
        folder,
        preferred_label_attr=preferred_label_attr,
        strict_preferred=strict_preferred,
    )
    if report is not None:
        report.write(f"[{name}] n_total={stats['n_total']} n0={stats['n0']} n1={stats['n1']} unknown={stats['n_unknown']}")

    if stats["n_total"] == 0:
        raise RuntimeError(f"[{name}] no .pt files found in {folder.resolve()}")

    if stats["n_unknown"] > 0:
        ex = stats["files_unknown"][0]
        raise RuntimeError(
            f"[{name}] unknown label for {stats['n_unknown']} files (example: {ex.name}). Refusing to proceed."
        )

    if stats["n0"] == 0 or stats["n1"] == 0:
        raise RuntimeError(f"[{name}] single-label dataset: n0={stats['n0']} n1={stats['n1']} in {folder.resolve()}")

    if require_balanced and abs(stats["n0"] - stats["n1"]) > int(balance_tolerance):
        raise RuntimeError(
            f"[{name}] test folder not balanced: n0={stats['n0']} n1={stats['n1']} "
            f"(tolerance={balance_tolerance}) in {folder.resolve()}"
        )

    return stats


# =============================================================================
# PT metadata / contiguity helpers
# =============================================================================

def _to_int_maybe(x: Any) -> Optional[int]:
    try:
        if hasattr(x, "item"):
            return int(x.item())
        return int(x)
    except Exception:
        return None


def _extract_sim_id_and_day_from_pt_obj(pt_path: Path, obj: Any) -> Tuple[str, Optional[int]]:
    sim_id: Optional[str] = None
    for key in ["sim_id", "simulation_id", "trajectory_id", "traj_id", "sim", "trajectory", "run_id"]:
        if hasattr(obj, key):
            v = getattr(obj, key)
            if isinstance(v, str) and v.strip():
                sim_id = v.strip()
                break

    day: Optional[int] = None
    for key in ["day", "t", "time", "time_idx", "day_idx", "step", "step_idx"]:
        if hasattr(obj, key):
            v = _to_int_maybe(getattr(obj, key))
            if v is not None:
                day = int(v)
                break

    name = pt_path.name
    if sim_id is None:
        m_flat = re.search(r"^(.+__sim_\d+?)__(.+)$", name, flags=re.IGNORECASE)
        if m_flat:
            sim_id = m_flat.group(1)
    if sim_id is None:
        m = re.search(r"^(sim[^_]*?__[^_]+__[^.]+?)(?:__|_|-)(?:day|d)\d{1,4}\.pt$", name, flags=re.IGNORECASE)
        if m:
            sim_id = m.group(1)
        else:
            m2 = re.search(r"^(sim[^.]+?)(?:__amr|__|\.pt$)", name, flags=re.IGNORECASE)
            if m2:
                sim_id = m2.group(1)

    if sim_id is None:
        sim_id = f"{pt_path.parent.name}::{pt_path.stem}"

    if day is None:
        for pat in [
            r"(?:^|[_-]|__)(?:day)[_-]?(\d{1,4})(?:[_-]|__|\.|$)",
            r"(?:^|[_-]|__)(?:d)[_-]?(\d{1,4})(?:[_-]|__|\.|$)",
        ]:
            mm = re.search(pat, name, flags=re.IGNORECASE)
            if mm:
                day = int(mm.group(1))
                break

    return str(sim_id), day


def _read_sim_day(pt_path: Path) -> Tuple[str, int]:
    import torch

    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    sim_id, day = _extract_sim_id_and_day_from_pt_obj(pt_path, obj)
    if day is None:
        raise RuntimeError(
            f"Could not extract day index for pt: {pt_path.name} (sim_id='{sim_id}'). "
            "Cannot validate contiguity safely."
        )
    if int(day) <= 0:
        raise RuntimeError(f"Invalid day={day} for pt: {pt_path.name} (sim_id='{sim_id}')")
    return str(sim_id), int(day)


def assert_contiguous_days_per_sim(
    folder: Path,
    *,
    name: str,
    report: Optional[Report] = None,
) -> None:
    if not folder.exists():
        raise FileNotFoundError(f"[{name}] folder not found: {folder.resolve()}")

    pts = sorted(folder.glob("*.pt"))
    if len(pts) == 0:
        raise RuntimeError(f"[{name}] no .pt files found in {folder.resolve()}")

    sim_days: Dict[str, List[int]] = {}
    for p in pts:
        sim_id, day = _read_sim_day(p)
        sim_days.setdefault(sim_id, []).append(int(day))

    dup_bad: List[Tuple[str, int]] = []
    gap_bad: List[Tuple[str, int, int]] = []

    for sim_id, days in sim_days.items():
        ds_sorted = sorted(int(d) for d in days)
        for i in range(len(ds_sorted) - 1):
            if ds_sorted[i + 1] == ds_sorted[i]:
                dup_bad.append((sim_id, ds_sorted[i]))
                break

        ds = sorted(set(ds_sorted))
        for i in range(len(ds) - 1):
            if ds[i + 1] != ds[i] + 1:
                gap_bad.append((sim_id, ds[i], ds[i + 1]))
                break

    if dup_bad:
        sim_id, d = dup_bad[0]
        msg = (
            f"[{name}] Duplicate day entries detected for sim_id='{sim_id}', day={d}. "
            "This will confuse TemporalGraphDataset metadata grouping."
        )
        if report is not None:
            report.write(msg)
        raise RuntimeError(msg)

    if gap_bad:
        sim_id, d0, d1 = gap_bad[0]
        msg = (
            f"[{name}] Non-contiguous days detected for sim_id='{sim_id}': "
            f"... {d0}, {d1}, ... (gap={d1 - d0}). "
            "This will crash TemporalGraphDataset in metadata mode."
        )
        if report is not None:
            report.write(msg)
        raise RuntimeError(msg)

    if report is not None:
        report.write(f"[{name}] contiguity OK (per-sim day sequences are consecutive, no duplicates).")


# =============================================================================
# Canonical trajectory generation / PT preparation / frozen baseline build
# =============================================================================

def _extra_args_from_env(env_key: str) -> List[str]:
    s = str(os.environ.get(env_key, "")).strip()
    return shlex.split(s) if s else []


def _canonical_sim_dirs(work_dir: Path, trajectory_name: str) -> List[Path]:
    traj_dir = work_dir / trajectory_name
    return sorted([p for p in traj_dir.iterdir() if p.is_dir() and p.name.startswith("sim_")]) if traj_dir.exists() else []


def _trajectory_pt_flat_dir(trajectory_name: str) -> Path:
    return Path(f"{trajectory_name}_pt_flat")


def _global_sim_id_from_sim_dir(sim_dir: Path) -> str:
    return f"{sim_dir.parent.name}__{sim_dir.name}"


def _prepare_graph_folder_figures_compare_pool(
    *,
    src_roots: List[Path],
    dst_root: Path,
    dry: bool,
    report: Optional[Report] = None,
) -> None:
    if dry:
        if report is not None:
            report.write(f"[DRY_RUN] would prepare compare pool -> {dst_root}")
        return

    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    sim_counter = 0

    for src_root in src_roots:
        if not src_root.exists():
            raise FileNotFoundError(f"Missing raw graph source folder: {src_root.resolve()}")

        sim_dirs = sorted([p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("sim_")])
        if sim_dirs:
            for sim_dir in sim_dirs:
                dst_sim = dst_root / f"sim_{sim_counter:03d}__{_safe_tag(src_root.name)}"
                shutil.copytree(sim_dir, dst_sim)
                sim_counter += 1
                copied += 1
        else:
            for graph_file in sorted(src_root.rglob("*.graphml")):
                rel = graph_file.relative_to(src_root)
                dst_file = dst_root / _safe_tag(src_root.name) / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(graph_file, dst_file)
                copied += 1

            labels_dir = src_root / "labels"
            if labels_dir.exists() and labels_dir.is_dir():
                for label_file in sorted(labels_dir.rglob("*")):
                    if not label_file.is_file():
                        continue
                    rel = label_file.relative_to(src_root)
                    dst_file = dst_root / _safe_tag(src_root.name) / rel
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(label_file, dst_file)

    if report is not None:
        report.write(f"GRAPH_COMPARE_POOL prepared={dst_root} copied_entries={copied}")


def archive_step1_baseline_pair_figures(
    *,
    py: str,
    work_dir: Path,
    archive_root: Path,
    dry: bool,
    report: Optional[Report] = None,
    cwd: Optional[Path] = None,
    identity: str = "Harry Triantafyllidis",
    enable_graph_folder_figures: bool = False,
) -> Optional[Path]:
    train_sources = [work_dir / name for name in CANONICAL_TRAIN_TRAJECTORIES]
    test_sources = [work_dir / name for name in CANONICAL_TEST_TRAJECTORIES]

    if not all(_has_graphml_files(p) for p in train_sources + test_sources):
        if report is not None:
            report.write(
                "SKIP step1 baseline pair figures: missing raw GraphML under one or more canonical trajectory folders."
            )
        return None

    tmp_root = work_dir / "_tmp_step1_baseline_graph_compare"
    train_pool = tmp_root / "baseline_train_raw"
    test_pool = tmp_root / "baseline_test_frozen_raw"
    out_dir = archive_root / DATASET_FIGURES_REL / "step1_baseline_train_vs_test"

    try:
        _prepare_graph_folder_figures_compare_pool(
            src_roots=train_sources,
            dst_root=train_pool,
            dry=dry,
            report=report,
        )
        _prepare_graph_folder_figures_compare_pool(
            src_roots=test_sources,
            dst_root=test_pool,
            dry=dry,
            report=report,
        )

        archive_dataset_pair_figures(
            py=py,
            graph_dir=train_pool,
            compare_dir=test_pool,
            dst_dir=out_dir,
            dry=dry,
            report=report,
            cwd=cwd,
            identity=identity,
            title="Step 1 baseline raw-graph summary: train vs frozen test",
            label="baseline_train",
            compare_label="baseline_test_frozen",
            enable_graph_folder_figures=enable_graph_folder_figures,
        )
    finally:
        if not dry and tmp_root.exists():
            shutil.rmtree(tmp_root)

    return out_dir


def ensure_step1_baseline_pair_figures_before_purge(
    *,
    py: str,
    work_dir: Path,
    archive_root: Path,
    dry: bool,
    report: Optional[Report] = None,
    cwd: Optional[Path] = None,
    identity: str = "Harry Triantafyllidis",
    enable_graph_folder_figures: bool = False,
) -> Optional[Path]:
    if not enable_graph_folder_figures:
        if report is not None:
            report.write("STEP1 baseline pair figures disabled by CLI")
        return None

    out_dir = archive_root / DATASET_FIGURES_REL / "step1_baseline_train_vs_test"

    if out_dir.exists() and any(out_dir.glob("*.png")):
        if report is not None:
            report.write(f"STEP1 baseline pair figures already present -> {out_dir}")
        return out_dir

    return archive_step1_baseline_pair_figures(
        py=py,
        work_dir=work_dir,
        archive_root=archive_root,
        dry=dry,
        report=report,
        cwd=cwd,
        identity=identity,
        enable_graph_folder_figures=enable_graph_folder_figures,
    )


def keep_step1_baseline_graphml_before_purge(
    *,
    work_dir: Path,
    graphml_keep_root: Path,
    dry: bool,
    report: Optional[Report] = None,
) -> None:
    train_sources = [work_dir / name for name in CANONICAL_TRAIN_TRAJECTORIES]
    test_sources = [work_dir / name for name in CANONICAL_TEST_TRAJECTORIES]

    keep_graphml_roots(
        src_roots=train_sources,
        dst_root=graphml_keep_root / "step4_baseline_train",
        dry=dry,
        report=report,
        label="step4_baseline_train",
    )
    keep_graphml_roots(
        src_roots=test_sources,
        dst_root=graphml_keep_root / "frozen_test_once",
        dry=dry,
        report=report,
        label="frozen_test_once",
    )


def _run_generate_amr_sim(
    *,
    py: str,
    out_dir: Path,
    seed: int,
    num_days: int,
    p_cs: float,
    p_cr: float,
    discharge_frac: float,
    discharge_min: int,
    extra_cli_args: Optional[List[str]],
    dry: bool,
    cwd: Path,
    report: Report,
) -> None:
    cmd: List[str] = [
        py,
        "generate_amr_data.py",
        "--output_dir", str(out_dir),
        "--seed", str(int(seed)),
        "--num_days", str(int(num_days)),
        "--daily_discharge_frac", str(float(discharge_frac)),
        "--daily_discharge_min_per_ward", str(int(discharge_min)),
        "--p_admit_import_cs", str(float(p_cs)),
        "--p_admit_import_cr", str(float(p_cr)),
    ]
    cmd += _extra_args_from_env("DT_SIM_EXTRA_ARGS")
    if extra_cli_args:
        cmd += [str(x) for x in extra_cli_args]
    run_cmd(cmd, dry=dry, cwd=cwd, report=report, stream_output=False)


def step1_generate_canonical_trajectories(*, work_dir: Path, py: str, dry: bool, report: Report) -> None:
    cfg = CONFIG["STEP1"]
    n_sims = int(cfg["n_sims_per_trajectory"])
    num_days = int(cfg["num_days"])
    traj_cfg: Dict[str, Dict[str, Any]] = dict(cfg["trajectories"])

    missing = [name for name in CANONICAL_TRAJECTORY_NAMES if name not in traj_cfg]
    if missing:
        raise RuntimeError(f"STEP1 config missing canonical trajectories: {missing}")

    total_jobs = len(CANONICAL_TRAJECTORY_NAMES) * n_sims
    done = 0

    for name in CANONICAL_TRAJECTORY_NAMES:
        spec = traj_cfg[name]
        traj_dir = work_dir / name
        report.kv(
            f"trajectory::{name}",
            {
                "dir": str(traj_dir),
                "seed_base": int(spec["seed_base"]),
                "p_admit_import_cs": float(spec["p_admit_import_cs"]),
                "p_admit_import_cr": float(spec["p_admit_import_cr"]),
                "daily_discharge_frac": float(spec["daily_discharge_frac"]),
                "daily_discharge_min_per_ward": int(spec["daily_discharge_min_per_ward"]),
                "extra_sim_args": list(spec.get("extra_sim_args", [])),
                "n_sims": n_sims,
                "num_days": num_days,
            },
        )

        if not dry and traj_dir.exists():
            shutil.rmtree(traj_dir)
        ensure_dir(traj_dir, dry)

        for r in range(n_sims):
            sim_dir = traj_dir / f"sim_{r:03d}"
            ensure_dir(sim_dir, dry)

            _report_progress(
                report,
                prefix="STEP1 progress",
                current=done + 1,
                total=total_jobs,
                extra=f"{name} / sim_{r:03d}",
            )

            _run_generate_amr_sim(
                py=py,
                out_dir=sim_dir,
                seed=int(spec["seed_base"]) + r,
                num_days=num_days,
                p_cs=float(spec["p_admit_import_cs"]),
                p_cr=float(spec["p_admit_import_cr"]),
                discharge_frac=float(spec["daily_discharge_frac"]),
                discharge_min=int(spec["daily_discharge_min_per_ward"]),
                extra_cli_args=list(spec.get("extra_sim_args", [])),
                dry=dry,
                cwd=work_dir,
                report=report,
            )
            done += 1


def _task_uses_fixed_early_outbreak_threshold(task_name: str) -> bool:
    return _task_family(task_name) == "early_outbreak_warning"


def _ensure_task_specific_convert_support_files(*, work_dir: Path, dry: bool, report: Optional[Report] = None) -> List[str]:
    task_name = str(CONFIG["MODEL"].get("task", "")).strip()
    extra_args: List[str] = []

    if _task_uses_fixed_early_outbreak_threshold(task_name):
        threshold_value = float(CONFIG["MODEL"].get("early_outbreak_fixed_threshold", 0.55))
        threshold_path = work_dir / "early_outbreak_warning_threshold.json"
        payload = {
            "h14_resistant_frac_threshold": float(threshold_value),
            "threshold": float(threshold_value),
            "task": str(task_name),
        }

        if not dry:
            threshold_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        if report is not None:
            report.write(
                f"EARLY_OUTBREAK_THRESHOLD file={threshold_path} value={float(threshold_value):.6f} task={task_name}"
            )

        extra_args += [
            "--early_res_frac_threshold", str(float(threshold_value)),
            "--early_res_frac_threshold_file", str(threshold_path),
            "--early_res_frac_threshold_out", str(threshold_path),
        ]

    return extra_args


def _run_convert_one_sim(*, py: str, sim_dir: Path, dry: bool, cwd: Path, report: Report) -> None:
    cmd: List[str] = [
        py,
        "convert_to_pt.py",
        "--graphml_dir", str(sim_dir),
        "--label_csv_dir", str(sim_dir / "labels"),
    ]
    cmd += _extra_args_from_env("DT_CONVERT_EXTRA_ARGS")
    cmd += _ensure_task_specific_convert_support_files(work_dir=cwd, dry=dry, report=report)
    run_cmd(cmd, dry=dry, cwd=cwd, report=report, stream_output=False)


def _collect_flat_pt_from_sim(*, sim_dir: Path, out_dir: Path, dry: bool, report: Optional[Report] = None) -> int:
    import torch

    pts = sorted(sim_dir.glob("*.pt"))
    if len(pts) == 0:
        raise RuntimeError(f"No .pt files found after conversion in {sim_dir.resolve()}")
    n = 0
    if dry:
        return len(pts)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_sim_id = _global_sim_id_from_sim_dir(sim_dir)

    for pt in pts:
        obj = torch.load(pt, map_location="cpu", weights_only=False)
        setattr(obj, "sim_id", global_sim_id)
        new_name = f"{sim_dir.parent.name}__{sim_dir.name}__{pt.name}"
        torch.save(obj, out_dir / new_name)
        n += 1

    if report is not None:
        report.write(f"FLATTEN_PT sim={sim_dir} copied={n} out={out_dir} global_sim_id={global_sim_id}")
    return n


def step2_convert_and_flatten_canonical_trajectories(
    *,
    work_dir: Path,
    py: str,
    dry: bool,
    report: Report,
    progress_start: int = 0,
    progress_total: Optional[int] = None,
    enable_graph_folder_figures: bool = False,
) -> List[Path]:
    out_dirs: List[Path] = []
    default_total_jobs = len(CANONICAL_TRAJECTORY_NAMES) * (int(CONFIG["STEP1"]["n_sims_per_trajectory"]) + 2)
    total_jobs = int(progress_total) if progress_total is not None else default_total_jobs
    done = int(progress_start)

    for name in CANONICAL_TRAJECTORY_NAMES:
        traj_dir = work_dir / name
        if dry:
            out_dir = work_dir / _trajectory_pt_flat_dir(name)
            out_dirs.append(out_dir)
            report.write(f"[DRY_RUN] would convert+flatten trajectory {traj_dir} -> {out_dir}")
            done += int(CONFIG["STEP1"]["n_sims_per_trajectory"]) + 2
            continue

        if not traj_dir.exists():
            raise FileNotFoundError(f"Missing Step 1 trajectory folder: {traj_dir.resolve()}")

        sim_dirs = _canonical_sim_dirs(work_dir, name)
        expected = int(CONFIG["STEP1"]["n_sims_per_trajectory"])
        if len(sim_dirs) != expected:
            raise RuntimeError(
                f"Trajectory {name} expected {expected} sim_* folders, found {len(sim_dirs)} in {traj_dir.resolve()}"
            )

        out_dir = work_dir / _trajectory_pt_flat_dir(name)
        out_dirs.append(out_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        ensure_dir(out_dir, dry)

        total_pt = 0
        for sim_dir in sim_dirs:
            _report_progress(
                report,
                prefix="STEP2 progress",
                current=done + 1,
                total=total_jobs,
                extra=f"convert {name} / {sim_dir.name}",
                target="step",
            )
            _run_convert_one_sim(py=py, sim_dir=sim_dir, dry=dry, cwd=work_dir, report=report)
            total_pt += _collect_flat_pt_from_sim(sim_dir=sim_dir, out_dir=out_dir, dry=dry, report=report)
            done += 1

        _report_progress(
            report,
            prefix="STEP2 progress",
            current=done + 1,
            total=total_jobs,
            extra=f"figures {name}",
            target="step",
        )
        archive_single_dataset_figures(
            py=py,
            graph_dir=traj_dir,
            dst_dir=work_dir / "repro_artifacts_steps_1_7" / DATASET_FIGURES_REL / "step2_canonical" / _safe_tag(name),
            dry=dry,
            report=report,
            cwd=work_dir,
            identity="Harry Triantafyllidis",
            title=f"Step 2 canonical dataset: {name}",
            label=_safe_tag(name),
            enable_graph_folder_figures=enable_graph_folder_figures,
        )
        done += 1

        _report_progress(
            report,
            prefix="STEP2 progress",
            current=done + 1,
            total=total_jobs,
            extra=f"purge {name}",
            target="step",
        )
        _purge_graphml_under(
            traj_dir,
            dry=dry,
            report=report,
            label=f"step2::{name}::after_dataset_figures",
        )
        done += 1

        if not dry:
            _assert_canonical_trajectory_expected_label_fraction(
                folder=out_dir,
                task_name=str(CONFIG["MODEL"]["task"]).strip(),
                trajectory_name=name,
                report=report,
            )

        report.write(f"STEP2_TRAJECTORY_DONE name={name} sims={len(sim_dirs)} total_pt={total_pt} out={out_dir}")

    return out_dirs

def _find_latest_dir(work_dir: Path, patterns: List[str]) -> Optional[Path]:
    candidates: List[Path] = []
    for pat in patterns:
        for candidate in work_dir.glob(pat):
            if candidate.is_dir():
                candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(key=lambda pp: pp.stat().st_mtime, reverse=True)
    return candidates[0]


def _pool_pt_folders(
    *,
    src_folders: List[Path],
    dst_folder: Path,
    dry: bool,
    report: Optional[Report] = None,
) -> None:
    if dry:
        if report is not None:
            report.write(f"[DRY_RUN] would pool PT folders into {dst_folder}")
        return

    if dst_folder.exists():
        shutil.rmtree(dst_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)

    all_pts: List[Tuple[Path, Path]] = []
    for src in src_folders:
        if not src.exists():
            raise FileNotFoundError(f"Missing source folder: {src.resolve()}")
        for pt_path in sorted(src.glob("*.pt")):
            all_pts.append((src, pt_path))

    total = len(all_pts)
    copied = 0

    if total > 0 and report is not None:
        report.start_subtask_console(f"_pool_pt_folders:{dst_folder.name}", total=total, extra=str(dst_folder))

    manifest = dst_folder / "manifest.csv"
    try:
        with manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pooled_file", "source_folder", "source_file"])
            for src, pt_path in all_pts:
                actual_name = _copy_with_collision_guard(pt_path, dst_folder, source_hint=src)
                w.writerow([actual_name, src.as_posix(), pt_path.name])
                copied += 1
                if report is not None:
                    report.update_subtask_console(current=copied, total=total, extra=str(dst_folder))
    finally:
        if total > 0 and report is not None:
            report.finish_subtask_console(force_complete=True)

    if report is not None:
        report.write(f"POOLED_PT total={total} out={dst_folder}")

def _task_requires_two_class_pooled_train(task_name: str) -> bool:
    family = _task_family(task_name)
    if family == "early_outbreak_warning":
        return bool(CONFIG["STEP1"].get("outbreak_require_two_class_pooled_baseline", True))
    if family == "mechanism_split":
        return True
    return False


def _canonical_expected_label_for_trajectory(task_name: str, trajectory_name: str) -> Optional[int]:
    if _task_family(task_name) != "early_outbreak_warning":
        return None

    name = str(trajectory_name).strip().lower()
    if name.startswith("outbreak_low_"):
        return 0
    if name.startswith("outbreak_high_"):
        return 1
    return None


def _assert_canonical_trajectory_expected_label_fraction(
    *,
    folder: Path,
    task_name: str,
    trajectory_name: str,
    report: Optional[Report] = None,
) -> Dict[str, Any]:
    expected_label = _canonical_expected_label_for_trajectory(task_name, trajectory_name)
    preferred_label_attr = _preferred_label_attr_for_task(task_name)
    if expected_label is None or preferred_label_attr is None:
        return label_stats_for_pt_folder(folder, preferred_label_attr=preferred_label_attr, strict_preferred=True)

    stats = assert_nonempty_known_labels_folder(
        folder,
        name=f"Canonical trajectory {trajectory_name}",
        report=report,
        preferred_label_attr=preferred_label_attr,
        strict_preferred=True,
    )

    n_expected = int(stats["n0"] if expected_label == 0 else stats["n1"])
    n_total = int(stats["n_total"])
    frac_expected = 0.0 if n_total <= 0 else float(n_expected) / float(n_total)
    min_frac = float(CONFIG["STEP1"].get("outbreak_expected_label_min_frac", 0.3))

    if report is not None:
        report.write(
            f"[Canonical trajectory {trajectory_name}] expected_label={expected_label} "
            f"n_total={n_total} n_expected={n_expected} frac_expected={frac_expected:.4f} "
            f"required_min_frac={min_frac:.4f}"
        )

    if frac_expected < min_frac:
        raise RuntimeError(
            f"[Canonical trajectory {trajectory_name}] expected outbreak label dominance not achieved. "
            f"Expected label {expected_label} fraction={frac_expected:.4f}, required >= {min_frac:.4f}. "
            "Tighten or further separate the Step 1 outbreak canonical regimes before training."
        )

    return stats


def build_frozen_baseline_from_canonical_trajectories(
    *,
    work_dir: Path,
    train_folder: Path,
    test_folder: Path,
    dry: bool,
    report: Report,
    preferred_label_attr: Optional[str] = None,
) -> None:
    if preferred_label_attr is None:
        raise ValueError(
            f"No supported preferred label attribute could be inferred for task '{CONFIG['MODEL']['task']}'. "
            "Refusing to build baseline with ambiguous labels."
        )

    train_srcs = [work_dir / _trajectory_pt_flat_dir(name) for name in CANONICAL_TRAIN_TRAJECTORIES]
    test_srcs = [work_dir / _trajectory_pt_flat_dir(name) for name in CANONICAL_TEST_TRAJECTORIES]
    require_balanced_test = bool(CONFIG["TEST"].get("require_balanced_test", False))

    report.section("BUILD FROZEN BASELINE FROM CANONICAL TRAJECTORIES")
    for src in train_srcs + test_srcs:
        report.kv(f"source::{src.name}", src)
        if not dry and not src.exists():
            raise FileNotFoundError(f"Missing Step 2 PT-flat folder: {src.resolve()}")

    _pool_pt_folders(src_folders=train_srcs, dst_folder=train_folder, dry=dry, report=report)
    _pool_pt_folders(src_folders=test_srcs, dst_folder=test_folder, dry=dry, report=report)

    if dry:
        return

    if _task_requires_two_class_pooled_train(str(CONFIG["MODEL"]["task"]).strip()):
        st_train = assert_two_class_folder(
            train_folder,
            name="Frozen BASELINE TRAIN",
            require_balanced=False,
            balance_tolerance=int(CONFIG["TEST"].get("balance_tolerance", 1)),
            report=report,
            preferred_label_attr=preferred_label_attr,
            strict_preferred=True,
        )
    else:
        st_train = assert_nonempty_known_labels_folder(
            train_folder,
            name="Frozen BASELINE TRAIN",
            report=report,
            preferred_label_attr=preferred_label_attr,
            strict_preferred=True,
        )
        warn_if_single_label(st_train, name="Frozen BASELINE TRAIN", report=report)
    assert_contiguous_days_per_sim(train_folder, name="Frozen BASELINE TRAIN", report=report)

    assert_two_class_folder(
        test_folder,
        name="Frozen BASELINE TEST",
        require_balanced=require_balanced_test,
        balance_tolerance=int(CONFIG["TEST"].get("balance_tolerance", 1)),
        report=report,
        preferred_label_attr=preferred_label_attr,
        strict_preferred=True,
    )
    assert_contiguous_days_per_sim(test_folder, name="Frozen BASELINE TEST", report=report)


def _restore_live_test_from_baseline(
    *,
    work_dir: Path,
    dry: bool,
    report: Optional[Report] = None,
) -> None:
    src = work_dir / BASELINE_TEST_REL
    dst = work_dir / LIVE_TEST_REL

    if dry:
        if report is not None:
            report.write(f"[DRY_RUN] would mirror baseline test {src} -> {dst}")
        return

    if not src.exists():
        raise FileNotFoundError(f"Missing baseline test folder: {src.resolve()}")

    if src.resolve() == dst.resolve():
        if report is not None:
            report.write(f"LIVE TEST already equals frozen baseline test: {src}")
        return

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    if report is not None:
        report.write(f"RESTORED live test folder from baseline: {src} -> {dst}")


def _require_pipeline_prereqs(
    *,
    work_dir: Path,
    start_v: float,
    report: Report,
) -> None:
    if start_v <= 1.0:
        return

    checks: List[Tuple[str, Path]] = []
    if start_v > 1.0:
        checks.extend([(name, work_dir / name) for name in CANONICAL_TRAJECTORY_NAMES])
    if start_v > 2.0:
        checks.extend([(f"{name}_pt_flat", work_dir / _trajectory_pt_flat_dir(name)) for name in CANONICAL_TRAJECTORY_NAMES])
    if start_v > 3.0:
        checks.extend([
            (str(BASELINE_TRAIN_REL), work_dir / BASELINE_TRAIN_REL),
            (str(BASELINE_TEST_REL), work_dir / BASELINE_TEST_REL),
        ])

    missing = [(label, p) for label, p in checks if not p.exists()]
    if missing:
        msg_lines = [
            f"Cannot start at step {start_v} because prerequisite artifacts are missing from this persistent work_dir.",
            "Missing prerequisites:",
        ]
        msg_lines.extend([f"- {label}: {p}" for label, p in missing[:20]])
        msg_lines.append("Run earlier steps first in the same experiments_results root, or restart from step 1 for this track.")
        msg = "\n".join(msg_lines)
        report.write(msg)
        raise FileNotFoundError(msg)


# =============================================================================
# Sanity / runtime consistency helpers
# =============================================================================

def _validate_step1_length_compatibility(
    *,
    report: Report,
    horizons: List[int],
    T_list: List[int],
) -> None:
    num_days = int(CONFIG["STEP1"]["num_days"])
    max_h = max(int(h) for h in horizons) if horizons else int(CONFIG["MODEL"].get("pred_horizon", 7))
    max_t = max(int(t) for t in T_list) if T_list else int(CONFIG["MODEL"].get("T", 7))

    required_min_days = max_t + max_h
    report.kv("step1_num_days", num_days)
    report.kv("max_requested_T", max_t)
    report.kv("max_requested_horizon", max_h)
    report.kv("required_min_days_sanity", required_min_days)

    if num_days < required_min_days:
        raise RuntimeError(
            "STEP1 num_days is too small for the requested runtime configuration. "
            f"Got num_days={num_days}, but max(T)+max(horizon)={required_min_days}. "
            "Increase CONFIG['STEP1']['num_days'] or reduce T / horizons."
        )


# =============================================================================
# Step-specific folder validators
# =============================================================================

def validate_train_folder_for_task(
    *,
    folder: Path,
    task_name: str,
    folder_name: str,
    report: Report,
) -> Dict[str, Any]:
    preferred = _preferred_label_attr_for_task(task_name)
    if preferred is None:
        raise ValueError(f"Unsupported task for strict TRAIN validation: {task_name}")

    stats = report_label_stats(
        report,
        folder=folder,
        name=f"{folder_name} [{task_name}]",
        preferred_label_attr=preferred,
        strict_preferred=True,
    )
    stats = assert_nonempty_known_labels_folder(
        folder,
        name=f"{folder_name} [{task_name}]",
        report=report,
        preferred_label_attr=preferred,
        strict_preferred=True,
    )
    warn_if_single_label(stats, name=f"{folder_name} [{task_name}]", report=report)
    assert_contiguous_days_per_sim(
        folder,
        name=f"{folder_name} [{task_name}] (contiguity gate)",
        report=report,
    )
    return stats


def _make_baseline_test_checker(*, work_dir: Path, tol: int, report: Report) -> Callable[[str], None]:
    def _checker(task_name: str) -> None:
        preferred = _preferred_label_attr_for_task(task_name)
        if preferred is None:
            raise ValueError(f"Unsupported task for strict label checking: {task_name}")
        live_test = work_dir / LIVE_TEST_REL
        require_balanced_test = bool(CONFIG["TEST"].get("require_balanced_test", False))
        report_label_stats(
            report,
            folder=live_test,
            name=f"TEST CHECK [{task_name}]",
            preferred_label_attr=preferred,
            strict_preferred=True,
        )
        assert_two_class_folder(
            live_test,
            name=f"TEST CHECK [{task_name}]",
            require_balanced=require_balanced_test,
            balance_tolerance=tol,
            report=report,
            preferred_label_attr=preferred,
            strict_preferred=True,
        )
        assert_contiguous_days_per_sim(
            live_test,
            name=f"TEST CHECK [{task_name}] (contiguity gate)",
            report=report,
        )
    return _checker


def _validate_baseline_for_all_requested_horizons(
    *,
    work_dir: Path,
    base_task: str,
    horizons: List[int],
    run_all_horizons: bool,
    report: Report,
    tol: int,
) -> None:
    test_folder = work_dir / BASELINE_TEST_REL
    require_balanced_test = bool(CONFIG["TEST"].get("require_balanced_test", False))

    if not run_all_horizons:
        task_eff = base_task
        preferred = _preferred_label_attr_for_task(task_eff)
        if preferred is None:
            raise ValueError(f"Unsupported task for strict baseline validation: {task_eff}")
        assert_two_class_folder(
            test_folder,
            name=f"Baseline TEST [{task_eff}]",
            require_balanced=require_balanced_test,
            balance_tolerance=tol,
            report=report,
            preferred_label_attr=preferred,
            strict_preferred=True,
        )
        assert_contiguous_days_per_sim(
            test_folder,
            name=f"Baseline TEST [{task_eff}] (contiguity gate)",
            report=report,
        )
        return

    for h_val in horizons:
        task_eff = _task_with_horizon(base_task, int(h_val))
        preferred = _preferred_label_attr_for_task(task_eff)
        if preferred is None:
            raise ValueError(f"Unsupported task for strict baseline validation: {task_eff}")
        assert_two_class_folder(
            test_folder,
            name=f"Baseline TEST [{task_eff}]",
            require_balanced=require_balanced_test,
            balance_tolerance=tol,
            report=report,
            preferred_label_attr=preferred,
            strict_preferred=True,
        )
        assert_contiguous_days_per_sim(
            test_folder,
            name=f"Baseline TEST [{task_eff}] (contiguity gate)",
            report=report,
        )


# =============================================================================
# Multi-(T,H) training orchestrator
# =============================================================================


def run_training_step(
    *,
    step_tag: str,
    data_folder: Path,
    work_dir: Path,
    archive_root: Path,
    py: str,
    dry: bool,
    report: Report,
    horizons: List[int],
    run_all_horizons: bool,
    T_list: List[int],
    run_all_T: bool,
    archive_train_test_folders: bool,
    ensure_test_folder_for_task: Optional[Callable[[str], None]] = None,
    eval_test_folder: Path = LIVE_TEST_REL,
    test_folder_for_archive: Path = LIVE_TEST_REL,
    no_train: bool = False,
) -> None:
    tag = _safe_tag(step_tag)

    if run_all_T:
        if len(T_list) == 0:
            raise RuntimeError("run_all_T enabled but T_list is empty.")
        ts_vals = [int(t) for t in T_list]
    else:
        ts_vals = [int(CONFIG["MODEL"]["T"])]

    if run_all_horizons:
        if len(horizons) == 0:
            raise RuntimeError("run_all_horizons enabled but horizons list is empty.")
        hs_vals = [int(h) for h in horizons]
    else:
        hs_vals = []

    base_task = str(CONFIG["MODEL"]["task"]).strip()
    total_jobs = len(ts_vals) * (len(hs_vals) if run_all_horizons else 1)
    done = 0

    report.start_subtask_console(f"run_training_step:{tag}", total=total_jobs, extra=str(data_folder))

    try:
        for t_val in ts_vals:
            horizon_iter: List[Optional[int]] = hs_vals if run_all_horizons else [None]

            for h_val in horizon_iter:
                task_eff = _task_with_horizon(base_task, int(h_val)) if h_val is not None else base_task

                dst = _archive_dst_for_training_run(
                    archive_root=archive_root,
                    step_tag=tag,
                    t_val=int(t_val),
                    h_val=h_val,
                    run_all_T=run_all_T,
                    run_all_horizons=run_all_horizons,
                )

                if no_train:
                    data_folder_abs = data_folder if data_folder.is_absolute() else (work_dir / data_folder)
                    if not data_folder_abs.exists():
                        archived_train_folder = dst / "train_folder"
                        if archived_train_folder.exists():
                            data_folder = archived_train_folder
                            data_folder_abs = archived_train_folder
                        else:
                            raise FileNotFoundError(
                                f"No-train repair could not find train folder for step '{tag}'. "
                                f"Tried live folder: {data_folder_abs.resolve()} and archived fallback: {archived_train_folder.resolve()}"
                            )
                    out_dir_rel = dst.relative_to(work_dir)
                else:
                    if h_val is None:
                        out_dir_rel = Path(f"training_outputs_{tag}_T{int(t_val)}") if run_all_T else Path(TRAINING_OUT_REL)
                    else:
                        out_dir_rel = Path(f"training_outputs_{tag}_T{int(t_val)}_h{int(h_val)}")

                extra_bits = [f"T={int(t_val)}"]
                if h_val is not None:
                    extra_bits.append(f"h={int(h_val)}")
                extra_bits.append("eval_only" if no_train else "train_eval")
                _report_progress(
                    report,
                    prefix=f"TRAIN {tag}",
                    current=done + 1,
                    total=total_jobs,
                    extra=" ".join(extra_bits),
                    target="subtask",
                )

                if not dry:
                    validate_train_folder_for_task(
                        folder=work_dir / data_folder,
                        task_name=task_eff,
                        folder_name=f"TRAIN CHECK {tag}",
                        report=report,
                    )

                if ensure_test_folder_for_task is not None:
                    ensure_test_folder_for_task(task_eff)

                if no_train and not dry:
                    if not dst.exists():
                        raise FileNotFoundError(
                            f"Evaluation-only repair requested, but archived step folder is missing: {dst.resolve()}"
                        )
                    model_path = dst / "trained_model.pt"
                    if not model_path.exists():
                        raise FileNotFoundError(
                            f"Evaluation-only repair requested, but trained_model.pt is missing in: {dst.resolve()}"
                        )

                train_args = _build_train_args_from_config(
                    CONFIG["MODEL"],
                    py,
                    data_folder=data_folder,
                    work_dir=work_dir,
                    task_override=task_eff,
                    out_dir=str(out_dir_rel),
                    T_override=int(t_val),
                    test_folder=eval_test_folder,
                    train_model_override=(False if no_train else None),
                )
                run_cmd(train_args, dry, cwd=work_dir, report=report, stream_output=True)

                if not no_train:
                    archive_training_outputs(dst, work_dir=work_dir, dry=dry, report=report, src_dir=out_dir_rel)

                    if archive_train_test_folders:
                        archive_dataset_folder(
                            dst / "train_folder",
                            work_dir=work_dir,
                            folder=data_folder,
                            dry=dry,
                            report=report,
                            label="train_folder",
                        )
                        archive_dataset_folder(
                            dst / "test_folder",
                            work_dir=work_dir,
                            folder=test_folder_for_archive,
                            dry=dry,
                            report=report,
                            label="test_folder",
                        )

                done += 1
    finally:
        report.finish_subtask_console(force_complete=True)

# =============================================================================
# Pipeline
# =============================================================================

def run_pipeline_once(
    *,
    work_dir: Path,
    archive_root: Path,
    graphml_keep_root: Path,
    py: str,
    dry: bool,
    start_v: float,
    stop_v: float,
    archive_train_test_folders: bool,
    test_frac_per_class: float,
    report: Report,
    horizons: List[int],
    run_all_horizons: bool,
    T_list: List[int],
    run_all_T: bool,
    enable_graph_folder_figures: bool,
    keep_step_train_graphml: bool,
    no_train: bool,
) -> None:
    del test_frac_per_class
    ensure_dir(archive_root, dry)

    _require_pipeline_prereqs(work_dir=work_dir, start_v=start_v, report=report)

    tol = int(CONFIG["TEST"].get("balance_tolerance", 1))
    require_balanced_test = bool(CONFIG["TEST"].get("require_balanced_test", False))
    base_task = str(CONFIG["MODEL"]["task"]).strip()
    preferred_label_attr = _preferred_label_attr_for_task(base_task)
    if preferred_label_attr is None:
        raise ValueError(f"Unsupported task for strict experiments pipeline: {base_task}")
    t_needed_runtime = _get_max_T_needed_runtime(T_list)

    report.section("PIPELINE SETTINGS")
    report.kv("T_needed_runtime_for_baseline", t_needed_runtime)
    report.kv("keep_step_train_graphml", int(bool(keep_step_train_graphml)))
    report.kv("graphml_keep_root", graphml_keep_root)
    report.kv("no_train", int(bool(no_train)))
    _validate_step1_length_compatibility(report=report, horizons=horizons, T_list=T_list)

    if start_v <= 1.0 <= stop_v:
        step1_total = len(CANONICAL_TRAJECTORY_NAMES) * int(CONFIG["STEP1"]["n_sims_per_trajectory"])
        report.section("STEP 1", total=step1_total)
        step1_generate_canonical_trajectories(work_dir=work_dir, py=py, dry=dry, report=report)
        report.finish_step_console(force_complete=True)

    if start_v <= 2.0 <= stop_v:
        step2_total = 1 + len(CANONICAL_TRAJECTORY_NAMES) * (int(CONFIG["STEP1"]["n_sims_per_trajectory"]) + 2)
        report.section("STEP 2", total=step2_total)

        _report_progress(
            report,
            prefix="STEP2 progress",
            current=1,
            total=step2_total,
            extra="baseline raw figures",
            target="step",
        )

        ensure_step1_baseline_pair_figures_before_purge(
            py=py,
            work_dir=work_dir,
            archive_root=archive_root,
            dry=dry,
            report=report,
            cwd=work_dir,
            identity="Harry Triantafyllidis",
            enable_graph_folder_figures=enable_graph_folder_figures,
        )
        if keep_step_train_graphml:
            keep_step1_baseline_graphml_before_purge(
                work_dir=work_dir,
                graphml_keep_root=graphml_keep_root,
                dry=dry,
                report=report,
            )

        step2_convert_and_flatten_canonical_trajectories(
            work_dir=work_dir,
            py=py,
            dry=dry,
            report=report,
            progress_start=1,
            progress_total=step2_total,
            enable_graph_folder_figures=enable_graph_folder_figures,
        )
        report.finish_step_console(force_complete=True)

    if start_v <= 3.0 <= stop_v:
        report.section("STEP 3", total=4)

        _report_progress(report, prefix="STEP3 progress", current=1, total=4, extra="pool canonical PT folders", target="step")
        build_frozen_baseline_from_canonical_trajectories(
            work_dir=work_dir,
            train_folder=work_dir / BASELINE_TRAIN_REL,
            test_folder=work_dir / BASELINE_TEST_REL,
            dry=dry,
            report=report,
            preferred_label_attr=preferred_label_attr,
        )

        _report_progress(report, prefix="STEP3 progress", current=2, total=4, extra="validate frozen baseline", target="step")
        _validate_baseline_for_all_requested_horizons(
            work_dir=work_dir,
            base_task=base_task,
            horizons=horizons,
            run_all_horizons=run_all_horizons,
            report=report,
            tol=tol,
        )

        _report_progress(report, prefix="STEP3 progress", current=3, total=4, extra="baseline ready", target="step")
        _report_progress(report, prefix="STEP3 progress", current=4, total=4, extra="done", target="step")
        report.finish_step_console(force_complete=True)

    if start_v <= 4.0 <= stop_v:
        report.section("STEP 4", total=2)

        _report_progress(report, prefix="STEP4 progress", current=1, total=2, extra="preflight", target="step")
        _restore_live_test_from_baseline(work_dir=work_dir, dry=dry, report=report)

        report_label_stats(
            report,
            folder=work_dir / BASELINE_TRAIN_REL,
            name="Step4 BASELINE TRAIN",
            preferred_label_attr=preferred_label_attr,
            strict_preferred=True,
        )
        report_label_stats(
            report,
            folder=work_dir / BASELINE_TEST_REL,
            name="Step4 BASELINE TEST",
            preferred_label_attr=preferred_label_attr,
            strict_preferred=True,
        )

        if _task_requires_two_class_pooled_train(base_task):
            st_train = assert_two_class_folder(
                work_dir / BASELINE_TRAIN_REL,
                name="Step4 BASELINE TRAIN",
                require_balanced=False,
                balance_tolerance=tol,
                report=report,
                preferred_label_attr=preferred_label_attr,
                strict_preferred=True,
            )
        else:
            st_train = assert_nonempty_known_labels_folder(
                work_dir / BASELINE_TRAIN_REL,
                name="Step4 BASELINE TRAIN",
                report=report,
                preferred_label_attr=preferred_label_attr,
                strict_preferred=True,
            )
            warn_if_single_label(st_train, name="Step4 BASELINE TRAIN", report=report)
        assert_contiguous_days_per_sim(work_dir / BASELINE_TRAIN_REL, name="Step4 BASELINE TRAIN", report=report)

        _validate_baseline_for_all_requested_horizons(
            work_dir=work_dir,
            base_task=base_task,
            horizons=horizons,
            run_all_horizons=run_all_horizons,
            report=report,
            tol=tol,
        )

        assert_two_class_folder(
            work_dir / LIVE_TEST_REL,
            name="Step4 LIVE TEST",
            require_balanced=require_balanced_test,
            balance_tolerance=tol,
            report=report,
            preferred_label_attr=preferred_label_attr,
            strict_preferred=True,
        )
        assert_contiguous_days_per_sim(work_dir / LIVE_TEST_REL, name="Step4 LIVE TEST", report=report)

        _report_progress(report, prefix="STEP4 progress", current=2, total=2, extra="train/evaluate", target="step")
        run_training_step(
            step_tag="step4_baseline",
            data_folder=BASELINE_TRAIN_REL,
            work_dir=work_dir,
            archive_root=archive_root,
            py=py,
            dry=dry,
            report=report,
            horizons=horizons,
            run_all_horizons=run_all_horizons,
            T_list=T_list,
            run_all_T=run_all_T,
            archive_train_test_folders=archive_train_test_folders,
            ensure_test_folder_for_task=_make_baseline_test_checker(work_dir=work_dir, tol=tol, report=report),
            eval_test_folder=LIVE_TEST_REL,
            test_folder_for_archive=BASELINE_TEST_REL,
            no_train=no_train,
        )
        report.finish_step_console(force_complete=True)

    if start_v <= 5.0 <= stop_v:
        step5_folders = [
            ("no_edge_weights", "step5_no_edge_weights"),
            ("core_node_features_only", "step5_core_node_features_only"),
        ]
        step5_total = 2 + len(step5_folders)
        report.section("STEP 5", total=step5_total)

        _report_progress(report, prefix="STEP5 progress", current=1, total=step5_total, extra="restore baseline test", target="step")
        _restore_live_test_from_baseline(work_dir=work_dir, dry=dry, report=report)

        _report_progress(
            report,
            prefix="STEP5 progress",
            current=2,
            total=step5_total,
            extra="reuse ablations" if no_train else "build ablations",
            target="step",
        )
        if not no_train:
            if not dry:
                (work_dir / STEP5_ABLATION_REL / "no_edge_weights").mkdir(parents=True, exist_ok=True)
                (work_dir / STEP5_ABLATION_REL / "core_node_features_only").mkdir(parents=True, exist_ok=True)
            run_cmd([py, "ablate_edge_weights.py"], dry, cwd=work_dir, report=report, stream_output=False)
            run_cmd([py, "ablate_node_features.py"], dry, cwd=work_dir, report=report, stream_output=False)

        if (work_dir / STEP5_ABLATION_REL).exists():
            base_idx = 2
            for idx, (ab_folder, tag) in enumerate(step5_folders, start=1):
                _report_progress(
                    report,
                    prefix="STEP5 progress",
                    current=base_idx + idx,
                    total=step5_total,
                    extra=ab_folder,
                    target="step",
                )

                data_folder = STEP5_ABLATION_REL / ab_folder
                if not (work_dir / data_folder).exists():
                    continue

                report_label_stats(
                    report,
                    folder=work_dir / data_folder,
                    name=f"{tag} TRAIN",
                    preferred_label_attr=preferred_label_attr,
                    strict_preferred=True,
                )
                st = assert_nonempty_known_labels_folder(
                    work_dir / data_folder,
                    name=f"{tag} TRAIN",
                    report=report,
                    preferred_label_attr=preferred_label_attr,
                    strict_preferred=True,
                )
                warn_if_single_label(st, name=f"{tag} TRAIN", report=report)
                assert_contiguous_days_per_sim(work_dir / data_folder, name=f"{tag} TRAIN", report=report)

                assert_two_class_folder(
                    work_dir / LIVE_TEST_REL,
                    name=f"{tag} TEST",
                    require_balanced=require_balanced_test,
                    balance_tolerance=tol,
                    report=report,
                    preferred_label_attr=preferred_label_attr,
                    strict_preferred=True,
                )
                assert_contiguous_days_per_sim(work_dir / LIVE_TEST_REL, name=f"{tag} TEST", report=report)

                run_training_step(
                    step_tag=tag,
                    data_folder=data_folder,
                    work_dir=work_dir,
                    archive_root=archive_root,
                    py=py,
                    dry=dry,
                    report=report,
                    horizons=horizons,
                    run_all_horizons=run_all_horizons,
                    T_list=T_list,
                    run_all_T=run_all_T,
                    archive_train_test_folders=archive_train_test_folders,
                    ensure_test_folder_for_task=_make_baseline_test_checker(work_dir=work_dir, tol=tol, report=report),
                    test_folder_for_archive=BASELINE_TEST_REL,
                    no_train=no_train,
                )
        report.finish_step_console(force_complete=True)

    if start_v <= 6.1 <= stop_v:
        report.section("STEP 6.1", total=3)

        _report_progress(
            report,
            prefix="STEP6.1 progress",
            current=1,
            total=3,
            extra="reuse existing delay grid" if no_train else "generate + convert delay grid",
            target="step",
        )
        _restore_live_test_from_baseline(work_dir=work_dir, dry=dry, report=report)
        if not no_train:
            run_cmd([py, "generate_observation_delay_grid.py"], dry, cwd=work_dir, report=report, stream_output=False)
            run_cmd([py, "convert_collect_delay_grid.py"], dry, cwd=work_dir, report=report, stream_output=False)

        _report_progress(
            report,
            prefix="STEP6.1 progress",
            current=2,
            total=3,
            extra="skip dataset figures + purge" if no_train else "dataset figures + purge",
            target="step",
        )
        if not no_train:
            delay_dataset_roots = archive_dataset_graph_figures_before_purge(
                py=py,
                work_dir=work_dir,
                search_root=work_dir,
                archive_root=archive_root,
                stage_tag="step6.1_delay_datasets",
                dry=dry,
                report=report,
                cwd=work_dir,
                identity="Harry Triantafyllidis",
                enable_graph_folder_figures=enable_graph_folder_figures,
            )
            if keep_step_train_graphml and delay_dataset_roots:
                keep_graphml_roots(
                    src_roots=delay_dataset_roots,
                    dst_root=graphml_keep_root / "step6.1_delay_train_sets",
                    dry=dry,
                    report=report,
                    label="step6.1_delay_train_sets",
                )
            _purge_graphml_under(work_dir, dry=dry, report=report, label="step6.1_post_figures")

        _report_progress(report, prefix="STEP6.1 progress", current=3, total=3, extra="train delay conditions", target="step")
        if not dry:
            delay_root = _find_latest_dir(work_dir, _step_dataset_globs(base_task, "delay"))
            if delay_root is not None:
                report.kv("delay_root", delay_root)
                subdirs = sorted([d for d in delay_root.iterdir() if d.is_dir()])
                if not subdirs:
                    raise FileNotFoundError(f"No delay condition folders found under {delay_root.resolve()}")

                for d in subdirs:
                    tag = f"step6_delay_{_safe_tag(d.name)}"

                    report_label_stats(
                        report,
                        folder=d,
                        name=f"{tag} TRAIN",
                        preferred_label_attr=preferred_label_attr,
                        strict_preferred=True,
                    )
                    st = assert_nonempty_known_labels_folder(
                        d,
                        name=f"{tag} TRAIN",
                        report=report,
                        preferred_label_attr=preferred_label_attr,
                        strict_preferred=True,
                    )
                    warn_if_single_label(st, name=f"{tag} TRAIN", report=report)
                    assert_contiguous_days_per_sim(d, name=f"{tag} TRAIN", report=report)

                    assert_two_class_folder(
                        work_dir / LIVE_TEST_REL,
                        name=f"{tag} TEST",
                        require_balanced=require_balanced_test,
                        balance_tolerance=tol,
                        report=report,
                        preferred_label_attr=preferred_label_attr,
                        strict_preferred=True,
                    )
                    assert_contiguous_days_per_sim(work_dir / LIVE_TEST_REL, name=f"{tag} TEST", report=report)

                    run_training_step(
                        step_tag=tag,
                        data_folder=d.relative_to(work_dir),
                        work_dir=work_dir,
                        archive_root=archive_root,
                        py=py,
                        dry=dry,
                        report=report,
                        horizons=horizons,
                        run_all_horizons=run_all_horizons,
                        T_list=T_list,
                        run_all_T=run_all_T,
                        archive_train_test_folders=archive_train_test_folders,
                        ensure_test_folder_for_task=_make_baseline_test_checker(work_dir=work_dir, tol=tol, report=report),
                        test_folder_for_archive=BASELINE_TEST_REL,
                        no_train=no_train,
                    )
            elif no_train:
                archived_tags = _list_archived_step_tags(archive_root, "step6_delay_")
                if not archived_tags:
                    raise FileNotFoundError(
                        "Could not find Step 6.1 delay PT-flat folder, and no archived step6_delay_* folders were found."
                    )
                report.kv("delay_root", "<archive fallback>")
                for tag in archived_tags:
                    run_training_step(
                        step_tag=tag,
                        data_folder=Path("__archive_fallback__"),
                        work_dir=work_dir,
                        archive_root=archive_root,
                        py=py,
                        dry=dry,
                        report=report,
                        horizons=horizons,
                        run_all_horizons=run_all_horizons,
                        T_list=T_list,
                        run_all_T=run_all_T,
                        archive_train_test_folders=archive_train_test_folders,
                        ensure_test_folder_for_task=_make_baseline_test_checker(work_dir=work_dir, tol=tol, report=report),
                        test_folder_for_archive=BASELINE_TEST_REL,
                        no_train=True,
                    )
            else:
                raise FileNotFoundError("Could not find Step 6.1 delay PT-flat folder.")
        report.finish_step_console(force_complete=True)

    if start_v <= 6.2 <= stop_v:
        report.section("STEP 6.2", total=3)

        _report_progress(
            report,
            prefix="STEP6.2 progress",
            current=1,
            total=3,
            extra="reuse existing frequency grid" if no_train else "generate + convert frequency grid",
            target="step",
        )
        _restore_live_test_from_baseline(work_dir=work_dir, dry=dry, report=report)
        if not no_train:
            run_cmd([py, "generate_screen_freq_grid.py"], dry, cwd=work_dir, report=report, stream_output=False)
            run_cmd([py, "convert_collect_freq_grid.py"], dry, cwd=work_dir, report=report, stream_output=False)

        _report_progress(
            report,
            prefix="STEP6.2 progress",
            current=2,
            total=3,
            extra="skip dataset figures + purge" if no_train else "dataset figures + purge",
            target="step",
        )
        if not no_train:
            freq_dataset_roots = archive_dataset_graph_figures_before_purge(
                py=py,
                work_dir=work_dir,
                search_root=work_dir,
                archive_root=archive_root,
                stage_tag="step6.2_frequency_datasets",
                dry=dry,
                report=report,
                cwd=work_dir,
                identity="Harry Triantafyllidis",
                enable_graph_folder_figures=enable_graph_folder_figures,
            )
            if keep_step_train_graphml and freq_dataset_roots:
                keep_graphml_roots(
                    src_roots=freq_dataset_roots,
                    dst_root=graphml_keep_root / "step6.2_frequency_train_sets",
                    dry=dry,
                    report=report,
                    label="step6.2_frequency_train_sets",
                )
            _purge_graphml_under(work_dir, dry=dry, report=report, label="step6.2_post_figures")

        _report_progress(report, prefix="STEP6.2 progress", current=3, total=3, extra="train frequency conditions", target="step")
        if not dry:
            freq_root = _find_latest_dir(work_dir, _step_dataset_globs(base_task, "frequency"))
            if freq_root is not None:
                report.kv("freq_root", freq_root)
                subdirs = sorted([d for d in freq_root.iterdir() if d.is_dir()])
                if not subdirs:
                    raise FileNotFoundError(f"No frequency condition folders found under {freq_root.resolve()}")

                for d in subdirs:
                    tag = f"step6_freq_{_safe_tag(d.name)}"

                    report_label_stats(
                        report,
                        folder=d,
                        name=f"{tag} TRAIN",
                        preferred_label_attr=preferred_label_attr,
                        strict_preferred=True,
                    )
                    st = assert_nonempty_known_labels_folder(
                        d,
                        name=f"{tag} TRAIN",
                        report=report,
                        preferred_label_attr=preferred_label_attr,
                        strict_preferred=True,
                    )
                    warn_if_single_label(st, name=f"{tag} TRAIN", report=report)
                    assert_contiguous_days_per_sim(d, name=f"{tag} TRAIN", report=report)

                    assert_two_class_folder(
                        work_dir / LIVE_TEST_REL,
                        name=f"{tag} TEST",
                        require_balanced=require_balanced_test,
                        balance_tolerance=tol,
                        report=report,
                        preferred_label_attr=preferred_label_attr,
                        strict_preferred=True,
                    )
                    assert_contiguous_days_per_sim(work_dir / LIVE_TEST_REL, name=f"{tag} TEST", report=report)

                    run_training_step(
                        step_tag=tag,
                        data_folder=d.relative_to(work_dir),
                        work_dir=work_dir,
                        archive_root=archive_root,
                        py=py,
                        dry=dry,
                        report=report,
                        horizons=horizons,
                        run_all_horizons=run_all_horizons,
                        T_list=T_list,
                        run_all_T=run_all_T,
                        archive_train_test_folders=archive_train_test_folders,
                        ensure_test_folder_for_task=_make_baseline_test_checker(work_dir=work_dir, tol=tol, report=report),
                        test_folder_for_archive=BASELINE_TEST_REL,
                        no_train=no_train,
                    )
            elif no_train:
                archived_tags = _list_archived_step_tags(archive_root, "step6_freq_")
                if not archived_tags:
                    raise FileNotFoundError(
                        "Could not find Step 6.2 frequency PT-flat folder, and no archived step6_freq_* folders were found."
                    )
                report.kv("freq_root", "<archive fallback>")
                for tag in archived_tags:
                    run_training_step(
                        step_tag=tag,
                        data_folder=Path("__archive_fallback__"),
                        work_dir=work_dir,
                        archive_root=archive_root,
                        py=py,
                        dry=dry,
                        report=report,
                        horizons=horizons,
                        run_all_horizons=run_all_horizons,
                        T_list=T_list,
                        run_all_T=run_all_T,
                        archive_train_test_folders=archive_train_test_folders,
                        ensure_test_folder_for_task=_make_baseline_test_checker(work_dir=work_dir, tol=tol, report=report),
                        test_folder_for_archive=BASELINE_TEST_REL,
                        no_train=True,
                    )
            else:
                raise FileNotFoundError("Could not find Step 6.2 frequency PT-flat folder.")
        report.finish_step_console(force_complete=True)

    if start_v <= 7.0 <= stop_v:
        report.section("STEP 7", total=3)

        _report_progress(
            report,
            prefix="STEP7 progress",
            current=1,
            total=3,
            extra="reuse existing sweep" if no_train else "generate + convert sweep",
            target="step",
        )
        _restore_live_test_from_baseline(work_dir=work_dir, dry=dry, report=report)
        if not no_train:
            run_cmd([py, "generate_sweep_regime.py"], dry, cwd=work_dir, report=report, stream_output=False)
            run_cmd([py, "convert_collect_sweep.py"], dry, cwd=work_dir, report=report, stream_output=False)

        _report_progress(
            report,
            prefix="STEP7 progress",
            current=2,
            total=3,
            extra="skip dataset figures + purge" if no_train else "dataset figures + purge",
            target="step",
        )
        if not no_train:
            sweep_dataset_roots = archive_dataset_graph_figures_before_purge(
                py=py,
                work_dir=work_dir,
                search_root=work_dir,
                archive_root=archive_root,
                stage_tag="step7_sweep_datasets",
                dry=dry,
                report=report,
                cwd=work_dir,
                identity="Harry Triantafyllidis",
                enable_graph_folder_figures=enable_graph_folder_figures,
            )
            if keep_step_train_graphml and sweep_dataset_roots:
                keep_graphml_roots(
                    src_roots=sweep_dataset_roots,
                    dst_root=graphml_keep_root / "step7_sweep_train_set",
                    dry=dry,
                    report=report,
                    label="step7_sweep_train_set",
                )
            _purge_graphml_under(work_dir, dry=dry, report=report, label="step7_post_figures")

        _report_progress(report, prefix="STEP7 progress", current=3, total=3, extra="train sweep", target="step")
        if not dry:
            sweep_root = _find_latest_dir(work_dir, _step_dataset_globs(base_task, "sweep"))
            if sweep_root is not None:
                report.kv("sweep_root", sweep_root)

                report_label_stats(
                    report,
                    folder=sweep_root,
                    name="Step7 SWEEP TRAIN",
                    preferred_label_attr=preferred_label_attr,
                    strict_preferred=True,
                )
                st = assert_nonempty_known_labels_folder(
                    sweep_root,
                    name="Step7 SWEEP TRAIN",
                    report=report,
                    preferred_label_attr=preferred_label_attr,
                    strict_preferred=True,
                )
                warn_if_single_label(st, name="Step7 SWEEP TRAIN", report=report)
                assert_contiguous_days_per_sim(sweep_root, name="Step7 SWEEP TRAIN", report=report)

                assert_two_class_folder(
                    work_dir / LIVE_TEST_REL,
                    name="Step7 TEST",
                    require_balanced=require_balanced_test,
                    balance_tolerance=tol,
                    report=report,
                    preferred_label_attr=preferred_label_attr,
                    strict_preferred=True,
                )
                assert_contiguous_days_per_sim(work_dir / LIVE_TEST_REL, name="Step7 TEST", report=report)

                data_folder_for_step7 = sweep_root.relative_to(work_dir)
            elif no_train:
                report.kv("sweep_root", "<archive fallback>")
                data_folder_for_step7 = Path("__archive_fallback__")
            else:
                raise FileNotFoundError("Could not find Step 7 sweep PT-flat folder.")

            run_training_step(
                step_tag="step7_sweep",
                data_folder=data_folder_for_step7,
                work_dir=work_dir,
                archive_root=archive_root,
                py=py,
                dry=dry,
                report=report,
                horizons=horizons,
                run_all_horizons=run_all_horizons,
                T_list=T_list,
                run_all_T=run_all_T,
                archive_train_test_folders=archive_train_test_folders,
                ensure_test_folder_for_task=_make_baseline_test_checker(work_dir=work_dir, tol=tol, report=report),
                test_folder_for_archive=BASELINE_TEST_REL,
                no_train=no_train,
            )
        report.finish_step_console(force_complete=True)

# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="1")
    ap.add_argument("--stop", type=str, default="7")
    ap.add_argument("--dry_run", action="store_true")

    ap.add_argument(
        "--keep_graphml",
        action="store_true",
        help="If set, preserve GraphML files instead of purging them after conversion/collection.",
    )
    ap.add_argument(
        "--keep_step_train_graphml",
        action="store_true",
        help="If set, keep only the GraphML datasets needed to preserve each generated train set per step and one frozen test set per track.",
    )
    ap.add_argument("--run_both_state_modes", action="store_true")
    ap.add_argument("--archive_train_test_folders", action="store_true")
    ap.add_argument(
        "--run_graph_folder_figures",
        action="store_true",
        help="If set, run graph_folder_figures.py before GraphML purge. Default: disabled.",
    )

    ap.add_argument("--emit_latex", action="store_true")
    ap.add_argument(
        "--emit_latex_only",
        action="store_true",
        help="Skip the full pipeline and only rebuild the Overleaf package from an already completed results root.",
    )
    ap.add_argument(
        "--no_train",
        action="store_true",
        help="Do not retrain. Reuse existing trained_model.pt files to rerun evaluation, regenerate missing test plots/metrics in place, and rebuild LaTeX.",
    )
    ap.add_argument("--overleaf_dir", type=str, default=DEFAULT_OVERLEAF_DIRNAME)

    ap.add_argument(
        "--test_frac_per_class",
        type=float,
        default=float(CONFIG["TEST"].get("test_frac_per_class", 0.5)),
        help="Legacy argument retained for CLI compatibility; ignored by the explicit canonical train/test baseline build.",
    )

    ap.add_argument(
        "--run_all_horizons",
        action="store_true",
        help="If set, train/evaluate each training step across a horizon list.",
    )
    ap.add_argument(
        "--horizons",
        type=str,
        default="",
        help="Optional comma-separated horizons overriding CONFIG['CONVERT']['horizons'] when --run_all_horizons is set.",
    )

    ap.add_argument(
        "--run_all_T",
        action="store_true",
        help="If set, train/evaluate each training step across a list of window lengths T.",
    )
    ap.add_argument(
        "--T_list",
        type=str,
        default="",
        help="Optional comma-separated T values overriding CONFIG['MODEL']['T_list'].",
    )

    ap.add_argument("--results_parent", type=str, default=str(DEFAULT_RESULTS_PARENT))
    ts = ap.add_mutually_exclusive_group()
    ts.add_argument("--timestamped", action="store_true")
    ts.add_argument("--no_timestamp", action="store_true")
    args = ap.parse_args()
    run_started_at = time.time()

    start_v = step_key_to_float(args.start)
    stop_v = step_key_to_float(args.stop)
    dry = bool(args.dry_run)
    py = sys.executable
    timestamped = False

    test_frac = float(args.test_frac_per_class)
    if not (0.0 < test_frac <= 1.0):
        raise ValueError(f"--test_frac_per_class must be in (0,1], got {test_frac}")

    run_all_horizons = bool(args.run_all_horizons)
    run_all_T = bool(args.run_all_T)
    archive_train_test_folders = bool(args.archive_train_test_folders)
    no_train = bool(args.no_train)

    horizons: List[int] = []
    if run_all_horizons:
        if str(args.horizons).strip() != "":
            horizons = _parse_horizons_list(args.horizons)
        else:
            horizons = _parse_horizons_list(str(CONFIG.get("CONVERT", {}).get("horizons", "")).strip())
        if len(horizons) == 0:
            raise RuntimeError("--run_all_horizons is set but no horizons were provided/found.")
    else:
        task_h = _infer_horizon_from_task_name(str(CONFIG["MODEL"].get("task", "")).strip())
        if task_h is not None:
            horizons = [int(task_h)]
        else:
            horizons = [int(CONFIG["MODEL"].get("pred_horizon", 7))]

    t_list: List[int] = []
    if run_all_T:
        if str(args.T_list).strip() != "":
            t_list = _parse_T_list(args.T_list)
        else:
            cfg_t_list = str(CONFIG.get("MODEL", {}).get("T_list", "")).strip()
            if cfg_t_list != "":
                t_list = _parse_T_list(cfg_t_list)
            else:
                t_list = [int(CONFIG["MODEL"]["T"])]
        if len(t_list) == 0:
            raise RuntimeError("--run_all_T is set but no T values were provided/found.")
    else:
        t_list = [int(CONFIG["MODEL"]["T"])]

    project_root = Path(__file__).resolve().parent
    results_parent = Path(args.results_parent)

    dirs = _make_run_dirs(results_parent, timestamped=timestamped, dry=dry)
    run_root = dirs["run_root"]
    report_path = dirs["report_path"]

    report = Report(report_path, dry=dry)
    report.section("RUN METADATA")
    report.kv("utc_started", _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"))
    report.kv("python", sys.executable)
    report.kv("run_root", run_root)
    report.kv("start", args.start)
    report.kv("stop", args.stop)
    report.kv("test_frac_per_class", test_frac)
    report.kv("test_frac_per_class_note", "ignored by explicit canonical train/test baseline build")
    report.kv("run_all_horizons", int(run_all_horizons))
    report.kv("horizons", ",".join(str(h) for h in horizons))
    report.kv("run_all_T", int(run_all_T))
    report.kv("T_list", ",".join(str(t) for t in t_list))
    report.kv("archive_train_test_folders", int(archive_train_test_folders))
    report.kv("run_graph_folder_figures", int(bool(args.run_graph_folder_figures)))
    report.kv("emit_latex", int(bool(args.emit_latex)))
    report.kv("emit_latex_only", int(bool(args.emit_latex_only)))
    report.kv("no_train", int(bool(args.no_train)))
    report.kv("resume_root_mode", 1)
    report.kv("timestamped_flags_ignored", 1)
    report.kv("results_parent", str(results_parent))

    if bool(args.emit_latex_only):
        if not run_root.exists():
            raise FileNotFoundError(f"Results root does not exist: {run_root}")

        total_runtime_sec = time.time() - run_started_at
        total_runtime_hms = _format_duration_hms(total_runtime_sec)

        export_overleaf_package(
            run_root=run_root,
            archive_root=run_root,
            out_dir=Path(str(args.overleaf_dir)),
            dry=dry,
            report=report,
        )

        report.section("DONE")
        report.kv("utc_finished", _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"))
        report.kv("total_runtime_sec", round(total_runtime_sec, 3))
        report.kv("total_runtime_hms", total_runtime_hms)
        report.kv("report_txt", report_path)
        print(f"TOTAL RUN TIME: {total_runtime_hms}", flush=True)
        return 0

    required = [
        "generate_amr_data.py",
        "convert_to_pt.py",
        "train_amr_dygformer.py",
        "temporal_graph_dataset.py",
        "ablate_edge_weights.py",
        "ablate_node_features.py",
        "generate_sweep_regime.py",
        "convert_collect_sweep.py",
        "generate_observation_delay_grid.py",
        "convert_collect_delay_grid.py",
        "generate_screen_freq_grid.py",
        "convert_collect_freq_grid.py",
        "graph_folder_figures.py",
        "models_amr.py",
        "tasks.py",
    ]

    if dry:
        for f in required:
            require_file(project_root / f)
        report.write(f"Required scripts present in project_root: {len(required)}")
        report.write("DRY_RUN: code staging skipped")
    else:
        report.write("Code will be staged into persistent per-track work_dir locations")

    os.environ["DT_SIM_EXTRA_ARGS"] = _build_sim_extra_args_global_only(CONFIG["SIM"])
    effective_convert_cfg = dict(CONFIG["CONVERT"])
    effective_convert_cfg["horizons"] = ",".join(str(h) for h in horizons)

    os.environ["DT_CONVERT_EXTRA_ARGS"] = _build_convert_extra_args(effective_convert_cfg)
    #os.environ["DT_KEEP_GRAPHML"] = "0"
    os.environ["DT_KEEP_GRAPHML"] = "1" if bool(args.keep_graphml) else "0"

    report.section("ENVIRONMENT")
    report.kv("DT_SIM_EXTRA_ARGS", os.environ["DT_SIM_EXTRA_ARGS"])
    report.kv("DT_CONVERT_EXTRA_ARGS", os.environ["DT_CONVERT_EXTRA_ARGS"])
    report.kv("DT_KEEP_GRAPHML", os.environ["DT_KEEP_GRAPHML"])
    report.kv("keep_graphml_note", "GraphML purge is skipped when --keep_graphml is set.")
    report.kv("keep_step_train_graphml", int(bool(args.keep_step_train_graphml)))
    report.kv("keep_step_train_graphml_note", "When set, step-specific train GraphML and one frozen test GraphML copy are preserved under each track's kept_graphml folder before purge.")
    
    def _run_one_track(track_name: str, state_mode: str, pt_out_dir: str) -> None:
        track_dirs = _make_track_dirs(run_root, track_name, dry=dry)
        work_dir = track_dirs["work_dir"]
        archive_root = track_dirs["archive_root"]

        report.section(f"TRACK: {track_name}")
        report.kv("track_root", track_dirs["track_root"])
        report.kv("work_dir", work_dir)
        report.kv("archive_root", archive_root)
        report.kv("graphml_keep_root", track_dirs["graphml_keep_root"])

        if dry:
            for f in required:
                require_file(project_root / f)
        else:
            stage_code_into_workdir(project_root, work_dir, dry=dry)
            for f in required:
                require_file(work_dir / f)

        os.environ["DT_STATE_MODE"] = state_mode
        os.environ["DT_PT_OUT_DIR"] = pt_out_dir
        os.environ["PYTHONPATH"] = str(work_dir.resolve())

        report.kv("DT_STATE_MODE", os.environ["DT_STATE_MODE"])
        report.kv("DT_PT_OUT_DIR", os.environ["DT_PT_OUT_DIR"])
        report.kv("PYTHONPATH", os.environ["PYTHONPATH"])

        run_pipeline_once(
            start_v=start_v,
            stop_v=stop_v,
            dry=dry,
            py=py,
            archive_train_test_folders=archive_train_test_folders,
            archive_root=archive_root,
            graphml_keep_root=track_dirs["graphml_keep_root"],
            work_dir=work_dir,
            test_frac_per_class=test_frac,
            report=report,
            horizons=horizons,
            run_all_horizons=run_all_horizons,
            T_list=t_list,
            run_all_T=run_all_T,
            enable_graph_folder_figures=bool(args.run_graph_folder_figures),
            keep_step_train_graphml=bool(args.keep_step_train_graphml),
            no_train=no_train,
        )

        # Optional cleanup: delete only transient .pt files for this track after it fully finishes.
        # Resume-critical assets are preserved: canonical *_pt_flat folders and baseline train/test folders.
        if no_train:
            cleaned_n = 0
            report.write(
                f"PT_CLEANUP skipped for track={track_name} under {work_dir} because --no_train was set"
            )
        else:
            cleaned_n = _cleanup_transient_pt_files(
                work_dir=work_dir,
                pt_out_dir=Path(pt_out_dir),
                dry=dry,
                report=report,
            )

            report.write(
                f"PT_CLEANUP completed for track={track_name} under {work_dir}; transient_pt_deleted={cleaned_n}"
            )

    try:
        if args.run_both_state_modes:
            _run_one_track(
                track_name="TRACK_ground_truth",
                state_mode="ground_truth",
                pt_out_dir=str(_default_pt_out_dir_for_track(run_root, "TRACK_ground_truth")),
            )
            _run_one_track(
                track_name="TRACK_partial_observation",
                state_mode="partial_observation",
                pt_out_dir=str(_default_pt_out_dir_for_track(run_root, "TRACK_partial_observation")),
            )
        else:
            state_mode_single = os.environ.get("DT_STATE_MODE", "ground_truth").strip() or "ground_truth"
            track_name_single = f"TRACK_{state_mode_single}"
            pt_out_dir_single = os.environ.get("DT_PT_OUT_DIR", "").strip()
            if pt_out_dir_single == "":
                pt_out_dir_single = str(_default_pt_out_dir_for_track(run_root, track_name_single))

            _run_one_track(
                track_name=track_name_single,
                state_mode=state_mode_single,
                pt_out_dir=pt_out_dir_single,
            )

        total_runtime_sec = time.time() - run_started_at
        total_runtime_hms = _format_duration_hms(total_runtime_sec)

        report.section("DONE")
        report.kv("utc_finished", _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"))
        report.kv("total_runtime_sec", round(total_runtime_sec, 3))
        report.kv("total_runtime_hms", total_runtime_hms)
        report.kv("report_txt", report_path)

        if bool(args.emit_latex) and not dry:
            export_overleaf_package(
                run_root=run_root,
                archive_root=run_root,
                out_dir=Path(str(args.overleaf_dir)),
                dry=dry,
                report=report,
            )

        print(f"TOTAL RUN TIME: {total_runtime_hms}", flush=True)

        return 0
    finally:
        report.finish_step_console(force_complete=True)


if __name__ == "__main__":
    raise SystemExit(main())
