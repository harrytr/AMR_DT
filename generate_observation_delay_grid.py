#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


DELAY_VALUES = [0, 2, 5]


def _extra_args_from_env(env_key: str) -> List[str]:
    s = str(os.environ.get(env_key, "")).strip()
    return shlex.split(s) if s else []


def run_one(out_dir: Path, seed: int, delay: int) -> int:
    cmd: List[str] = [
        sys.executable,
        "generate_amr_data.py",
        "--output_dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--num_days",
        "30",
        "--daily_discharge_frac",
        "0.07",
        "--daily_discharge_min_per_ward",
        "1",
        "--p_admit_import_cs",
        "0.08",
        "--p_admit_import_cr",
        "0.08",
        "--screen_every_k_days",
        "7",
        "--screen_on_admission",
        "1",
        "--screen_result_delay_days",
        str(delay),
        "--persist_observations",
        "1",
        "--export_yaml",
    ]

    # NEW
    cmd += _extra_args_from_env("DT_SIM_EXTRA_ARGS")

    print("RUN:", " ".join(cmd), flush=True)
    return int(subprocess.run(cmd, text=True).returncode)


def main() -> int:
    base = Path("synthetic_endog_import_step6_delay_v2")
    base.mkdir(parents=True, exist_ok=True)

    for delay in DELAY_VALUES:
        ddir = base / f"delay_{delay}"
        ddir.mkdir(parents=True, exist_ok=True)
        for r in range(10):
            out = ddir / f"sim_{r:03d}"
            out.mkdir(parents=True, exist_ok=True)
            rc = run_one(out, seed=9000 + delay * 100 + r, delay=delay)
            if rc != 0:
                print(f"FAILED delay={delay} sim={r} rc={rc}", flush=True)
                return rc

    print("STEP6A_DONE: generated delay grid", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
