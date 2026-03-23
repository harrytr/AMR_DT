# AMR Digital Twin: Hospital Simulation and Temporal Graph Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![PyTorch%20Geometric](https://img.shields.io/badge/PyTorch%20Geometric-enabled-orange)
![R](https://img.shields.io/badge/R-Shiny-276DC3)
![Workflow](https://img.shields.io/badge/Workflow-Reproducible-success)

A reproducible research platform for simulating hospital antimicrobial resistance (AMR) dynamics as daily directed contact graphs, transforming them into temporal graph-learning datasets, and training temporal graph neural networks for mechanism-aware forecasting tasks.

This repository integrates three layers in a single workflow:

1. **Mechanistic hospital simulation** of transmission, importation, selection, screening, and isolation.
2. **Graph conversion and target construction** from daily GraphML trajectories to PyTorch Geometric datasets.
3. **End-to-end experimental orchestration** for baselines, ablations, robustness analyses, and paper-ready artifact generation.

The main reproducibility entrypoint is:

```bash
python experiments_pb.py --run_both_state_modes --emit_latex --run_all_T --archive_train_test_folders
```

---

## Contents

- [Overview](#overview)
- [Core capabilities](#core-capabilities)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Full reproducibility pipeline](#full-reproducibility-pipeline)
- [Resuming and partial reruns](#resuming-and-partial-reruns)
- [Important command-line options](#important-command-line-options)
- [Outputs](#outputs)
- [Overleaf export](#overleaf-export)
- [Reproducibility and design notes](#reproducibility-and-design-notes)
- [Citation](#citation)
- [License](#license)

---

## Overview

The platform is designed for controlled, mechanism-grounded AMR experimentation in hospital environments. It represents each hospital day as a directed contact graph whose nodes correspond to patients and staff, and whose edges encode contact opportunities and transmission-relevant interactions. These daily graphs are then assembled into temporal sequences for downstream graph learning.

The repository currently supports two experimental state modes:

- **`ground_truth`**
- **`partial_observation`**

The active reproducibility driver in this repository is **`experiments_pb.py`**.

---

## Core capabilities

### 1) Mechanistic AMR hospital simulator

`generate_amr_data.py` simulates ward-structured hospital AMR dynamics with support for:

- patients and staff as distinct node types
- directed, weighted contact graphs
- ward assignment, turnover, and multi-ward staffing structure
- admission-based importation pressure
- within-hospital transmission
- antibiotic exposure and within-host selection effects
- screening and isolation interventions
- optional partial-observation regimes
- configurable outbreak and surveillance settings

The simulator writes daily GraphML snapshots together with trajectory-level metadata and label-relevant summaries used downstream for audit and task construction.

### 2) GraphML to PyTorch Geometric conversion

`convert_to_pt.py` converts GraphML trajectories into `.pt` graph objects for temporal learning.

Converted samples store, depending on task and configuration:

- node features
- edge features
- graph-level targets
- stable node identifiers for traceability
- horizon-specific labels for forecasting tasks

Auxiliary CSV outputs are also generated for auditing and downstream checks.

### 3) Temporal graph learning

`train_amr_dygformer.py` trains the temporal learning model used throughout the experiments. The model stack combines:

- a daily graph encoder built around **GraphSAGE-style message passing**
- a **Transformer-based temporal encoder** over windows of length `T`
- task-specific target logic from `tasks.py`

The repository includes both regression-style and classification-style AMR forecasting tasks.

At the time of inspection, the default configuration in `experiments_pb.py` is:

```text
task = early_outbreak_warning_h14
T = 7
horizon = 14
```

Available task definitions can be listed via:

```bash
python list_tasks.py
```

### 4) End-to-end reproducibility workflow

`experiments_pb.py` orchestrates the full paper-style workflow, including:

- canonical data generation
- conversion and flattening of PT datasets
- baseline train/test construction
- frozen-test evaluation
- ablation studies
- observation-delay robustness experiments
- screening-frequency robustness experiments
- sweep-regime benchmarking
- figure generation and archive creation
- Overleaf-ready export

---

## Repository layout

### Core pipeline

- `experiments_pb.py` — main reproducibility driver for Steps 1–7
- `generate_amr_data.py` — mechanistic AMR simulator
- `convert_to_pt.py` — GraphML to PyTorch Geometric conversion
- `train_amr_dygformer.py` — temporal graph learning trainer
- `temporal_graph_dataset.py` — temporal dataset construction
- `models_amr.py` — model definitions
- `tasks.py` — forecasting tasks and target extraction
- `list_tasks.py` — utility to inspect registered tasks

### Dataset generation and transformation

- `run_turnover_cohorts.py`
- `prepare_pt_flat_from_turnover.py`
- `make_combined_pt_folder.py`
- `generate_observation_delay_grid.py`
- `convert_collect_delay_grid.py`
- `generate_screen_freq_grid.py`
- `convert_collect_freq_grid.py`
- `generate_sweep_regime.py`
- `convert_collect_sweep.py`

### Evaluation, diagnostics, and figures

- `ablate_edge_weights.py`
- `ablate_node_features.py`
- `graph_folder_figures.py`
- `mechanism_separation_from_sims.py`
- `summarise_mechanism_components.py`
- `audit_endog_import_labels.py`
- `audit_pt_endog_import_h7.py`
- `check_folder_label_balance.py`
- `check_test_label_balance.py`
- `prune_overleaf_package.py`

### Dataset-folder builders

- `build_balanced_test_folder.py`
- `build_contiguous_test_folder.py`
- `build_delay_test_folder.py`
- `build_sweep_test_folder.py`

### R / Shiny interface

- `run.R`
- `simulator.R`
- `renv.lock`

---

## Installation

### Python

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate gnn_env
```

Verify key Python dependencies:

```bash
python -c "import torch, networkx, numpy, pandas; print('Python environment OK')"
```

If `torch_geometric` is not available after environment creation on your platform, install the matching build for your local PyTorch/CUDA configuration before training.

### R

Restore the R environment with `renv`:

```bash
Rscript -e "if(!requireNamespace('renv', quietly=TRUE)) install.packages('renv', repos='https://cloud.r-project.org'); renv::restore()"
```

---

## Quick start

### 1) Generate a single simulation

```bash
python generate_amr_data.py --output_dir demo_sim --seed 123 --num_days 30 --export_yaml
```

### 2) Convert GraphML snapshots to `.pt`

```bash
python convert_to_pt.py --graphml_dir demo_sim
```

### 3) Train a temporal model manually

```bash
python train_amr_dygformer.py --data_folder demo_sim --task early_outbreak_warning_h14 --T 7 --epochs 10 --batch_size 16
```

### 4) Inspect registered tasks

```bash
python list_tasks.py
```

---

## Full reproducibility pipeline

The recommended full experiment run is:

```bash
python experiments_pb.py --run_both_state_modes --emit_latex --run_all_T --archive_train_test_folders
```

This pipeline is designed to be:

- deterministic
- resumable
- strict about train/test dataset integrity
- track-aware across `ground_truth` and `partial_observation`
- suitable for paper-grade figure and archive generation

### Canonical design

The pipeline is organized around four explicit canonical Step 1 trajectory families:

- `endog_high_train`
- `import_high_train`
- `endog_high_test`
- `import_high_test`

These are used to construct a frozen baseline split:

- `synthetic_amr_graphs_train`
- `synthetic_amr_graphs_test_frozen`

### Step structure

- **Step 1**: generate the four canonical trajectory families
- **Step 2**: convert each trajectory family to `.pt` and flatten family-specific PT folders
- **Step 3**: build the baseline train/test pair from the canonical families
- **Step 4**: train and evaluate the baseline against the frozen test set
- **Step 5**: run ablations
- **Step 6.1**: run observation-delay robustness experiments
- **Step 6.2**: run screening-frequency robustness experiments
- **Step 7**: run sweep-regime experiments

A central design safeguard is that later steps may alter the **training side** while preserving evaluation semantics through the **frozen baseline test set**.

Before each training call, the frozen baseline test dataset is restored to the live trainer path:

```text
synthetic_amr_graphs_test
```

This prevents accidental evaluation drift across experimental stages.

### Single-track runs

Ground-truth mode only:

```bash
DT_STATE_MODE=ground_truth python experiments_pb.py --emit_latex
```

Partial-observation mode only:

```bash
DT_STATE_MODE=partial_observation python experiments_pb.py --emit_latex
```

### Multi-horizon and multi-window runs

```bash
python experiments_pb.py --run_both_state_modes --run_all_horizons --run_all_T --emit_latex
```

---

## Resuming and partial reruns

The main pipeline is resumable by step.

### Example: rerun Steps 6.2 to 7 for both tracks

```bash
python experiments_pb.py --run_both_state_modes --start 6.2 --stop 7 --emit_latex --run_all_T --archive_train_test_folders
```

### Example: rerun Steps 6.2 to 7 for ground-truth mode only

```bash
DT_STATE_MODE=ground_truth python experiments_pb.py --start 6.2 --stop 7 --emit_latex --run_all_T --archive_train_test_folders
```

### Supported step keys

- `1`
- `2`
- `3`
- `4`
- `5`
- `6.1`
- `6.2`
- `7`

---

## Important command-line options

Commonly used options in `experiments_pb.py` include:

```bash
--start
--stop
--dry_run
--run_both_state_modes
--archive_train_test_folders
--run_graph_folder_figures
--emit_latex
--emit_latex_only
--overleaf_dir
--results_parent
--run_all_horizons
--horizons 7,14
--run_all_T
--T_list 7,14
```

Retained for compatibility but effectively ignored by the pipeline:

```bash
--timestamped
--no_timestamp
--keep_graphml
--test_frac_per_class
```

Notes:

- `--keep_graphml` is deprecated and ignored in this workflow.
- `--test_frac_per_class` is retained for CLI compatibility but not used in the explicit canonical baseline build.

---

## Outputs

A typical full run writes to a deterministic results root such as:

```text
experiments_results/
├── run_report.txt
├── TRACK_ground_truth/
│   └── work/
│       ├── synthetic_amr_graphs_train/
│       ├── synthetic_amr_graphs_test/
│       ├── synthetic_amr_graphs_test_frozen/
│       ├── training_outputs*/
│       └── repro_artifacts_steps_1_7/
└── TRACK_partial_observation/
    └── work/
        ├── synthetic_amr_graphs_train/
        ├── synthetic_amr_graphs_test/
        ├── synthetic_amr_graphs_test_frozen/
        ├── training_outputs*/
        └── repro_artifacts_steps_1_7/
```

Archived outputs may include:

- training curves
- confusion matrices
- ROC figures
- saved model checkpoints
- copied train/test datasets when archiving is enabled
- dataset graph-figure summaries
- LaTeX-ready figure bundles for manuscript integration

---

## Overleaf export

When `--emit_latex` is enabled, the pipeline assembles an Overleaf-ready package by collecting archived figures and generating a `latex.txt` file with reusable manuscript figure blocks.

Example:

```bash
python experiments_pb.py --run_both_state_modes --emit_latex
```

Default export location:

```text
experiments_results/overleaf_package/
```

Typical contents:

- `figures/`
- `latex.txt`

For rebuilding only the manuscript figure package from an already completed results root, use:

```bash
python experiments_pb.py --emit_latex_only
```

---

## Reproducibility and design notes

This repository is structured to support rigorous, repeatable experimentation.

Key implementation safeguards include:

- deterministic per-track working directories
- validation of required scripts before execution
- strict folder and label integrity checks
- horizon-aware validation of training labels
- contiguity checks across simulations and days
- frozen-test restoration before each training run
- optional graph-folder figure generation before cleanup
- automatic GraphML cleanup after conversion to control storage growth
- archiving of step-specific training outputs
- optional archiving of the exact train/test folders used in each experiment

Two practical points are worth noting:

1. The pipeline writes into a deterministic results layout rather than timestamped run folders.
2. The experimental workflow includes cleanup logic for transient artifacts to keep large runs manageable in disk usage.

---

## Citation

If you use this repository, please cite the associated manuscript.
