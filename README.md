# AMR Digital Twin: Hospital Simulation and Temporal Graph Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![PyTorch%20Geometric](https://img.shields.io/badge/PyTorch%20Geometric-enabled-orange)
![R](https://img.shields.io/badge/R-Shiny-276DC3)
![Workflow](https://img.shields.io/badge/Workflow-Reproducible-success)

A reproducible research platform for simulating hospital antimicrobial resistance (AMR) dynamics as daily directed contact graphs, converting them into temporal graph-learning datasets, and training temporal graph neural networks for mechanism-aware forecasting tasks.

This repository integrates three connected layers:

1. **Mechanistic hospital simulation** of transmission, importation, selection, screening, isolation, seasonality, and regime shifts.
2. **Graph conversion and target construction** from daily GraphML trajectories to PyTorch Geometric datasets.
3. **End-to-end experimental orchestration** for baselines, ablations, robustness analyses, translational attribution figures, and Overleaf-ready manuscript export.

The current recommended reproducibility entrypoint in this repository is:

```bash
python experiments_pbc7.py --run_both_state_modes --emit_latex --run_all_T --archive_train_test_folders --keep_step_train_graphml
```

---

## Contents

- [Overview](#overview)
- [Core capabilities](#core-capabilities)
- [Repository layout](#repository-layout)
- [Environment reproduction](#environment-reproduction)
- [Quick start](#quick-start)
- [Full reproducibility pipeline](#full-reproducibility-pipeline)
- [Resuming and partial reruns](#resuming-and-partial-reruns)
- [Important command-line options](#important-command-line-options)
- [Current default experimental configuration](#current-default-experimental-configuration)
- [Outputs](#outputs)
- [GraphML retention and statistics workflow](#graphml-retention-and-statistics-workflow)
- [Overleaf export](#overleaf-export)
- [Reproducibility and design notes](#reproducibility-and-design-notes)
- [Citation](#citation)

---

## Overview

The platform is designed for controlled, mechanism-grounded AMR experimentation in hospital environments. Each hospital day is represented as a directed contact graph whose nodes correspond to patients and staff, and whose edges encode contact opportunities and transmission-relevant interactions. These daily graphs are then assembled into temporal sequences for downstream graph learning.

The repository supports two state-observation tracks:

- **`ground_truth`**
- **`partial_observation`**

The repository’s reproducibility workflow and CLI are defined by **`experiments_pbc7.py`**.

---

## Core capabilities

### 1) Mechanistic AMR hospital simulator

`generate_amr_data.py` simulates ward-structured hospital AMR dynamics with support for:

- patients and staff as distinct node types
- directed, weighted contact graphs
- ward assignment, home-ward structure, and multi-ward staffing
- admission-based importation pressure
- within-hospital transmission
- antibiotic exposure and within-host selection effects
- screening and isolation interventions
- optional partial-observation regimes
- configurable outbreak, surveillance, seasonality, and superspreader settings

The simulator writes daily GraphML snapshots together with trajectory-level metadata and label-relevant summaries used downstream for audit and task construction.

### 2) GraphML to PyTorch Geometric conversion

`convert_to_pt.py` converts GraphML trajectories into `.pt` graph objects for temporal learning.

Converted samples can store, depending on task and configuration:

- node features
- edge features
- graph-level targets
- stable node identifiers for traceability
- horizon-specific labels for forecasting tasks
- ward-aware metadata used for downstream translational attribution

Auxiliary CSV outputs are also generated for auditing and downstream checks.

### 3) Temporal graph learning

`train_amr_dygformer.py` trains the temporal learning model used throughout the experiments. The current stack combines:

- a daily graph encoder built around **GraphSAGE-style message passing**
- a **Transformer-based temporal encoder** over windows of length `T`
- task-specific target logic from `tasks.py`
- optional full-graph attribution passes for translational figure export

The default paper-style configuration in the current driver is:

```text
task = endogenous_importation_majority_h7
T = 7
horizon = 7
```

Available task definitions can be listed via:

```bash
python list_tasks.py
```

### 4) Translational attribution outputs

The training pipeline can export post hoc translational figures derived from learned node-attention summaries and preserved graph metadata. These are interpretability summaries rather than causal claims.

The current training driver exposes:

- `fullgraph_attribution_pass`
- `emit_translational_figures`
- `translational_top_k`

Translational figures are exported under `translational_figures/` inside training outputs and can be collected into the Overleaf package.

### 5) End-to-end reproducibility workflow

`experiments_pbc7.py` orchestrates the full workflow, including:

- canonical data generation
- conversion and flattening of PT datasets
- baseline train/test construction
- frozen-test evaluation
- ablation studies
- observation-delay robustness experiments
- screening-frequency robustness experiments
- sweep-regime benchmarking
- complete distribution-shift benchmarking
- baseline-to-shift transfer evaluation
- optional lightweight tuning for Step 4 and Step 8
- translational figure export
- figure generation and archive creation
- Overleaf-ready export

---

## Repository layout

### Core pipeline

- `experiments_pbc7.py` — reproducibility driver for Steps 1–9
- `generate_amr_data.py` — mechanistic AMR simulator
- `convert_to_pt.py` — GraphML to PyTorch Geometric conversion
- `train_amr_dygformer.py` — temporal graph learning trainer
- `temporal_graph_dataset.py` — temporal dataset construction
- `models_amr.py` — model definitions
- `tasks.py` — forecasting tasks and target extraction
- `list_tasks.py` — utility to inspect registered tasks
- `tune_hparams.py` — lightweight tuning utility used by the latest driver

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
- `run_graph_folder_figures_batch.py`
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

## Environment reproduction

The exact working conda environment used for these experiments may vary by machine, but the codebase expects a Python environment with PyTorch and PyTorch Geometric support.

At minimum, confirm the availability of:

```bash
python -c "import torch, torch_geometric; print('torch', torch.__version__); print('pyg', torch_geometric.__version__)"
```

If your environment also uses sparse PyG extensions, verify:

```bash
python -c "import torch, torch_geometric, torch_sparse, torch_scatter; print('torch', torch.__version__); print('pyg', torch_geometric.__version__); print('torch_sparse', torch_sparse.__version__); print('torch_scatter', torch_scatter.__version__)"
```

The repository currently includes `renv.lock` for the R-side interface. If you maintain a conda export separately, store it alongside the repository for exact environment reconstruction.

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
python train_amr_dygformer.py --data_folder demo_sim --task endogenous_importation_majority_h7 --T 7 --epochs 10 --batch_size 16
```

### 4) Inspect registered tasks

```bash
python list_tasks.py
```

---

## Full reproducibility pipeline

The recommended full experiment run is:

```bash
python experiments_pbc7.py --run_both_state_modes --emit_latex --run_all_T --archive_train_test_folders --keep_step_train_graphml
```

If you also want dataset graph summaries generated during the pipeline, add:

```bash
--run_graph_folder_figures
```

Example:

```bash
python experiments_pbc7.py --run_both_state_modes --emit_latex --run_all_T --archive_train_test_folders --keep_step_train_graphml --run_graph_folder_figures
```

This pipeline is designed to be:

- deterministic in layout
- resumable by step
- strict about train/test dataset integrity
- track-aware across `ground_truth` and `partial_observation`
- suitable for paper-grade figure and archive generation

### Parallel dual-track execution

When `--run_both_state_modes` is used, the latest driver runs the two tracks in parallel as isolated subprocesses and performs the combined Overleaf export only after both tracks finish. The parent run keeps `run_report.txt`, and track-specific child reports are also written.

### Canonical design

For the default task family, the pipeline is organized around four canonical Step 1 trajectory families:

- `endog_high_train`
- `import_high_train`
- `endog_high_test`
- `import_high_test`

These are pooled into the baseline split:

- `synthetic_amr_graphs_train`
- `synthetic_amr_graphs_test_frozen`

### Step structure

- **Step 1**: generate the canonical trajectory families
- **Step 2**: convert each trajectory family to `.pt` and flatten family-specific PT folders
- **Step 3**: build the baseline train/test pair from the canonical families
- **Step 4**: train and evaluate the baseline against the frozen test set
- **Step 5**: run ablations
- **Step 6.1**: run observation-delay robustness experiments
- **Step 6.2**: run screening-frequency robustness experiments
- **Step 7**: run sweep-regime experiments
- **Step 8**: run a complete distribution-shift benchmark with shifted train and shifted test cohorts
- **Step 9**: evaluate the Step 4 baseline model directly on the Step 8 shifted test set without retraining

Before each canonical training call, the frozen baseline test dataset is restored to the live trainer path:

```text
synthetic_amr_graphs_test
```

This prevents accidental evaluation drift across stages.

### Single-track runs

Ground-truth mode only:

```bash
DT_STATE_MODE=ground_truth python experiments_pbc7.py --emit_latex
```

Partial-observation mode only:

```bash
DT_STATE_MODE=partial_observation python experiments_pbc7.py --emit_latex
```

### Multi-horizon and multi-window runs

```bash
python experiments_pbc7.py --run_both_state_modes --run_all_horizons --run_all_T --emit_latex
```

---

## Resuming and partial reruns

The pipeline is resumable by step.

### Example: rerun Steps 4 to 9 for both tracks

```bash
python experiments_pbc7.py --run_both_state_modes --start 4 --stop 9 --emit_latex --run_all_T --archive_train_test_folders --keep_step_train_graphml
```

### Example: rerun Steps 6.2 to 9 for both tracks

```bash
python experiments_pbc7.py --run_both_state_modes --start 6.2 --stop 9 --emit_latex --run_all_T --archive_train_test_folders --keep_step_train_graphml
```

### Example: reuse all existing datasets and retrain only models

```bash
python experiments_pbc7.py --run_both_state_modes --start 4 --stop 9 --no_simulation --emit_latex --run_all_T
```

### Example: evaluation-only repair mode

```bash
python experiments_pbc7.py --run_both_state_modes --start 4 --stop 9 --no_train --emit_latex --run_all_T
```

### Example: Step 4 tuning only

```bash
python experiments_pbc7.py --run_both_state_modes --start 4 --stop 4 --tune_step4
```

### Example: Step 8 tuning only

```bash
python experiments_pbc7.py --run_both_state_modes --start 8 --stop 8 --tune_step8
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
- `8`
- `9`

Step 9 is evaluation-only by design and therefore depends on archived Step 4 and Step 8 artifacts already being present for that track.

---

## Important command-line options

Commonly used options in `experiments_pbc7.py` include:

```bash
--start
--stop
--dry_run
--keep_graphml
--keep_step_train_graphml
--run_both_state_modes
--archive_train_test_folders
--run_graph_folder_figures
--emit_latex
--emit_latex_only
--no_train
--no_simulation
--overleaf_dir
--tune_step4
--tune_step8
--tune_trials_quick
--tune_finalists
--tune_quick_epochs
--tune_full_epochs
--tune_split_seed
--run_all_horizons
--horizons 7,14
--run_all_T
--T_list 7,14
--results_parent
```

Retained mainly for compatibility:

```bash
--timestamped
--no_timestamp
--test_frac_per_class
```

Notes:

- `--keep_step_train_graphml` keeps selected GraphML train/test folders under `kept_graphml/` for later statistics and figure workflows.
- `--keep_graphml` preserves GraphML globally by skipping purge.
- `--test_frac_per_class` is retained for CLI compatibility but ignored by the explicit canonical baseline build in the current driver.

---

## Current default experimental configuration

The current checked-in defaults in `experiments_pbc7.py` are:

### Simulator defaults

```text
num_regions = 1
num_wards = 10
num_patients = 200
num_staff = 300
staff_wards_per_staff = 2
```

### Baseline training defaults

```text
task = endogenous_importation_majority_h7
pred_horizon = 7
T = 7
T_list = 7
hidden = 32
heads = 2
dropout = 0.2
transformer_layers = 2
sage_layers = 2
batch_size = 16
epochs = 50
lr = 1e-5
neighbor_sampling = true
num_neighbors = 15,10
seed_count = 256
seed_batch_size = 64
max_sub_batches = 4
attn_top_k = 10
attn_rank_by = abs_diff
fullgraph_attribution_pass = true
emit_translational_figures = true
translational_top_k = 20
```

### Step 1 defaults

```text
n_sims_per_trajectory = 10
num_days = 60
```

### Step 8 defaults

```text
n_sims_per_trajectory = 10
num_days = 360
```

Step 8 uses explicit shifted train/test trajectory families defined directly inside `CONFIG["STEP8"]`, including superspreader and seasonal-importation settings.

---

## Outputs

A typical full run writes to a deterministic results root such as:

```text
experiments_results/
├── run_report.txt
├── run_report_TRACK_ground_truth.txt
├── run_report_TRACK_partial_observation.txt
├── TRACK_ground_truth/
│   ├── kept_graphml/
│   └── work/
│       ├── synthetic_amr_graphs_train/
│       ├── synthetic_amr_graphs_test/
│       ├── synthetic_amr_graphs_test_frozen/
│       ├── training_outputs*/
│       └── repro_artifacts_steps_1_9/
└── TRACK_partial_observation/
    ├── kept_graphml/
    └── work/
        ├── synthetic_amr_graphs_train/
        ├── synthetic_amr_graphs_test/
        ├── synthetic_amr_graphs_test_frozen/
        ├── training_outputs*/
        └── repro_artifacts_steps_1_9/
```

Archived outputs may include:

- training curves
- confusion matrices
- ROC figures
- saved model checkpoints
- translational figures
- copied train/test datasets when archiving is enabled
- dataset graph-figure summaries
- LaTeX-ready figure bundles for manuscript integration
- tuning outputs for Step 4 or Step 8 when enabled

---

## GraphML retention and statistics workflow

If you want post-run graph-folder statistics and comparison pages, keep GraphML during the experiment:

```bash
python experiments_pbc7.py --run_both_state_modes --emit_latex --run_all_T --archive_train_test_folders --keep_step_train_graphml
```

This preserves selected GraphML folders under:

```text
experiments_results/TRACK_ground_truth/kept_graphml/
experiments_results/TRACK_partial_observation/kept_graphml/
```

The retained layout is step-based:

```text
kept_graphml/
└── <step_name>/
    ├── train/
    └── test/
```

These retained folders can then be processed by:

```bash
python run_graph_folder_figures_batch.py --max_graphs 70%
```

That batch script scans `kept_graphml/`, detects step folders that contain both `train/` and `test/`, runs `graph_folder_figures.py` in compare mode for each eligible step, and writes summary outputs including `statistics.tex`.

Important distinction:

- `--run_graph_folder_figures` inside `experiments_pbc7.py` invokes `graph_folder_figures.py` directly during the pipeline.
- `run_graph_folder_figures_batch.py` is a separate post-processing step over retained GraphML.

---

## Overleaf export

When `--emit_latex` is enabled, the pipeline assembles an Overleaf-ready package by collecting archived figures and generating `latex.txt` with reusable figure and table blocks.

Example:

```bash
python experiments_pbc7.py --run_both_state_modes --emit_latex
```

Default export location:

```text
experiments_results/overleaf_package/
```

Typical contents:

- `figures/`
- `latex.txt`

The Overleaf package can include:

- main predictive figures
- cross-track comparison figures
- metrics tables
- tuning tables
- dataset diagnostic figures
- translational attribution figures

To rebuild only the manuscript figure package from an already completed results root:

```bash
python experiments_pbc7.py --emit_latex_only
```

---

## Reproducibility and design notes

This repository is structured for repeatable experimentation.

Key safeguards in the current driver include:

- deterministic per-track working directories
- validation of required scripts before execution
- strict folder and label integrity checks
- task-aware label validation
- contiguity checks across simulations and days
- frozen-test restoration before canonical training runs
- optional graph-folder figure generation before cleanup
- automatic GraphML cleanup to control storage growth
- selective GraphML retention when `--keep_step_train_graphml` is used
- optional full GraphML retention when `--keep_graphml` is used
- archiving of step-specific training outputs
- optional archiving of the exact train/test folders used in each experiment
- isolated subprocess-based dual-track execution
- optional lightweight tuning for Step 4 and Step 8

Two practical points:

1. The pipeline writes into a deterministic results layout rather than timestamped experiment folders by default.
2. The checked-in reproducibility workflow is `experiments_pbc7.py`.

---

## Citation

If you use this repository, please cite the associated manuscript.
