# AMR Digital Twin (Causal): Intervention-Conditioned Policy Learning and Constrained Decision Support

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![PyTorch%20Geometric](https://img.shields.io/badge/PyTorch%20Geometric-enabled-orange)
![Pyomo](https://img.shields.io/badge/Pyomo-MILP-green)
![Workflow](https://img.shields.io/badge/Workflow-Reproducible-success)

A reproducible research platform for constructing intervention-conditioned causal datasets from hospital antimicrobial resistance (AMR) simulations, training temporal graph neural networks to predict action-conditioned outcomes from shared pre-intervention states, evaluating learned policy ranking against simulator-defined oracle actions, and optimizing feasible intervention choices through a constrained mixed-integer linear programming layer.

This repository integrates four connected layers:

1. **Mechanistic hospital simulation** under temporal heterogeneity, including surveillance, isolation, seasonality, and superspreader effects.
2. **Causal policy dataset construction** by branching multiple candidate interventions from the same pre-intervention hospital state.
3. **Action-conditioned temporal graph learning** for baseline-relative improvement prediction and within-state policy ranking.
4. **Constrained policy optimization** through a MILP layer operating on learned state-action utilities.

The current recommended reproducibility entrypoint in this repository is:

```bash
python experiments_causal.py --run_both_state_modes --emit_latex
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
- [Current default causal configuration](#current-default-causal-configuration)
- [Outputs](#outputs)
- [Overleaf export](#overleaf-export)
- [Reproducibility and design notes](#reproducibility-and-design-notes)
- [Citation](#citation)

---

## Overview

This repository is the causal-policy branch of the AMR digital-twin workflow. It is designed to answer a different question from the predictive benchmark. Instead of asking only what mechanism is likely to dominate future resistant emergence, it asks:

**given the same observed hospital state, which candidate intervention is predicted to produce the best downstream outcome?**

The platform builds causal policy datasets by extracting shared pre-intervention temporal graph windows and then branching a finite intervention library from each decision state. Each resulting sample pairs:

- a common graph-history window,
- a specific candidate action,
- and a simulator-defined post-intervention outcome.

The repository supports the same two state-observation tracks used in the broader platform:

- **`ground_truth`**
- **`partial_observation`**

The repository’s reproducibility workflow and CLI are defined by **`experiments_causal.py`**.

---

## Core capabilities

### 1) Mechanistic AMR hospital simulation for causal branching

The causal workflow reuses the hospital AMR simulator, but under a long-horizon temporal-heterogeneity regime intended for intervention analysis rather than only benchmark classification.

The simulation layer supports:

- patients and staff as distinct node types
- directed, weighted contact graphs
- ward assignment, home-ward structure, and multi-ward staffing
- admission-based importation pressure
- within-hospital transmission
- antibiotic exposure and within-host selection effects
- screening and isolation interventions
- observation-limited regimes
- temporal seasonality and superspreader dynamics
- explicit branching from a shared pre-intervention state into multiple intervention futures

The simulator writes daily GraphML snapshots together with trajectory metadata and event summaries used downstream for policy-target construction.

### 2) Causal policy dataset construction

`build_causal_policy_dataset.py` constructs the core intervention-conditioned dataset used by the causal experiments.

For each base simulation seed and each selected decision day, it:

- extracts a temporal window of length `T`
- freezes that window as the observed state at decision time
- branches each action from a finite candidate intervention library
- rolls the simulator forward under each action
- computes horizon-specific policy outcomes
- writes a policy manifest linking windows, actions, and targets

This creates repeated state-action samples of the form:

```text
(shared history window, action) -> post-intervention outcome
```

The current causal workflow is centered on **baseline-relative improvement in future transmission-driven resistant burden over horizon 14**, operationalized through:

```text
y_h14_trans_res_gain
```

### 3) Graph conversion and policy-manifest temporal datasets

`convert_to_pt.py` converts GraphML outputs into `.pt` graph objects for temporal graph learning.

`temporal_graph_dataset.py` supports the policy-manifest mode used by the causal branch, allowing the trainer to build temporal windows from manifest-defined state-action samples rather than only from standard forecasting trajectories.

Converted samples can store:

- node features
- edge features
- graph-level metadata
- action descriptors
- stable node identifiers for traceability
- policy targets and auxiliary policy labels
- split information inherited from seed-level partitioning

This preserves the causal design principle that all action-conditioned samples derived from the same underlying seed remain in the same split.

### 4) Action-conditioned temporal graph learning

`train_amr_dygformer.py` is used in action-conditioning mode to train the causal policy model.

The current causal stack combines:

- a daily graph encoder built around **GraphSAGE-style message passing**
- a **Transformer-based temporal encoder** over windows of length `T`
- action conditioning through projected action descriptors
- a main regression objective on the intervention-conditioned target
- optional auxiliary policy supervision
- optional within-state pairwise ranking loss

In the current causal driver, the policy-learning configuration is aligned to:

```text
task = transmission_resistant_burden_gain_h14
use_action_conditioning = true
```

This means the model is trained to score candidate interventions from the same state according to predicted baseline-relative improvement in future transmission-driven resistant burden.

### 5) Held-out policy evaluation

`evaluate_policy_selector.py` evaluates the learned action-conditioned predictor on held-out decision states.

It compares the model-selected action against the simulator-defined oracle action using metrics such as:

- policy accuracy
- top-k policy accuracy
- regret
- baseline improvement of selected actions
- exported action-score tables

This stage is designed to answer whether the learned graph-temporal representation can recover the within-state ranking of candidate actions.

### 6) Constrained MILP decision support

`optimize_policy_milp.py` turns held-out action scores into explicit feasible decisions.

The current proof-of-concept decision layer supports:

- one action per decision state
- baseline-relative utility transforms
- action-specific costs
- optional budget constraints
- maximum-use limits
- cooldown constraints
- solver-based exact optimization through Pyomo

The default workflow uses a **MILP policy layer** rather than embedding the simulator directly into the optimization loop.

### 7) End-to-end causal reproducibility workflow

`experiments_causal.py` orchestrates the full causal workflow, including:

- causal dataset building
- action-conditioned model training
- held-out policy evaluation
- MILP policy optimization
- dual-track execution across `ground_truth` and `partial_observation`
- archive generation
- Overleaf-ready manuscript export

---

## Repository layout

### Core pipeline

- `experiments_causal.py` — reproducibility driver for the full causal workflow
- `build_causal_policy_dataset.py` — intervention-conditioned dataset builder
- `convert_to_pt.py` — GraphML to PyTorch Geometric conversion
- `train_amr_dygformer.py` — temporal graph learning trainer with action conditioning support
- `temporal_graph_dataset.py` — temporal dataset construction including policy-manifest mode
- `models_amr.py` or `models.py` — model definitions, depending on repository naming
- `tasks.py` — registered forecasting and policy-learning targets
- `evaluate_policy_selector.py` — held-out policy evaluation
- `optimize_policy_milp.py` — constrained MILP decision layer
- `candidate_interventions.json` — finite intervention library used for branching
- `milp_policy_config.json` — MILP configuration for action costs, cooldowns, and related constraints

### Utilities and support scripts

Depending on the repository snapshot, additional utilities may include:

- task inspection utilities
- figure-export scripts
- dataset audit scripts
- conversion helpers
- reporting helpers for policy outputs and action-score exports

### R / Shiny interface

If the causal repository also includes a Shiny frontend or related helpers, these should be documented alongside the main Python workflow in the checked-in repository state.

---

## Environment reproduction

The codebase expects a Python environment with PyTorch, PyTorch Geometric, and Pyomo support.

At minimum, confirm the availability of:

```bash
python -c "import torch, torch_geometric; print('torch', torch.__version__); print('pyg', torch_geometric.__version__)"
```

For MILP support, also confirm:

```bash
python -c "import pyomo; print('pyomo ok')"
```

If your environment uses sparse PyG extensions, verify:

```bash
python -c "import torch, torch_geometric, torch_sparse, torch_scatter; print('torch', torch.__version__); print('pyg', torch_geometric.__version__); print('torch_sparse', torch_sparse.__version__); print('torch_scatter', torch_scatter.__version__)"
```

If you use a commercial solver such as CPLEX, ensure it is installed and accessible from the environment used to run `optimize_policy_milp.py`.

A good practice is to export the full conda environment used for the causal experiments and store it alongside the repository for exact reconstruction.

---

## Quick start

### 1) Build a causal policy dataset

```bash
python build_causal_policy_dataset.py \
  --out_root causal_demo \
  --state_mode ground_truth \
  --candidate_interventions_json candidate_interventions.json \
  --window_T 7 \
  --horizons 14 \
  --decision_days 7:35:7 \
  --decision_stride 1 \
  --action_start_mode branch_at_decision_day \
  --decision_applies_from next_day \
  --include_baseline
```

### 2) Convert GraphML outputs to `.pt`

```bash
python convert_to_pt.py --graphml_dir causal_demo
```

### 3) Train the action-conditioned model manually

```bash
python train_amr_dygformer.py \
  --data_folder causal_demo \
  --task transmission_resistant_burden_gain_h14 \
  --T 7 \
  --use_action_conditioning true \
  --action_hidden_dim 16 \
  --epochs 20
```

### 4) Evaluate the held-out policy selector

```bash
python evaluate_policy_selector.py --help
```

### 5) Run the MILP decision layer

```bash
python optimize_policy_milp.py --help
```

---

## Full reproducibility pipeline

The recommended full causal run is:

```bash
python experiments_causal.py --run_both_state_modes --emit_latex
```

This pipeline is designed to be:

- deterministic in folder layout
- resumable by stage
- track-aware across `ground_truth` and `partial_observation`
- suitable for paper-grade archive generation
- compatible with held-out policy evaluation and downstream optimization

### Dual-track execution

When `--run_both_state_modes` is used, the driver runs the two state modes as isolated track-specific workflows and writes separate outputs per track.

### Canonical causal workflow structure

The current causal pipeline is organized around four main stages:

- **STEP C1**: build the causal policy dataset
- **STEP C2**: train the action-conditioned temporal GNN
- **STEP C3**: evaluate policy selection on held-out decision states
- **STEP C4**: optimize a feasible policy with MILP over the scored candidate actions

### Typical causal design

The standard design uses:

- a long-horizon temporal-heterogeneity simulation regime
- explicit decision days
- a fixed intervention library
- baseline inclusion
- disjoint train / validation / test seed sets
- policy targets aligned to horizon-14 baseline-relative improvement

### Single-track runs

Ground-truth mode only:

```bash
DT_STATE_MODE=ground_truth python experiments_causal.py --emit_latex
```

Partial-observation mode only:

```bash
DT_STATE_MODE=partial_observation python experiments_causal.py --emit_latex
```

---

## Resuming and partial reruns

The pipeline is resumable by stage.

### Example: rerun training through optimization

```bash
python experiments_causal.py --run_both_state_modes --start C2 --stop C4 --emit_latex
```

### Example: rebuild only the dataset

```bash
python experiments_causal.py --run_both_state_modes --start C1 --stop C1
```

### Example: reevaluate and rerun MILP from existing trained outputs

```bash
python experiments_causal.py --run_both_state_modes --start C3 --stop C4 --no_train --emit_latex
```

### Example: reuse prepared datasets and retrain only policy models

```bash
python experiments_causal.py --run_both_state_modes --start C2 --stop C4 --no_simulation --emit_latex
```

Supported stage keys typically include:

- `C1`
- `C2`
- `C3`
- `C4`

Use the exact stage-key syntax implemented by the checked-in `experiments_causal.py` in your repository snapshot.

---

## Important command-line options

Commonly used options in `experiments_causal.py` typically include:

```bash
--start
--stop
--dry_run
--run_both_state_modes
--emit_latex
--emit_latex_only
--no_train
--no_simulation
--results_parent
--overleaf_dir
```

Depending on the checked-in driver, the causal workflow may also expose options controlling:

```bash
--candidate_interventions_json
--include_baseline
--window_T
--horizons
--decision_days
--decision_stride
--action_start_mode
--decision_applies_from
```

Consult:

```bash
python experiments_causal.py --help
```

for the exact CLI of the repository version you are running.

---

## Current default causal configuration

The current causal workflow is centered on:

### Policy-learning target

```text
task = transmission_resistant_burden_gain_h14
oracle target = y_h14_trans_res_gain
selection direction = larger is better
```

### Typical temporal settings

```text
window T = 7
horizon H = 14
decision days = periodic within long-horizon trajectories
sliding_step = 1
```

### Typical model settings

```text
use_action_conditioning = true
action_hidden_dim = 16
hidden = 32
heads = 2
dropout = 0.2
transformer_layers = 2
sage_layers = 3
batch_size = 16
epochs = 20
lr = 1e-4
neighbor_sampling = true
num_neighbors = 15,10
seed_count = 256
seed_strategy = random
seed_batch_size = 64
max_sub_batches = 4
max_neighbors = 20
emit_translational_figures = false
```

### Policy-aligned losses

Typical recent settings include:

```text
aux_policy_loss = true
pairwise_policy_ranking_loss = true
aux_policy_target_name = y_h14_trans_res_gain
aux_policy_loss_weight = 1
pairwise_policy_ranking_weight = 2.0
pairwise_policy_margin = 0.1
pairwise_policy_min_target_gap = 1
```

### Causal simulation regime

Typical recent runs have used:

```text
num_days = 180
seasonal admission-importation forcing = enabled
superspreader episode = enabled
```

### Intervention library

The default action set includes:

- baseline
- screening every 3 days
- screening every 7 days with admission screening
- isolation regime A
- isolation regime B
- one-day screening delay

The exact definitions are taken from `candidate_interventions.json`.

### MILP layer

Typical recent settings include:

```text
score_transform = baseline_delta
solver = cplex
time_limit_sec = 300
mip_gap = 0.0
```

The exact optimization constraints are taken from `milp_policy_config.json`.

---

## Outputs

A typical full run writes to a results root such as:

```text
experiments_causal_results/
├── run_report.txt
├── TRACK_ground_truth/
│   └── work/
│       ├── causal_policy_temporal_heterogeneity/
│       ├── policy_training*/
│       ├── policy_evaluation*/
│       ├── milp_outputs*/
│       └── repro_artifacts_causal/
└── TRACK_partial_observation/
    └── work/
        ├── causal_policy_temporal_heterogeneity/
        ├── policy_training*/
        ├── policy_evaluation*/
        ├── milp_outputs*/
        └── repro_artifacts_causal/
```

Outputs may include:

- policy manifests
- converted `.pt` graph datasets
- trained model checkpoints
- validation and test metrics
- action-score CSV exports
- regret and policy-accuracy summaries
- MILP solution summaries
- archived artifacts for manuscript integration
- causal figures and tables prepared for Overleaf export

---

## Overleaf export

When `--emit_latex` is enabled, the driver assembles an Overleaf-ready package by collecting archived figures and generating reusable LaTeX blocks.

Example:

```bash
python experiments_causal.py --run_both_state_modes --emit_latex
```

Typical export location:

```text
experiments_causal_results/overleaf_package2/
```

Typical contents:

- `figures/`
- `latex.txt`

This package may include:

- causal evaluation figures
- policy tables
- action-ranking summaries
- MILP results summaries
- appendix-ready parameter tables

To rebuild only the manuscript package from existing results:

```bash
python experiments_causal.py --emit_latex_only
```

---

## Reproducibility and design notes

This repository is structured for repeatable causal-policy experimentation.

Key safeguards in the current workflow include:

- deterministic per-track working directories
- explicit seed-level split assignment
- prevention of leakage across action-conditioned branches from the same seed
- manifest-based temporal dataset construction
- explicit baseline inclusion in the intervention library
- held-out policy evaluation against simulator-defined oracle actions
- separation between learned action scoring and downstream optimization
- support for dual-track execution across `ground_truth` and `partial_observation`
- archive generation for manuscript integration

Two practical points:

1. The causal workflow is distinct from the predictive benchmark workflow and should be reproduced with `experiments_causal.py` rather than `experiments_pbc7.py`.
2. The MILP layer is a downstream constrained selector over learned state-action utilities; it is not yet a fully simulator-embedded hospital-control optimizer.

---

## Citation

If you use this repository, please cite the associated manuscript and the causal-policy extension of the AMR digital-twin framework.
