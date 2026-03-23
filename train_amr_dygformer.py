#!/usr/bin/env python3
"""
train_amr_dygformer.py
=============================================================

AMR-only trainer that matches the previous Shiny (run.R) behavior:

- Trains on: synthetic_amr_graphs_train (passed via --data_folder)
- Tests on: synthetic_amr_graphs_test (auto-detected next to this script)
- Emits progress markers for Shiny progress bar:
    DT_PROGRESS_META epochs=<E>
    DT_PROGRESS_EPOCH <e>
    DT_PROGRESS_TEST_META batches=<B>
    DT_PROGRESS_TEST_BATCH <b>

- Writes plots into an output directory (default: ./training_outputs):
    loss_curves.png
    confusion_matrix.png
    roc_curve.png
    confusion_matrix_test.png
    roc_curve_test.png
    trained_model.pt

Adds OPTIONAL "true GraphSAGE-style" neighbor sampling via PyG NeighborLoader:

  --neighbor_sampling true
  --num_neighbors "15,10"
  --seed_count 256
  --seed_strategy random|all
  --seed_batch_size 64
  --max_sub_batches 4

Precedence when --use_task_hparams is enabled:
  CLI sampling flags (from Shiny) > task.train_config sampling fields > script defaults

UPDATED (trajectory metadata):
- TemporalGraphDataset now prefers Data.sim_id + Data.day to build sequential windows.
- This script can enforce metadata via:
    --require_pt_metadata
    --fail_on_noncontiguous
- Trajectory-level split is based on sim_id (trajectory id), not filename prefix.

UPDATED (clean multi-horizon support):
- New flag --out_dir to avoid output overwrites when experiments.py loops over horizons.

(RESTORED) Attention heatmaps:
- Accepts and USES:
    --attn_top_k
    --attn_rank_by
- Produces:
    attention_heatmap.png / attention_heatmap.csv
    attention_heatmap_test.png / attention_heatmap_test.csv
"""

import argparse
import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

import matplotlib.pyplot as plt

from temporal_graph_dataset import TemporalGraphDataset, collate_temporal_graph_batch
from models_amr import AMRDyGFormer
from tasks import TASK_REGISTRY, BaseTask, get_task

from torch_geometric.loader import NeighborLoader


# =============================================================================
# CLI helpers
# =============================================================================

def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("yes", "true", "t", "y", "1"):
        return True
    if s in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_num_neighbors(s: Optional[str]) -> List[int]:
    if s is None:
        return []
    s = str(s).strip()
    if s == "":
        return []
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    out: List[int] = []
    for p in parts:
        out.append(int(p))
    return out


# =============================================================================
# Legacy edge-thinning (kept)
# =============================================================================

def _subsample_edges_per_graph(g, max_neighbors: int):
    if max_neighbors <= 0:
        return g

    edge_index = g.edge_index
    edge_attr = getattr(g, "edge_attr", None)
    if edge_index.numel() == 0:
        return g

    dst = edge_index[1].cpu().numpy()
    num_nodes = g.num_nodes
    indices_by_dst = [[] for _ in range(num_nodes)]
    for e_idx, j in enumerate(dst):
        if 0 <= j < num_nodes:
            indices_by_dst[j].append(e_idx)

    keep: List[int] = []
    for group in indices_by_dst:
        if len(group) <= max_neighbors:
            keep.extend(group)
        else:
            choice = np.random.choice(group, size=max_neighbors, replace=False)
            keep.extend(choice.tolist())

    keep = sorted(set(keep))
    if len(keep) == 0:
        return g

    g.edge_index = edge_index[:, keep]
    if edge_attr is not None:
        g.edge_attr = edge_attr[keep]
    return g


def subsample_neighbors_in_batches(batched_graphs, max_neighbors: int):
    if max_neighbors <= 0:
        return batched_graphs
    out = []
    for g in batched_graphs:
        g_cpu = g.cpu()
        out.append(_subsample_edges_per_graph(g_cpu, max_neighbors))
    return out


# =============================================================================
# True GraphSAGE neighbor sampling (NeighborLoader)
# =============================================================================

def _select_seed_nodes(num_nodes: int, seed_count: int, seed_strategy: str) -> torch.Tensor:
    if num_nodes <= 0:
        return torch.empty((0,), dtype=torch.long)

    strat = str(seed_strategy).lower().strip()
    if strat == "all" or seed_count <= 0 or seed_count >= num_nodes:
        return torch.arange(num_nodes, dtype=torch.long)

    perm = torch.randperm(num_nodes)
    return perm[:seed_count].to(torch.long)


def _embed_one_graph_with_neighbor_sampling(
    model: AMRDyGFormer,
    g_data,
    device: torch.device,
    num_neighbors: List[int],
    seed_count: int,
    seed_strategy: str,
    max_sub_batches: int,
    seed_batch_size: int,
) -> torch.Tensor:
    num_nodes = int(g_data.num_nodes)
    seeds = _select_seed_nodes(num_nodes=num_nodes, seed_count=seed_count, seed_strategy=seed_strategy)

    if seeds.numel() == 0:
        return torch.zeros((model.hidden_channels,), device=device, dtype=torch.float32)

    g_cpu = g_data.cpu()
    seeds = seeds.cpu()

    bs = int(min(seed_batch_size, seeds.numel()))
    loader = NeighborLoader(
        g_cpu,
        input_nodes=seeds,
        num_neighbors=num_neighbors,
        batch_size=bs,
        shuffle=True,
    )

    pooled_list: List[torch.Tensor] = []
    for b_idx, sub in enumerate(loader):
        if max_sub_batches > 0 and b_idx >= max_sub_batches:
            break

        sub = sub.to(device)
        x = sub.x
        edge_index = sub.edge_index
        edge_attr = getattr(sub, "edge_attr", None)

        batch_vec = torch.zeros((x.size(0),), device=device, dtype=torch.long)
        pooled = model.gnn(x, edge_index, edge_attr, batch_vec)
        pooled_list.append(pooled.squeeze(0))

    if len(pooled_list) == 0:
        return torch.zeros((model.hidden_channels,), device=device, dtype=torch.float32)

    return torch.stack(pooled_list, dim=0).mean(dim=0)


def build_day_embeddings_neighbor_sampling(
    model: AMRDyGFormer,
    batched_graphs: List,
    device: torch.device,
    num_neighbors: List[int],
    seed_count: int,
    seed_strategy: str,
    max_sub_batches: int,
    seed_batch_size: int,
) -> torch.Tensor:
    day_embs: List[torch.Tensor] = []
    for g_batch in batched_graphs:
        data_list = g_batch.to_data_list()
        emb_rows: List[torch.Tensor] = []
        for g_data in data_list:
            emb_i = _embed_one_graph_with_neighbor_sampling(
                model=model,
                g_data=g_data,
                device=device,
                num_neighbors=num_neighbors,
                seed_count=seed_count,
                seed_strategy=seed_strategy,
                max_sub_batches=max_sub_batches,
                seed_batch_size=seed_batch_size,
            )
            emb_rows.append(emb_i)
        day_emb = torch.stack(emb_rows, dim=0)
        day_embs.append(day_emb)

    H = torch.stack(day_embs, dim=1)
    return H


# =============================================================================
# ROC / confusion-matrix plotting helpers
# =============================================================================

def _get_class_names(task, n_classes: int):
    names = getattr(task, "class_names", None)
    if isinstance(names, (list, tuple)) and len(names) >= n_classes:
        return list(names)[:n_classes]
    return [f"class {i}" for i in range(n_classes)]


def plot_confusion_matrix_annotated(
    cm: np.ndarray,
    out_path: str,
    title: str,
    class_names: Optional[List[str]] = None,
):
    cm = np.asarray(cm)
    n_classes = int(cm.shape[0])

    if class_names is None or len(class_names) < n_classes:
        class_names = [str(i) for i in range(n_classes)]
    else:
        class_names = [str(x) for x in class_names[:n_classes]]

    row_sums = cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    fig_h = 5.0 if n_classes <= 4 else min(12.0, 4.0 + 0.45 * n_classes)
    fig_w = 6.5 if n_classes <= 4 else min(12.0, 5.0 + 0.45 * n_classes)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(norm_cm, cmap="cividis", vmin=0.0, vmax=1.0)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized proportion", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Predicted", fontsize=15)
    ax.set_ylabel("True", fontsize=15)
    ax.set_xticks(list(range(n_classes)))
    ax.set_xticklabels(class_names, rotation=25, ha="right", fontsize=13)
    ax.set_yticks(list(range(n_classes)))
    ax.set_yticklabels(class_names, fontsize=13)

    ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            pct = norm_cm[i, j] * 100.0
            txt = f"{int(cm[i, j])}\n{pct:.1f}%"
            color = "white" if norm_cm[i, j] >= threshold else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=11, color=color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

def plot_roc_curves(y_true, probs, n_classes: int, out_path: str, title: str, class_names=None):
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs)

    plt.figure(figsize=(7.8, 5.8))
    plotted_any = False

    if y_true.size == 0:
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.6, label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.text(0.5, 0.5, "No samples to plot", ha="center", va="center")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.28)
        plt.tight_layout()
        plt.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close()
        return

    if n_classes == 2:
        y_score = probs[:, 1]
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=False)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2.8, label=f"Model (AUC = {roc_auc:.3f})")
            plt.fill_between(fpr, tpr, alpha=0.15)
            plotted_any = True
        except ValueError as e:
            print(f"⚠️ ROC curve skipped ({title}): {e}")

    else:
        classes = list(range(n_classes))
        y_bin = label_binarize(y_true, classes=classes)
        if y_bin.ndim == 1:
            y_bin = y_bin.reshape(-1, 1)

        if class_names is None:
            class_names = [f"class {i}" for i in classes]

        for i in classes:
            positives = float(y_bin[:, i].sum())
            if positives == 0.0 or positives == float(len(y_bin)):
                print(f"⚠️ ROC for '{class_names[i]}' skipped ({title}): class not present in this split.")
                continue
            try:
                fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], probs[:, i], drop_intermediate=False)
                auc_i = auc(fpr_i, tpr_i)
                plt.plot(fpr_i, tpr_i, linewidth=2.2, label=f"{class_names[i]} (AUC = {auc_i:.3f})")
                plotted_any = True
            except ValueError as e:
                print(f"⚠️ ROC for '{class_names[i]}' skipped ({title}): {e}")

        try:
            fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs[:, :n_classes].ravel(), drop_intermediate=False)
            auc_micro = auc(fpr_micro, tpr_micro)
            plt.plot(
                fpr_micro,
                tpr_micro,
                linestyle=":",
                linewidth=2.4,
                label=f"Micro-average (AUC = {auc_micro:.3f})",
            )
            plotted_any = True
        except Exception as e:
            print(f"⚠️ Micro-average ROC skipped ({title}): {e}")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if plotted_any:
        plt.legend(
        loc="lower right",
        fontsize=15,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="black",
        borderpad=0.6,
        labelspacing=0.4,
        handlelength=2.0,
    )
    else:
        plt.text(0.5, 0.5, "ROC undefined (single-class labels)", ha="center", va="center")
    plt.grid(alpha=0.28)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def _classification_split_summary(
    task: BaseTask,
    y_true: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    probs = np.asarray(probs)
    preds = np.asarray(preds).astype(int).reshape(-1)

    from sklearn.metrics import accuracy_score

    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true,
        preds,
        labels=list(range(task.out_dim)),
        average=None,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, preds, labels=list(range(task.out_dim)))
    unique_classes = sorted(set(int(x) for x in y_true.tolist()))

    roc_details: Dict[str, Any] = {}
    try:
        if len(unique_classes) >= 2:
            if task.out_dim == 2:
                roc_details["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
            else:
                roc_details["roc_auc_macro_ovr"] = float(
                    roc_auc_score(
                        y_true,
                        probs[:, : task.out_dim],
                        multi_class="ovr",
                        average="macro",
                    )
                )
    except Exception:
        pass

    per_class = []
    for i in range(task.out_dim):
        per_class.append(
            {
                "class_index": int(i),
                "class_name": str(class_names[i]) if i < len(class_names) else f"class {i}",
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1": float(f1_per_class[i]),
                "support": int(support_per_class[i]),
            }
        )

    out = {
        "n_samples": int(y_true.shape[0]),
        "class_names": [str(x) for x in class_names],
        "metrics": {
            "accuracy": float(accuracy_score(y_true, preds)),
            "precision_macro": float(np.mean(precision_per_class)) if precision_per_class.size > 0 else 0.0,
            "recall_macro": float(np.mean(recall_per_class)) if recall_per_class.size > 0 else 0.0,
            "f1_macro": float(np.mean(f1_per_class)) if f1_per_class.size > 0 else 0.0,
            "precision_weighted": float(
                np.average(precision_per_class, weights=support_per_class)
            ) if support_per_class.sum() > 0 else 0.0,
            "recall_weighted": float(
                np.average(recall_per_class, weights=support_per_class)
            ) if support_per_class.sum() > 0 else 0.0,
            "f1_weighted": float(
                np.average(f1_per_class, weights=support_per_class)
            ) if support_per_class.sum() > 0 else 0.0,
            **roc_details,
        },
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "y_true": y_true.tolist(),
        "y_pred": preds.tolist(),
        "probabilities": probs.tolist(),
    }
    return out


def _regression_split_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = float(np.mean((y_pred - y_true) ** 2)) if y_true.size > 0 else 0.0
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_pred - y_true))) if y_true.size > 0 else 0.0
    return {
        "n_samples": int(y_true.shape[0]),
        "metrics": {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
        },
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def _write_run_summary_files(out_dir: str, summary: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "metrics_summary.json")
    txt_path = os.path.join(out_dir, "metrics_summary.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(summary), f, indent=2)

    lines: List[str] = []
    lines.append("RUN METRICS SUMMARY")
    lines.append("=" * 80)
    lines.append(f"task: {summary.get('task', '')}")
    lines.append(f"is_classification: {summary.get('is_classification', '')}")
    lines.append(f"output_dir: {summary.get('output_dir', '')}")
    lines.append("")

    config = summary.get("config", {})
    if isinstance(config, dict) and config:
        lines.append("CONFIG")
        lines.append("-" * 80)
        for k in sorted(config.keys()):
            lines.append(f"{k}: {config[k]}")
        lines.append("")

    for split_name in ("validation", "test"):
        split = summary.get(split_name, None)
        if not isinstance(split, dict) or not split:
            continue

        lines.append(split_name.upper())
        lines.append("-" * 80)
        lines.append(f"n_samples: {split.get('n_samples', 0)}")

        metrics = split.get("metrics", {})
        if isinstance(metrics, dict):
            for k in sorted(metrics.keys()):
                v = metrics[k]
                if isinstance(v, float):
                    lines.append(f"{k}: {v:.10f}")
                else:
                    lines.append(f"{k}: {v}")

        cm = split.get("confusion_matrix", None)
        if cm is not None:
            lines.append("confusion_matrix:")
            for row in cm:
                lines.append("  " + " ".join(str(int(x)) for x in row))

        per_class = split.get("per_class", None)
        if isinstance(per_class, list) and per_class:
            lines.append("per_class:")
            for row in per_class:
                cname = row.get("class_name", row.get("class_index", ""))
                lines.append(
                    "  "
                    f"{cname}: precision={float(row.get('precision', 0.0)):.10f}, "
                    f"recall={float(row.get('recall', 0.0)):.10f}, "
                    f"f1={float(row.get('f1', 0.0)):.10f}, "
                    f"support={int(row.get('support', 0))}"
                )
        lines.append("")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


# =============================================================================
# Attention heatmaps (RESTORED)
# =============================================================================

def _load_node_vocab_inv(folder: str, filename: str = "node_vocab.json") -> Optional[List[str]]:
    """
    Load node_vocab_inv list where index == node_id.
    Expected JSON:
      { "node_vocab_inv": ["name0","name1", ...] }
    """
    path = os.path.join(os.path.abspath(folder), filename)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        inv = obj.get("node_vocab_inv", None)
        if isinstance(inv, list) and len(inv) > 0:
            return [str(x) for x in inv]
    except Exception:
        return None
    return None


def save_attention_heatmap(
    model: AMRDyGFormer,
    task: BaseTask,
    loader,
    device: torch.device,
    max_neighbors: int,
    neighbor_sampling: bool,
    num_neighbors: List[int],
    seed_count: int,
    seed_strategy: str,
    max_sub_batches: int,
    seed_batch_size: int,
    out_png: str,
    out_csv: str,
    split_name: str,
    node_vocab_inv: Optional[List[str]] = None,
    top_k: int = 50,
    rank_by: str = "abs_diff",
):
    """
    Save a 600dpi PNG + CSV attention heatmap relating node-attention to task performance.

    For classification:
      Groups are Correct vs Incorrect
    For regression:
      Groups are Low error vs High error using median(|err|) split.

    Ranking:
      - abs_diff: rank by |mean(attn_good) - mean(attn_bad)|
      - mean: rank by mean attention across both groups
    """
    model.eval()

    # Defensive defaults
    try:
        top_k = int(top_k)
    except Exception:
        top_k = 50
    if top_k <= 0:
        top_k = 50
    rank_by = str(rank_by).strip().lower()
    if rank_by not in ("abs_diff", "mean"):
        rank_by = "abs_diff"

    # Under NeighborLoader we cannot align attention to full-graph node IDs consistently.
    if neighbor_sampling:
        plt.figure(figsize=(6, 4))
        plt.axis("off")
        plt.text(
            0.5,
            0.5,
            f"Attention heatmap unavailable ({split_name}).\nDisable neighbor sampling to export node attention.",
            ha="center",
            va="center",
        )
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        plt.savefig(out_png, dpi=600, bbox_inches="tight")
        plt.close()
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("note\n")
            f.write("Attention heatmap unavailable under neighbor sampling\n")
        return

    # Per-sample storage:
    #   sample_attn[i] : dict(node_id -> mean attention across days)
    #   sample_perf[i] : classification correctness (1/0) OR regression abs error
    sample_attn: List[Dict[int, float]] = []
    sample_perf: List[float] = []

    with torch.no_grad():
        for batched_graphs, labels_dict in loader:
            batched_graphs = subsample_neighbors_in_batches(batched_graphs, max_neighbors)
            batched_graphs_dev = [g.to(device) for g in batched_graphs]
            labels_dict = {k: v.to(device) for k, v in labels_dict.items()}

            # Predictions
            y_hat = model(batched_graphs_dev)

            # Compute per-day per-sample node attention maps: [T][B]{node_id:attn}
            per_day_maps: List[List[Dict[int, float]]] = []

            for g in batched_graphs_dev:
                x = g.x
                edge_index = g.edge_index
                edge_attr = getattr(g, "edge_attr", None)
                batch_vec = g.batch

                # Expected that gnn supports return_attention=True
                try:
                    _, attn_vec = model.gnn(x, edge_index, edge_attr, batch_vec, return_attention=True)
                except TypeError as e:
                    raise TypeError(
                        "models_amr.AMRDyGFormer.gnn must support return_attention=True to export heatmaps."
                    ) from e

                node_id_vec = getattr(g, "node_id", None)
                if node_id_vec is None:
                    node_id_vec = torch.arange(x.size(0), device=device, dtype=torch.long)

                num_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 0
                day_list: List[Dict[int, float]] = []
                for i in range(num_graphs):
                    idx = (batch_vec == i).nonzero(as_tuple=False).view(-1)
                    if idx.numel() == 0:
                        day_list.append({})
                        continue

                    ids_i = node_id_vec[idx].detach().cpu().numpy().astype(int).tolist()
                    attn_i = attn_vec[idx].detach().cpu().numpy().astype(float).tolist()

                    d: Dict[int, float] = {}
                    c: Dict[int, int] = {}
                    for nid, a in zip(ids_i, attn_i):
                        if nid in d:
                            d[nid] += float(a)
                            c[nid] += 1
                        else:
                            d[nid] = float(a)
                            c[nid] = 1
                    for nid in list(d.keys()):
                        d[nid] /= float(c[nid])
                    day_list.append(d)

                per_day_maps.append(day_list)

            if len(per_day_maps) == 0:
                continue

            B = len(per_day_maps[0])
            T = len(per_day_maps)

            # Average per-node attention across days, per sample
            for i in range(B):
                maps_i = [per_day_maps[t][i] for t in range(T)]
                all_ids = set()
                for d in maps_i:
                    all_ids |= set(d.keys())

                if len(all_ids) == 0:
                    sample_attn.append({})
                    continue

                avg_map: Dict[int, float] = {}
                for nid in all_ids:
                    vals = [d.get(nid, np.nan) for d in maps_i]
                    vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
                    if len(vals) == 0:
                        continue
                    avg_map[int(nid)] = float(np.mean(vals))
                sample_attn.append(avg_map)

            # Performance signal aligned to B
            if task.is_classification:
                probs = F.softmax(y_hat, dim=1)
                pred = probs.argmax(dim=1).detach().cpu().numpy().astype(int)
                y_true = task.get_targets(batched_graphs_dev, labels_dict).detach().cpu().numpy().astype(int).reshape(-1)
                m = min(B, len(pred), len(y_true))
                for j in range(m):
                    sample_perf.append(float(pred[j] == y_true[j]))
            else:
                y_true = task.get_targets(batched_graphs_dev, labels_dict).detach().cpu().numpy().reshape(-1)
                y_pred = y_hat.detach().cpu().numpy().reshape(-1)
                m = min(B, len(y_true), len(y_pred))
                for j in range(m):
                    sample_perf.append(float(abs(y_pred[j] - y_true[j])))

    # Align lengths
    m = min(len(sample_attn), len(sample_perf))
    sample_attn = sample_attn[:m]
    perf = np.asarray(sample_perf[:m], dtype=float)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    if m == 0:
        plt.figure(figsize=(6, 4))
        plt.axis("off")
        plt.text(0.5, 0.5, f"No samples to plot ({split_name}).", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out_png, dpi=600, bbox_inches="tight")
        plt.close()
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("note\n")
            f.write("No samples to plot\n")
        return

    # Group masks and labels
    if task.is_classification:
        good_mask = perf >= 0.5
        bad_mask = ~good_mask
        col_labels = ["Correct", "Incorrect"]
        acc = float(np.mean(perf)) if perf.size else float("nan")
        title = f"Node attention vs performance ({split_name}) | acc={acc:.3f} | n={m}"
    else:
        thr = float(np.median(perf)) if perf.size else 0.0
        good_mask = perf <= thr
        bad_mask = perf > thr
        col_labels = ["Low error", "High error"]
        title = f"Node attention vs error ({split_name}) | median|err|={thr:.3g} | n={m}"

    def _agg(mask: np.ndarray) -> Dict[int, float]:
        sums: Dict[int, float] = {}
        cnts: Dict[int, int] = {}
        idxs = np.where(mask)[0].tolist()
        for k in idxs:
            d = sample_attn[k]
            for nid, a in d.items():
                sums[nid] = sums.get(nid, 0.0) + float(a)
                cnts[nid] = cnts.get(nid, 0) + 1
        out: Dict[int, float] = {}
        for nid, ssum in sums.items():
            c = cnts.get(nid, 0)
            if c > 0:
                out[nid] = ssum / float(c)
        return out

    mean_good = _agg(good_mask)
    mean_bad = _agg(bad_mask)

    node_ids = sorted(set(mean_good.keys()) | set(mean_bad.keys()))
    if len(node_ids) == 0:
        plt.figure(figsize=(6, 4))
        plt.axis("off")
        plt.text(0.5, 0.5, f"No node IDs to plot ({split_name}).", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out_png, dpi=600, bbox_inches="tight")
        plt.close()
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("note\n")
            f.write("No node IDs to plot\n")
        return

    rows: List[Tuple[int, float, float, float, float]] = []
    for nid in node_ids:
        a = float(mean_good.get(nid, np.nan))
        b = float(mean_bad.get(nid, np.nan))
        diff = a - b if (not np.isnan(a) and not np.isnan(b)) else np.nan
        mean_ = float(np.nanmean([a, b]))
        rows.append((int(nid), a, b, diff, mean_))

    if rank_by == "mean":
        rows.sort(key=lambda r: (-(r[4] if not np.isnan(r[4]) else -1e18), r[0]))
    else:
        rows.sort(key=lambda r: (-(abs(r[3]) if not np.isnan(r[3]) else -1e18), r[0]))

    rows = rows[: min(top_k, len(rows))]
    Hm = np.stack([[r[1], r[2]] for r in rows], axis=0)

    # Write CSV
    col1 = col_labels[0].replace(" ", "_")
    col2 = col_labels[1].replace(" ", "_")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(f"node_id,node_name,{col1},{col2},diff\n")
        for nid, a, b, diff, _ in rows:
            name = str(nid)
            if node_vocab_inv is not None and 0 <= nid < len(node_vocab_inv):
                name = str(node_vocab_inv[nid])
            f.write(f"{nid},{name},{a:.10f},{b:.10f},{diff:.10f}\n")

    # Plot heatmap
    fig_h = max(4.0, min(18.0, Hm.shape[0] / 10.0))
    plt.figure(figsize=(7.5, fig_h))
    plt.imshow(Hm, aspect="auto")
    plt.colorbar(label="Mean attention")
    plt.xticks(ticks=[0, 1], labels=col_labels)

    y_labels = []
    for nid, *_ in rows:
        if node_vocab_inv is not None and 0 <= nid < len(node_vocab_inv):
            y_labels.append(str(node_vocab_inv[nid]))
        else:
            y_labels.append(str(nid))

    plt.yticks(ticks=list(range(len(y_labels))), labels=y_labels, fontsize=7)
    plt.ylabel("Nodes (stable ID)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# Train/eval
# =============================================================================

def train_epoch(
    model,
    task,
    loader,
    optimizer,
    device,
    max_neighbors: int,
    neighbor_sampling: bool,
    num_neighbors: List[int],
    seed_count: int,
    seed_strategy: str,
    max_sub_batches: int,
    seed_batch_size: int,
):
    model.train()
    total_loss = 0.0
    n = 0

    for batched_graphs, labels_dict in loader:
        if neighbor_sampling:
            H = build_day_embeddings_neighbor_sampling(
                model=model,
                batched_graphs=batched_graphs,
                device=device,
                num_neighbors=num_neighbors,
                seed_count=seed_count,
                seed_strategy=seed_strategy,
                max_sub_batches=max_sub_batches,
                seed_batch_size=seed_batch_size,
            )
            y_hat = model.forward_from_day_embeddings(H)
            batched_graphs_dev = [g.to(device) for g in batched_graphs]
        else:
            batched_graphs = subsample_neighbors_in_batches(batched_graphs, max_neighbors)
            batched_graphs_dev = [g.to(device) for g in batched_graphs]
            y_hat = model(batched_graphs_dev)

        labels_dict = {k: v.to(device) for k, v in labels_dict.items()}

        optimizer.zero_grad()
        loss = task.compute_loss(y_hat, batched_graphs_dev, labels_dict)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_hat.size(0)
        n += y_hat.size(0)

    return total_loss / max(1, n)


def eval_epoch(
    model,
    task,
    loader,
    device,
    max_neighbors: int,
    neighbor_sampling: bool,
    num_neighbors: List[int],
    seed_count: int,
    seed_strategy: str,
    max_sub_batches: int,
    seed_batch_size: int,
    progress_prefix: str = None,
):
    model.eval()
    total_loss = 0.0
    n = 0

    collected_preds = []
    collected_labels: Dict[str, List[torch.Tensor]] = {}
    collected_graphs = []

    num_batches = len(loader) if hasattr(loader, "__len__") else None
    if progress_prefix is not None and num_batches is not None:
        print(f"DT_PROGRESS_{progress_prefix}_META batches={num_batches}", flush=True)

    with torch.no_grad():
        for batch_idx, (batched_graphs, labels_dict) in enumerate(loader):
            if progress_prefix is not None:
                print(f"DT_PROGRESS_{progress_prefix}_BATCH {batch_idx + 1}", flush=True)

            if neighbor_sampling:
                H = build_day_embeddings_neighbor_sampling(
                    model=model,
                    batched_graphs=batched_graphs,
                    device=device,
                    num_neighbors=num_neighbors,
                    seed_count=seed_count,
                    seed_strategy=seed_strategy,
                    max_sub_batches=max_sub_batches,
                    seed_batch_size=seed_batch_size,
                )
                y_hat = model.forward_from_day_embeddings(H)
                batched_graphs_dev = [g.to(device) for g in batched_graphs]
            else:
                batched_graphs = subsample_neighbors_in_batches(batched_graphs, max_neighbors)
                batched_graphs_dev = [g.to(device) for g in batched_graphs]
                y_hat = model(batched_graphs_dev)

            labels_dict = {k: v.to(device) for k, v in labels_dict.items()}
            loss = task.compute_loss(y_hat, batched_graphs_dev, labels_dict)

            total_loss += loss.item() * y_hat.size(0)
            n += y_hat.size(0)

            collected_preds.append(y_hat.cpu())
            collected_graphs.append([g.cpu() for g in batched_graphs_dev])

            for k, v in labels_dict.items():
                collected_labels.setdefault(k, []).append(v.cpu())

    if len(collected_preds) == 0:
        empty_yhat = torch.empty((0, task.out_dim), dtype=torch.float32)
        return 0.0, {}, empty_yhat, [], {}

    y_hat_all = torch.cat(collected_preds, dim=0)
    labels_dict_all = {k: torch.cat(v, dim=0) for k, v in collected_labels.items()} if collected_labels else {}
    metrics = task.compute_eval_metrics(y_hat_all, collected_graphs, labels_dict_all)

    return total_loss / max(1, n), metrics, y_hat_all, collected_graphs, labels_dict_all


def infer_feature_dims(dataset):
    graphs, _ = dataset[0]
    g = graphs[0]
    in_channels = g.x.size(1)
    edge_dim = g.edge_attr.size(1) if hasattr(g, "edge_attr") else 0
    return in_channels, edge_dim


def _validate_task_labels(task: BaseTask, data_folder: str) -> None:
    try:
        pt_files = [f for f in os.listdir(data_folder) if f.endswith(".pt")]
    except Exception:
        pt_files = []
    if len(pt_files) == 0:
        raise FileNotFoundError(f"No .pt files found in data_folder: {data_folder}")

    sample_path = os.path.join(data_folder, sorted(pt_files)[0])
    sample = torch.load(sample_path, weights_only=False)

    try:
        _ = task.get_targets([sample], {})
    except Exception as e:
        avail = sorted([a for a in dir(sample) if a.startswith("y_")])
        msg = (
            f"Selected task '{task.name}' requires a label that is missing in '{sample_path}'.\n"
            f"Error: {e}\n"
            f"Available y_* attributes include (truncated): {avail[:40]}"
        )
        raise ValueError(msg) from e


def _get_window_sim_id(dataset: TemporalGraphDataset, window_fnames: List[str], cache: Dict[str, str]) -> str:
    """
    Determine the trajectory id for a window.
    Prefer reading Data.sim_id from the first file in the window.
    """
    if not window_fnames:
        return "unknown"
    f0 = window_fnames[0]
    if f0 in cache:
        return cache[f0]

    path = dataset._disk_paths.get(f0, None)
    if path is None:
        cache[f0] = "unknown"
        return "unknown"

    try:
        d = torch.load(path, weights_only=False)
    except Exception:
        cache[f0] = "unknown"
        return "unknown"

    sim_id = getattr(d, "sim_id", None)
    if sim_id is not None and str(sim_id).strip() != "":
        cache[f0] = str(sim_id)
        return cache[f0]

    m = re.match(r"(.+?)_t(\d+)", str(f0))
    prefix = f0 if m is None else m.group(1)
    cache[f0] = str(prefix)
    return cache[f0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)

    parser.add_argument("--T", type=int, default=7)
    parser.add_argument("--sliding_step", type=int, default=1)

    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--sage_layers", type=int, default=3)
    parser.add_argument("--use_cls", action="store_true")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--max_neighbors", type=int, default=20)

    parser.add_argument("--neighbor_sampling", type=str2bool, default=None)
    parser.add_argument("--num_neighbors", type=str, default=None)
    parser.add_argument("--seed_count", type=int, default=None)
    parser.add_argument("--seed_strategy", type=str, default=None)
    parser.add_argument("--seed_batch_size", type=int, default=None)
    parser.add_argument("--max_sub_batches", type=int, default=None)

    # RESTORED: Attention heatmap controls
    parser.add_argument("--attn_top_k", type=int, default=50, help="Top-K nodes to show in attention heatmaps.")
    parser.add_argument(
        "--attn_rank_by",
        type=str,
        default="abs_diff",
        choices=["abs_diff", "mean"],
        help="Rank nodes for heatmap by absolute difference between groups or by mean attention.",
    )

    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--use_task_hparams", action="store_true")

    parser.add_argument(
        "--train_model",
        type=str2bool,
        default=True,
        help="true/false: if false, skip training and only run evaluation using a saved model.",
    )

    parser.add_argument(
        "--require_pt_metadata",
        type=str2bool,
        default=True,
        help="If true, require Data.sim_id and Data.day in all .pt files (recommended).",
    )
    parser.add_argument(
        "--fail_on_noncontiguous",
        type=str2bool,
        default=True,
        help="If true, raise if any non-contiguous temporal windows are detected.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="training_outputs",
        help="Directory for all outputs (plots + trained_model.pt). Default: training_outputs",
    )

    args = parser.parse_args()
    args.data_folder = os.path.abspath(args.data_folder)

    out_dir = os.path.abspath(str(args.out_dir))

    # Script-level defaults (applied if still None after parsing/task override)
    if args.neighbor_sampling is None:
        args.neighbor_sampling = False
    if args.num_neighbors is None:
        args.num_neighbors = "15,10"
    if args.seed_count is None:
        args.seed_count = 256
    if args.seed_strategy is None:
        args.seed_strategy = "random"
    if args.seed_batch_size is None:
        args.seed_batch_size = 64
    if args.max_sub_batches is None:
        args.max_sub_batches = 4

    args.num_neighbors = parse_num_neighbors(args.num_neighbors)
    if len(args.num_neighbors) == 0:
        args.num_neighbors = [15, 10]

    dataset = TemporalGraphDataset(
        folder=args.data_folder,
        T=args.T,
        sliding_step=args.sliding_step,
        prefer_pt_metadata=True,
        require_pt_metadata=bool(args.require_pt_metadata),
        fail_on_noncontiguous=bool(args.fail_on_noncontiguous),
    )

    # -------------------------------------------------------------------------
    # Train/Val split (trajectory-level by sim_id)
    # -------------------------------------------------------------------------
    val_ratio = 0.2
    sim_cache: Dict[str, str] = {}
    sim_to_indices: Dict[str, List[int]] = {}

    for i, fnames in enumerate(dataset.groups):
        sim_id = _get_window_sim_id(dataset, fnames, sim_cache)
        sim_to_indices.setdefault(sim_id, []).append(i)

    sim_ids = sorted(sim_to_indices.keys())
    if len(sim_ids) < 2:
        n_val = max(1, int(len(dataset) * val_ratio))
        n_train = max(1, len(dataset) - n_val)
        train_set, val_set = random_split(dataset, [n_train, n_val])
        print("⚠️ Split fallback: only one trajectory detected; using window-level random_split.")
    else:
        rng = np.random.RandomState(args.split_seed)
        rng.shuffle(sim_ids)

        n_val_sims = int(round(len(sim_ids) * val_ratio))
        n_val_sims = max(1, min(len(sim_ids) - 1, n_val_sims))

        val_sims = set(sim_ids[:n_val_sims])
        train_indices: List[int] = []
        val_indices: List[int] = []

        for sid, idxs in sim_to_indices.items():
            if sid in val_sims:
                val_indices.extend(idxs)
            else:
                train_indices.extend(idxs)

        train_indices = sorted(train_indices)
        val_indices = sorted(val_indices)

        if len(train_indices) == 0 or len(val_indices) == 0:
            n_val = max(1, int(len(dataset) * val_ratio))
            n_train = max(1, len(dataset) - n_val)
            train_set, val_set = random_split(dataset, [n_train, n_val])
            print("⚠️ Split fallback: degenerate trajectory split; using window-level random_split.")
        else:
            train_set = Subset(dataset, train_indices)
            val_set = Subset(dataset, val_indices)
            print(
                f"✅ Trajectory-level split: {len(set(sim_ids) - val_sims)} train sims ({len(train_indices)} windows) | "
                f"{len(val_sims)} val sims ({len(val_indices)} windows) | seed={args.split_seed}"
            )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_temporal_graph_batch,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_temporal_graph_batch,
    )

    # -------------------------------------------------------------------------
    # Test set auto-detect
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_folder = os.path.abspath(os.path.join(script_dir, "synthetic_amr_graphs_test"))
    alt_test_folder = os.path.abspath(os.path.join(script_dir, "synthetic_amr_graphs_test_pt"))

    test_loader = None
    test_dataset = None
    chosen_test_folder = None

    if os.path.isdir(test_folder):
        chosen_test_folder = test_folder
    elif os.path.isdir(alt_test_folder):
        chosen_test_folder = alt_test_folder

    if chosen_test_folder is not None:
        test_dataset = TemporalGraphDataset(
            folder=chosen_test_folder,
            T=args.T,
            sliding_step=args.sliding_step,
            prefer_pt_metadata=True,
            require_pt_metadata=bool(args.require_pt_metadata),
            fail_on_noncontiguous=bool(args.fail_on_noncontiguous),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_temporal_graph_batch,
        )
        print(f"✅ Loaded test dataset from '{chosen_test_folder}' with {len(test_dataset)} windows.")
    else:
        print(
            f"⚠️ Test dataset folder not found at '{test_folder}' (or '{alt_test_folder}'). "
            f"Test evaluation will be skipped."
        )

    in_channels, edge_dim = infer_feature_dims(dataset)

    # -------------------------------------------------------------------------
    # Task resolution
    # -------------------------------------------------------------------------
    if args.task in TASK_REGISTRY:
        task_obj = TASK_REGISTRY[args.task]
        if isinstance(task_obj, BaseTask):
            task = task_obj
        elif isinstance(task_obj, type):
            task = task_obj()
        elif callable(task_obj):
            task = task_obj()
        else:
            raise TypeError(
                f"TASK_REGISTRY['{args.task}'] must be a BaseTask instance or a callable/class returning one; "
                f"got type={type(task_obj)}"
            )
    else:
        try:
            task = get_task(args.task)
        except Exception as e:
            raise ValueError(
                f"Unknown task '{args.task}'. Available registry keys: {list(TASK_REGISTRY.keys())}. "
                f"Also supports horizonised names like '<base>_h<H>' for supported bases. Error: {e}"
            ) from e

    _validate_task_labels(task, args.data_folder)

    # -------------------------------------------------------------------------
    # Task overrides (but do NOT stomp user-provided sampling flags)
    # -------------------------------------------------------------------------
    if args.use_task_hparams:
        model_cfg = getattr(task, "model_config", {}) or {}
        train_cfg = getattr(task, "train_config", {}) or {}

        args.hidden = model_cfg.get("hidden_channels", args.hidden)
        args.heads = model_cfg.get("heads", args.heads)
        args.dropout = model_cfg.get("dropout", args.dropout)
        args.transformer_layers = model_cfg.get("transformer_layers", args.transformer_layers)
        args.sage_layers = model_cfg.get("sage_layers", args.sage_layers)
        args.use_cls = model_cfg.get("use_cls_token", args.use_cls)

        args.batch_size = train_cfg.get("batch_size", args.batch_size)
        args.epochs = train_cfg.get("epochs", args.epochs)
        args.lr = train_cfg.get("lr", args.lr)
        args.max_neighbors = train_cfg.get("max_neighbors", args.max_neighbors)

        args.neighbor_sampling = train_cfg.get("neighbor_sampling", args.neighbor_sampling)
        cfg_nn = train_cfg.get("num_neighbors", None)
        if cfg_nn is not None and args.num_neighbors == [15, 10]:
            args.num_neighbors = parse_num_neighbors(cfg_nn) if isinstance(cfg_nn, str) else list(cfg_nn)
        if train_cfg.get("seed_count", None) is not None and args.seed_count == 256:
            args.seed_count = int(train_cfg["seed_count"])
        if train_cfg.get("seed_strategy", None) is not None and args.seed_strategy == "random":
            args.seed_strategy = str(train_cfg["seed_strategy"])
        if train_cfg.get("seed_batch_size", None) is not None and args.seed_batch_size == 64:
            args.seed_batch_size = int(train_cfg["seed_batch_size"])
        if train_cfg.get("max_sub_batches", None) is not None and args.max_sub_batches == 4:
            args.max_sub_batches = int(train_cfg["max_sub_batches"])

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_temporal_graph_batch,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_temporal_graph_batch,
        )
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_temporal_graph_batch,
            )

    output_activation = str(
        getattr(task, "output_activation", "identity" if task.is_classification else "softplus")
    ).lower().strip()
    use_softplus = (output_activation == "softplus")

    model = AMRDyGFormer(
        in_channels=in_channels,
        hidden_channels=args.hidden,
        edge_dim=edge_dim,
        heads=args.heads,
        T=args.T,
        dropout=args.dropout,
        use_cls_token=args.use_cls,
        n_outputs=task.out_dim,
        n_layers=args.transformer_layers,
        sage_layers=args.sage_layers,
        use_softplus=use_softplus,
        output_activation=output_activation,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    import shutil

    # Only wipe outputs when we are actually training; otherwise preserve the saved model.
    if args.train_model:
        shutil.rmtree(out_dir, ignore_errors=True)

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "trained_model.pt")

    train_losses: List[float] = []
    val_losses: List[float] = []

    print(f"DT_PROGRESS_META epochs={args.epochs}", flush=True)
    print(f"DT_OUT_DIR {out_dir}", flush=True)

    if args.neighbor_sampling:
        print(
            f"✅ Neighbor sampling ENABLED: num_neighbors={args.num_neighbors} | "
            f"seed_count={args.seed_count} | seed_strategy={args.seed_strategy} | "
            f"seed_batch_size={args.seed_batch_size} | max_sub_batches={args.max_sub_batches}"
        )
    else:
        print(f"✅ Neighbor sampling DISABLED: using legacy max_neighbors={args.max_neighbors} edge-thinning.")

    if args.train_model:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(
                model=model,
                task=task,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                max_neighbors=args.max_neighbors,
                neighbor_sampling=args.neighbor_sampling,
                num_neighbors=args.num_neighbors,
                seed_count=args.seed_count,
                seed_strategy=args.seed_strategy,
                max_sub_batches=args.max_sub_batches,
                seed_batch_size=args.seed_batch_size,
            )
            val_loss, val_metrics, _, _, _ = eval_epoch(
                model=model,
                task=task,
                loader=val_loader,
                device=device,
                max_neighbors=args.max_neighbors,
                neighbor_sampling=args.neighbor_sampling,
                num_neighbors=args.num_neighbors,
                seed_count=args.seed_count,
                seed_strategy=args.seed_strategy,
                max_sub_batches=args.max_sub_batches,
                seed_batch_size=args.seed_batch_size,
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            metrics_str = " | ".join(f"{k}={v:.3f}" for k, v in val_metrics.items())
            if metrics_str:
                metrics_str = " | " + metrics_str

            print(f"Epoch {epoch:03d} | train_loss={train_loss:.3f} | val_loss={val_loss:.3f}{metrics_str}")
            print(f"DT_PROGRESS_EPOCH {epoch}", flush=True)

        plt.figure()
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=600, bbox_inches="tight")
        plt.close()

        torch.save(model.state_dict(), model_path)
    else:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"--train_model=false but no trained model found at '{model_path}'.")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

    _, _, val_y_hat_all, val_graphs_list, val_labels_all = eval_epoch(
        model=model,
        task=task,
        loader=val_loader,
        device=device,
        max_neighbors=args.max_neighbors,
        neighbor_sampling=args.neighbor_sampling,
        num_neighbors=args.num_neighbors,
        seed_count=args.seed_count,
        seed_strategy=args.seed_strategy,
        max_sub_batches=args.max_sub_batches,
        seed_batch_size=args.seed_batch_size,
    )

    test_y_hat_all = None
    test_graphs_list = None
    test_labels_all = None

    if test_loader is not None:
        _, _, test_y_hat_all, test_graphs_list, test_labels_all = eval_epoch(
            model=model,
            task=task,
            loader=test_loader,
            device=device,
            max_neighbors=args.max_neighbors,
            neighbor_sampling=args.neighbor_sampling,
            num_neighbors=args.num_neighbors,
            seed_count=args.seed_count,
            seed_strategy=args.seed_strategy,
            max_sub_batches=args.max_sub_batches,
            seed_batch_size=args.seed_batch_size,
            progress_prefix="TEST",
        )

    run_summary: Dict[str, Any] = {
        "task": str(task.name),
        "is_classification": bool(task.is_classification),
        "output_dir": out_dir,
        "config": {
            "data_folder": args.data_folder,
            "test_folder": chosen_test_folder,
            "T": int(args.T),
            "sliding_step": int(args.sliding_step),
            "hidden": int(args.hidden),
            "heads": int(args.heads),
            "dropout": float(args.dropout),
            "transformer_layers": int(args.transformer_layers),
            "sage_layers": int(args.sage_layers),
            "use_cls": bool(args.use_cls),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "max_neighbors": int(args.max_neighbors),
            "neighbor_sampling": bool(args.neighbor_sampling),
            "num_neighbors": list(args.num_neighbors),
            "seed_count": int(args.seed_count),
            "seed_strategy": str(args.seed_strategy),
            "seed_batch_size": int(args.seed_batch_size),
            "max_sub_batches": int(args.max_sub_batches),
            "split_seed": int(args.split_seed),
            "train_model": bool(args.train_model),
        },
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses],
    }

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    if task.is_classification:
        from sklearn.metrics import confusion_matrix

        y_true_val = task.get_targets(val_graphs_list, val_labels_all).cpu().numpy().astype(int).reshape(-1)
        probs_val = F.softmax(val_y_hat_all, dim=1).cpu().numpy()
        preds_val = probs_val.argmax(axis=1)

        class_names = _get_class_names(task, task.out_dim)

        cm_val = confusion_matrix(y_true_val, preds_val, labels=list(range(task.out_dim)))
        plot_confusion_matrix_annotated(
            cm=cm_val,
            out_path=os.path.join(out_dir, "confusion_matrix.png"),
            title="Confusion Matrix (Validation)",
            class_names=class_names,
        )

        plot_roc_curves(
            y_true=y_true_val,
            probs=probs_val,
            n_classes=task.out_dim,
            out_path=os.path.join(out_dir, "roc_curve.png"),
            title="ROC Curve (Validation)",
            class_names=class_names,
        )

        run_summary["validation"] = _classification_split_summary(
            task=task,
            y_true=y_true_val,
            probs=probs_val,
            preds=preds_val,
            class_names=class_names,
        )

        if (test_loader is not None) and (test_y_hat_all is not None) and test_y_hat_all.size(0) > 0:
            y_true_test = task.get_targets(test_graphs_list, test_labels_all).cpu().numpy().astype(int).reshape(-1)
            probs_test = F.softmax(test_y_hat_all, dim=1).cpu().numpy()
            preds_test = probs_test.argmax(axis=1)

            cm_test = confusion_matrix(y_true_test, preds_test, labels=list(range(task.out_dim)))
            plot_confusion_matrix_annotated(
                cm=cm_test,
                out_path=os.path.join(out_dir, "confusion_matrix_test.png"),
                title="Confusion Matrix (Test)",
                class_names=class_names,
            )

            plot_roc_curves(
                y_true=y_true_test,
                probs=probs_test,
                n_classes=task.out_dim,
                out_path=os.path.join(out_dir, "roc_curve_test.png"),
                title="ROC Curve (Test)",
                class_names=class_names,
            )

            run_summary["test"] = _classification_split_summary(
                task=task,
                y_true=y_true_test,
                probs=probs_test,
                preds=preds_test,
                class_names=class_names,
            )
        elif test_loader is not None:
            print("⚠️ Test dataset produced 0 windows for this T/sliding_step; skipping test plots.")
    else:
        y_true_val = task.get_targets(val_graphs_list, val_labels_all).cpu().numpy().reshape(-1)
        y_pred_val = val_y_hat_all.cpu().numpy().reshape(-1)

        def plot_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, title: str):
            y_true = np.asarray(y_true).reshape(-1)
            y_pred = np.asarray(y_pred).reshape(-1)
            plt.figure(figsize=(6, 5))
            plt.scatter(y_true, y_pred, s=12, alpha=0.7)
            if y_true.size > 0:
                lo = float(min(y_true.min(), y_pred.min()))
                hi = float(max(y_true.max(), y_pred.max()))
                plt.plot([lo, hi], [lo, hi], linestyle="--")
            plt.xlabel("True")
            plt.ylabel("Predicted")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(out_path, dpi=600, bbox_inches="tight")
            plt.close()

        def plot_residual_hist(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, title: str):
            y_true = np.asarray(y_true).reshape(-1)
            y_pred = np.asarray(y_pred).reshape(-1)
            resid = y_pred - y_true
            plt.figure(figsize=(6, 5))
            plt.hist(resid, bins=30)
            plt.xlabel("Residual (pred - true)")
            plt.ylabel("Count")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(out_path, dpi=600, bbox_inches="tight")
            plt.close()

        plot_regression_scatter(
            y_true=y_true_val,
            y_pred=y_pred_val,
            out_path=os.path.join(out_dir, "confusion_matrix.png"),
            title="Predicted vs True (Validation)",
        )
        plot_residual_hist(
            y_true=y_true_val,
            y_pred=y_pred_val,
            out_path=os.path.join(out_dir, "roc_curve.png"),
            title="Residuals (Validation)",
        )

        run_summary["validation"] = _regression_split_summary(
            y_true=y_true_val,
            y_pred=y_pred_val,
        )

        if (test_loader is not None) and (test_y_hat_all is not None) and test_y_hat_all.size(0) > 0:
            y_true_test = task.get_targets(test_graphs_list, test_labels_all).cpu().numpy().reshape(-1)
            y_pred_test = test_y_hat_all.cpu().numpy().reshape(-1)

            plot_regression_scatter(
                y_true=y_true_test,
                y_pred=y_pred_test,
                out_path=os.path.join(out_dir, "confusion_matrix_test.png"),
                title="Predicted vs True (Test)",
            )
            plot_residual_hist(
                y_true=y_true_test,
                y_pred=y_pred_test,
                out_path=os.path.join(out_dir, "roc_curve_test.png"),
                title="Residuals (Test)",
            )

            run_summary["test"] = _regression_split_summary(
                y_true=y_true_test,
                y_pred=y_pred_test,
            )
        else:
            plt.figure(figsize=(6, 5))
            plt.axis("off")
            plt.text(0.5, 0.5, "No test results available", ha="center", va="center")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "confusion_matrix_test.png"), dpi=600, bbox_inches="tight")
            plt.savefig(os.path.join(out_dir, "roc_curve_test.png"), dpi=600, bbox_inches="tight")
            plt.close()

    _write_run_summary_files(out_dir=out_dir, summary=run_summary)

    # -------------------------------------------------------------------------
    # RESTORED: Attention heatmaps (node attention vs performance)
    # -------------------------------------------------------------------------
    try:
        train_node_vocab_inv = _load_node_vocab_inv(args.data_folder)
    except Exception:
        train_node_vocab_inv = None

    test_node_vocab_inv = None
    if chosen_test_folder is not None:
        try:
            test_node_vocab_inv = _load_node_vocab_inv(chosen_test_folder)
        except Exception:
            test_node_vocab_inv = None

    save_attention_heatmap(
        model=model,
        task=task,
        loader=val_loader,
        device=device,
        max_neighbors=args.max_neighbors,
        neighbor_sampling=args.neighbor_sampling,
        num_neighbors=args.num_neighbors,
        seed_count=args.seed_count,
        seed_strategy=args.seed_strategy,
        max_sub_batches=args.max_sub_batches,
        seed_batch_size=args.seed_batch_size,
        out_png=os.path.join(out_dir, "attention_heatmap.png"),
        out_csv=os.path.join(out_dir, "attention_heatmap.csv"),
        split_name="Validation",
        node_vocab_inv=train_node_vocab_inv,
        top_k=args.attn_top_k,
        rank_by=args.attn_rank_by,
    )

    if test_loader is not None:
        save_attention_heatmap(
            model=model,
            task=task,
            loader=test_loader,
            device=device,
            max_neighbors=args.max_neighbors,
            neighbor_sampling=args.neighbor_sampling,
            num_neighbors=args.num_neighbors,
            seed_count=args.seed_count,
            seed_strategy=args.seed_strategy,
            max_sub_batches=args.max_sub_batches,
            seed_batch_size=args.seed_batch_size,
            out_png=os.path.join(out_dir, "attention_heatmap_test.png"),
            out_csv=os.path.join(out_dir, "attention_heatmap_test.csv"),
            split_name="Test",
            node_vocab_inv=test_node_vocab_inv,
            top_k=args.attn_top_k,
            rank_by=args.attn_rank_by,
        )


if __name__ == "__main__":
    main()
