"""
Visualization utilities for Spatio-Temporal GAT TTM.

Features:
  - Graph attention weight visualization
  - Training curve plots
  - Per-clip graph overlay on video frames
  - Confusion matrix heatmap
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve


# ─────────────────────────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────────────────────────

def plot_training_curves(history_path: str, save_dir: str = "./plots"):
    """Plot training and validation curves from history JSON.

    Generates two separate plots:
      - training_curves_node.png  (node-level: train_node vs val_node)
      - training_curves_person.png (person-level: train_person vs val_person)
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]

    # ── Node-level plot ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Progress — Node Level", fontsize=16, fontweight="bold")

    # Loss (only node has loss)
    ax = axes[0, 0]
    ax.plot(epochs, [h["train_node"]["loss"] for h in history], label="Train", linewidth=2)
    ax.plot(epochs, [h["val_node"]["loss"] for h in history], label="Val", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # AUC-ROC
    ax = axes[0, 1]
    ax.plot(epochs, [h["train_node"]["auc_roc"] for h in history], label="Train", linewidth=2)
    ax.plot(epochs, [h["val_node"]["auc_roc"] for h in history], label="Val", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("AUC-ROC"); ax.set_title("AUC-ROC")
    ax.legend(); ax.grid(True, alpha=0.3)

    # mAP
    ax = axes[0, 2]
    ax.plot(epochs, [h["train_node"]["mAP"] for h in history], label="Train", linewidth=2)
    ax.plot(epochs, [h["val_node"]["mAP"] for h in history], label="Val", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("mAP"); ax.set_title("Mean Average Precision")
    ax.legend(); ax.grid(True, alpha=0.3)

    # F1
    ax = axes[1, 0]
    ax.plot(epochs, [h["train_node"]["f1"] for h in history], label="Train", linewidth=2)
    ax.plot(epochs, [h["val_node"]["f1"] for h in history], label="Val", linewidth=2)
    ax.plot(epochs, [h["val_node"]["best_f1"] for h in history], label="Val (Best-F1)",
            linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1 Score"); ax.set_title("F1 Score")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Precision & Recall
    ax = axes[1, 1]
    ax.plot(epochs, [h["val_node"]["precision"] for h in history], label="Precision", linewidth=2)
    ax.plot(epochs, [h["val_node"]["recall"] for h in history], label="Recall", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_title("Precision / Recall")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 2]
    ax.plot(epochs, [h["lr"] for h in history], linewidth=2, color="green")
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3); ax.set_yscale("log")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves_node.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Node-level training curves saved to {save_path}")

    # ── Person-level plot ────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Progress — Person Level", fontsize=16, fontweight="bold")

    # Accuracy (person has no loss, use accuracy instead)
    ax = axes[0, 0]
    ax.plot(epochs, [h["train_person"]["accuracy"] for h in history], label="Train", linewidth=2)
    ax.plot(epochs, [h["val_person"]["accuracy"] for h in history], label="Val", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)

    # AUC-ROC
    ax = axes[0, 1]
    ax.plot(epochs, [h["train_person"]["auc_roc"] for h in history], label="Train", linewidth=2)
    ax.plot(epochs, [h["val_person"]["auc_roc"] for h in history], label="Val", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("AUC-ROC"); ax.set_title("AUC-ROC")
    ax.legend(); ax.grid(True, alpha=0.3)

    # mAP
    ax = axes[0, 2]
    ax.plot(epochs, [h["train_person"]["mAP"] for h in history], label="Train", linewidth=2)
    ax.plot(epochs, [h["val_person"]["mAP"] for h in history], label="Val", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("mAP"); ax.set_title("Mean Average Precision")
    ax.legend(); ax.grid(True, alpha=0.3)

    # F1
    ax = axes[1, 0]
    ax.plot(epochs, [h["train_person"]["f1"] for h in history], label="Train", linewidth=2)
    ax.plot(epochs, [h["val_person"]["f1"] for h in history], label="Val", linewidth=2)
    ax.plot(epochs, [h["val_person"]["best_f1"] for h in history], label="Val (Best-F1)",
            linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1 Score"); ax.set_title("F1 Score")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Precision & Recall
    ax = axes[1, 1]
    ax.plot(epochs, [h["val_person"]["precision"] for h in history], label="Precision", linewidth=2)
    ax.plot(epochs, [h["val_person"]["recall"] for h in history], label="Recall", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_title("Precision / Recall")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Learning rate (same for both)
    ax = axes[1, 2]
    ax.plot(epochs, [h["lr"] for h in history], linewidth=2, color="green")
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3); ax.set_yscale("log")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves_person.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Person-level training curves saved to {save_path}")


# ─────────────────────────────────────────────────────────────────
# ROC and PR curves
# ─────────────────────────────────────────────────────────────────

def plot_roc_pr_curves(labels: np.ndarray, probs: np.ndarray,
                       save_dir: str = "./plots"):
    """Plot ROC and Precision-Recall curves."""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # PR curve
    precision, recall, _ = precision_recall_curve(labels, probs)
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(labels, probs)
    ax2.plot(recall, precision, linewidth=2, label=f"AP = {ap:.4f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "roc_pr_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC/PR curves saved to {save_path}")


# ─────────────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray,
                          save_dir: str = "./plots"):
    """Plot confusion matrix heatmap."""
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                xticklabels=["Not TTM (0)", "TTM (1)"],
                yticklabels=["Not TTM (0)", "TTM (1)"])
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Confusion Matrix (Counts)")

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues", ax=ax2,
                xticklabels=["Not TTM (0)", "TTM (1)"],
                yticklabels=["Not TTM (0)", "TTM (1)"])
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("Confusion Matrix (Normalized)")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


# ─────────────────────────────────────────────────────────────────
# Graph overlay visualization
# ─────────────────────────────────────────────────────────────────

def visualize_graph_on_frame(frame: np.ndarray, bboxes: list,
                             person_ids: list, predictions: list,
                             labels: list, edges: list = None,
                             save_path: str = None):
    """
    Overlay graph nodes (bboxes) and edges on a video frame.

    Args:
        frame: (H, W, 3) BGR image
        bboxes: list of [x, y, w, h] for each person
        person_ids: list of person ID strings
        predictions: list of TTM probabilities
        labels: list of ground truth labels
        edges: list of (i, j) edge pairs (optional)
        save_path: where to save the visualization
    """
    import cv2

    vis = frame.copy()
    h, w = vis.shape[:2]

    colors = {
        "tp": (0, 255, 0),    # Green: correct positive
        "tn": (200, 200, 200),  # Gray: correct negative
        "fp": (0, 165, 255),  # Orange: false positive
        "fn": (0, 0, 255),    # Red: false negative
    }

    centers = []

    for i, (bbox, pid, pred, label) in enumerate(zip(bboxes, person_ids,
                                                      predictions, labels)):
        x, y, bw, bh = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)

        # Determine color based on prediction correctness
        pred_binary = 1 if pred >= 0.5 else 0
        if label == 1 and pred_binary == 1:
            color = colors["tp"]
            status = "TP"
        elif label == 0 and pred_binary == 0:
            color = colors["tn"]
            status = "TN"
        elif label == 0 and pred_binary == 1:
            color = colors["fp"]
            status = "FP"
        else:
            color = colors["fn"]
            status = "FN"

        # Draw bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

        # Draw label
        text = f"P{pid} | {pred:.2f} ({status})"
        cv2.putText(vis, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                     0.7, color, 2)

        # Store center for edge drawing
        cx = int(x + bw / 2)
        cy = int(y + bh / 2)
        centers.append((cx, cy))

    # Draw edges
    if edges and centers:
        for (i, j) in edges:
            if i < len(centers) and j < len(centers):
                cv2.line(vis, centers[i], centers[j], (255, 200, 0), 2)

    if save_path:
        cv2.imwrite(save_path, vis)

    return vis


# ─────────────────────────────────────────────────────────────────
# Dataset statistics
# ─────────────────────────────────────────────────────────────────

def plot_dataset_statistics(json_path: str, save_dir: str = "./plots"):
    """
    Plot EDA statistics for the TTM dataset.
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ["clips", "data", "annotations"]:
            if key in data:
                data = data[key]
                break
        if isinstance(data, dict):
            all_entries = []
            for v in data.values():
                all_entries.extend(v)
            data = all_entries

    # Group by clip
    clips = defaultdict(list)
    for e in data:
        clips[e["clip_uid"]].append(e)

    # Stats
    labels = [e["ttm_label"] for e in data]
    persons_per_clip = [len(set(e["person_id"] for e in v)) for v in clips.values()]
    frames_per_clip = [len(set(e["frame"] for e in v)) for v in clips.values()]
    bbox_widths = [e["bbox"][2] for e in data]
    bbox_heights = [e["bbox"][3] for e in data]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Dataset Statistics", fontsize=16, fontweight="bold")

    # Class distribution
    ax = axes[0, 0]
    unique, counts = np.unique(labels, return_counts=True)
    colors_bar = ["#ff6b6b", "#51cf66"]
    ax.bar(["Not TTM (0)", "TTM (1)"], counts, color=colors_bar)
    for i, (u, c) in enumerate(zip(unique, counts)):
        ax.text(i, c + len(data) * 0.01, f"{c}\n({c/len(data)*100:.1f}%)",
                ha="center", fontsize=11)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")

    # Persons per clip
    ax = axes[0, 1]
    ax.hist(persons_per_clip, bins=range(1, max(persons_per_clip) + 2),
            edgecolor="black", alpha=0.7, color="#748ffc")
    ax.set_title("Persons per Clip")
    ax.set_xlabel("Number of Persons")
    ax.set_ylabel("Number of Clips")

    # Frames per clip
    ax = axes[0, 2]
    ax.hist(frames_per_clip, bins=30, edgecolor="black", alpha=0.7, color="#ff922b")
    ax.set_title("Frames per Clip")
    ax.set_xlabel("Number of Frames")
    ax.set_ylabel("Number of Clips")

    # Bbox size distribution
    ax = axes[1, 0]
    ax.hist(bbox_widths, bins=50, alpha=0.6, label="Width", color="#51cf66")
    ax.hist(bbox_heights, bins=50, alpha=0.6, label="Height", color="#748ffc")
    ax.set_title("Bbox Size Distribution")
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Count")
    ax.legend()

    # Bbox center positions
    ax = axes[1, 1]
    cx = [e["bbox"][0] + e["bbox"][2] / 2 for e in data]
    cy = [e["bbox"][1] + e["bbox"][3] / 2 for e in data]
    ax.scatter(cx, cy, s=1, alpha=0.1, c=[labels], cmap="coolwarm")
    ax.set_title("Bbox Centers (colored by label)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.invert_yaxis()

    # Positive rate per clip
    ax = axes[1, 2]
    pos_rates = []
    for v in clips.values():
        l = [e["ttm_label"] for e in v]
        pos_rates.append(sum(l) / len(l))
    ax.hist(pos_rates, bins=20, edgecolor="black", alpha=0.7, color="#e64980")
    ax.set_title("Positive Rate per Clip")
    ax.set_xlabel("Fraction of TTM=1 entries")
    ax.set_ylabel("Number of Clips")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "dataset_statistics.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Dataset statistics saved to {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, default="./checkpoints/training_history.json")
    parser.add_argument("--data", type=str, default=None, help="JSON annotation path for EDA")
    parser.add_argument("--save_dir", type=str, default="./plots")
    args = parser.parse_args()

    if args.history and os.path.isfile(args.history):
        plot_training_curves(args.history, args.save_dir)

    if args.data and os.path.isfile(args.data):
        plot_dataset_statistics(args.data, args.save_dir)
