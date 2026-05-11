"""
Training loop for Spatio-Temporal GAT TTM model.

Features:
  - Focal loss with pos_weight for extreme class imbalance
  - Cosine annealing LR with warmup
  - Gradient clipping
  - Automatic mixed precision (AMP)
  - Early stopping on validation mAP
  - Best model checkpointing
  - TensorBoard logging
"""

import os
import time
import json
import random
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

from config import Config
from dataset import TTMGraphDataset, TTMGraphDatasetDirect
from ttm_model import SpatioTemporalGAT, FocalLoss, FocalLossWithPosWeight, model_summary, count_parameters


# ─────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────
# Compute class weights from dataset
# ─────────────────────────────────────────────────────────────────

def _format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m {s}s"


def compute_class_weights(loader: DataLoader, device: str = "cuda") -> float:
    """
    Scan the training data to compute pos_weight for BCEWithLogitsLoss.
    pos_weight = num_negative / num_positive
    """
    scan_start = time.time()
    print("Computing class weights from training data ...")
    n_pos = 0
    n_neg = 0
    for data in tqdm(loader, desc="Scanning labels"):
        labels = data.y
        n_pos += (labels == 1).sum().item()
        n_neg += (labels == 0).sum().item()

    total = n_pos + n_neg
    scan_time = time.time() - scan_start
    if n_pos == 0:
        print("  WARNING: No positive samples found! Using pos_weight=1.0")
        return 1.0

    pos_weight = n_neg / n_pos
    print(f"  Positive: {n_pos} ({n_pos/total*100:.2f}%)")
    print(f"  Negative: {n_neg} ({n_neg/total*100:.2f}%)")
    print(f"  pos_weight: {pos_weight:.2f}")
    print(f"  Scan time: {_format_time(scan_time)}")
    return pos_weight


# ─────────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler, cfg, epoch):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_nodes = 0
    
    # Trackers for node-level and person-level metrics
    all_logits = []
    all_labels = []
    all_person_ids = []
    all_clip_uids = []

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    grad_accum = getattr(cfg, "grad_accum_steps", 1)
    smooth_eps = getattr(cfg, "label_smooth_eps", 0.0)

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)
    optimizer.zero_grad(set_to_none=True)   # zero once before loop (accumulation pattern)

    for i, batch in enumerate(pbar):
        batch = batch.to(device)
        valid_mask = (batch.person_ids != -1) & (batch.person_ids != -999)

        # Skip batches with no valid nodes (all ego/padding) — empty criterion → NaN
        if valid_mask.sum() == 0:
            continue

        # Label smoothing: 1 → (1 - eps) on positives only; keeps negatives at 0
        y = batch.y[valid_mask]
        if smooth_eps > 0:
            y = y.clone()
            y[y == 1] = 1.0 - smooth_eps

        is_last_batch = (i + 1 == len(loader))
        should_step   = ((i + 1) % grad_accum == 0) or is_last_batch

        if cfg.use_amp and device.type == "cuda":
            with torch.amp.autocast('cuda'):
                logits = model(batch)
                loss = criterion(logits[valid_mask], y) / grad_accum
            scaler.scale(loss).backward()
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            logits = model(batch)
            loss = criterion(logits[valid_mask], y) / grad_accum
            loss.backward()
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # Undo the accumulation normalization for readable loss logging
        raw_loss = loss.item() * grad_accum
        batch_nodes = valid_mask.sum().item()

        # Guard: skip NaN/inf losses — don't let one bad batch poison the epoch average
        if raw_loss == raw_loss and not (raw_loss == float('inf')):   # NaN != NaN
            total_loss += raw_loss * batch_nodes
            total_nodes += batch_nodes

        # Collect for metrics — nan_to_num ensures NaN logits don't corrupt AUC/mAP
        safe_logits = torch.nan_to_num(logits[valid_mask], nan=0.0, posinf=10.0, neginf=-10.0)
        all_logits.append(safe_logits.detach().cpu())
        all_labels.append(batch.y[valid_mask].detach().cpu())
        all_person_ids.append(batch.person_ids[valid_mask].detach().cpu())

        valid_batch_indices = batch.batch[valid_mask].tolist()
        node_clip_uids = [batch.clip_uid[b_idx] for b_idx in valid_batch_indices]
        all_clip_uids.extend(node_clip_uids)

        pbar.set_postfix(loss=f"{raw_loss:.4f}")


    avg_loss = total_loss / max(total_nodes, 1)

    # Compute training metrics
    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
    all_person_ids = torch.cat(all_person_ids).numpy()

    # Node-level metrics
    node_metrics = compute_metrics(all_probs, all_labels)
    node_metrics["loss"] = avg_loss

    # Person-level metrics
    person_metrics = compute_person_metrics(
        all_probs, all_labels, all_person_ids, all_clip_uids, aggregate=cfg.eval_aggregate
    )

    return node_metrics, person_metrics


# ─────────────────────────────────────────────────────────────────
# Validation epoch
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate_epoch(model, loader, criterion, cfg, epoch):
    """Run one validation epoch."""
    model.eval()
    total_loss = 0.0
    total_nodes = 0
    all_logits = []
    all_labels = []
    all_person_ids = []
    all_clip_uids = []
    all_batch_ids = []

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Val]  ", leave=False)
    for batch in pbar:
        batch = batch.to(device)

        if cfg.use_amp and device.type == "cuda":
            with torch.amp.autocast('cuda'):
                logits = model(batch)
                valid_mask = (batch.person_ids != -1) & (batch.person_ids != -999)
                loss = criterion(logits[valid_mask], batch.y[valid_mask])
        else:
            logits = model(batch)
            valid_mask = (batch.person_ids != -1) & (batch.person_ids != -999)
            loss = criterion(logits[valid_mask], batch.y[valid_mask])

        batch_nodes = valid_mask.sum().item()
        total_loss += loss.item() * batch_nodes
        total_nodes += batch_nodes

        all_logits.append(logits[valid_mask].cpu())
        all_labels.append(batch.y[valid_mask].cpu())
        all_person_ids.append(batch.person_ids[valid_mask].cpu())
        
        valid_batch_indices = batch.batch[valid_mask].tolist()
        node_clip_uids = [batch.clip_uid[b_idx] for b_idx in valid_batch_indices]
        all_clip_uids.extend(node_clip_uids)

    avg_loss = total_loss / max(total_nodes, 1)

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = 1 / (1 + np.exp(-all_logits))

    # Node-level metrics
    node_metrics = compute_metrics(all_probs, all_labels)
    node_metrics["loss"] = avg_loss

    # Per-person aggregated metrics
    all_person_ids = torch.cat(all_person_ids).numpy()
    person_metrics = compute_person_metrics(all_probs, all_labels, all_person_ids,
                                            all_clip_uids, aggregate=cfg.eval_aggregate)

    return node_metrics, person_metrics


# ─────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────

def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute classification metrics."""
    metrics = {}

    # Binary predictions at threshold
    preds = (probs >= threshold).astype(int)

    try:
        metrics["auc_roc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auc_roc"] = 0.0

    try:
        metrics["mAP"] = average_precision_score(labels, probs)
    except ValueError:
        metrics["mAP"] = 0.0

    metrics["f1"] = f1_score(labels, preds, zero_division=0)
    metrics["precision"] = precision_score(labels, preds, zero_division=0)
    metrics["recall"] = recall_score(labels, preds, zero_division=0)
    metrics["accuracy"] = (preds == labels).mean()

    # Also compute with optimal threshold
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        p = (probs >= thresh).astype(int)
        f = f1_score(labels, p, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = thresh
    metrics["best_f1"] = best_f1
    metrics["best_threshold"] = best_thresh

    return metrics


def compute_person_metrics(probs: np.ndarray, labels: np.ndarray,
                           person_ids: np.ndarray, clip_uids: list, aggregate: str = "mean", threshold: float = 0.5) -> dict:
    """
    Compute per-person aggregated metrics.
    Groups nodes by (clip_uid, person_id) and aggregates predictions.
    """
    from collections import defaultdict

    person_probs = defaultdict(list)
    person_labels = defaultdict(list)

    for i in range(len(probs)):
        pid = f"{clip_uids[i]}_{person_ids[i]}"
        person_probs[pid].append(probs[i])
        person_labels[pid].append(labels[i])

    agg_probs = []
    agg_labels = []
    for pid in person_probs:
        if aggregate == "mean":
            agg_probs.append(np.mean(person_probs[pid]))
        else:
            agg_probs.append(np.max(person_probs[pid]))
        agg_labels.append(int(np.max(person_labels[pid])))

    agg_probs = np.array(agg_probs)
    agg_labels = np.array(agg_labels)

    return compute_metrics(agg_probs, agg_labels, threshold=threshold)


# ─────────────────────────────────────────────────────────────────
# Checkpoint management
# ─────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, cfg, filename="best_model.pt"):
    """Save model checkpoint."""
    path = os.path.join(cfg.checkpoint_dir, filename)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": cfg.__dict__,
    }, path)
    print(f"  ✓ Checkpoint saved: {path}")


def load_checkpoint(model, path, device="cuda"):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  ✓ Model loaded from {path} (epoch {checkpoint['epoch']})")
    return checkpoint


def plot_training_history(history_data, save_dir):
    """Plot training curves from history data."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping plots.")
        return

    if not history_data:
        return

    epochs = [r["epoch"] for r in history_data]
    
    plt.figure(figsize=(15, 10))
    
    # 1. Loss
    plt.subplot(2, 2, 1)
    if "train_node" in history_data[0]:
        plt.plot(epochs, [r["train_node"]["loss"] for r in history_data], label="Train Node Loss")
        plt.plot(epochs, [r["val_node"]["loss"] for r in history_data], label="Val Node Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    
    # 2. mAP (Person)
    plt.subplot(2, 2, 2)
    if "train_person" in history_data[0]:
        plt.plot(epochs, [r["train_person"]["mAP"] for r in history_data], label="Train Person mAP")
        plt.plot(epochs, [r["val_person"]["mAP"] for r in history_data], label="Val Person mAP")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Person-Level mAP")
    plt.legend()
    plt.grid(True)
    
    # 3. AUC (Person)
    plt.subplot(2, 2, 3)
    if "train_person" in history_data[0]:
        plt.plot(epochs, [r["train_person"]["auc_roc"] for r in history_data], label="Train Person AUC")
        plt.plot(epochs, [r["val_person"]["auc_roc"] for r in history_data], label="Val Person AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC ROC")
    plt.title("Person-Level AUC")
    plt.legend()
    plt.grid(True)
    
    # 4. F1 Score (Person)
    plt.subplot(2, 2, 4)
    if "train_person" in history_data[0]:
        plt.plot(epochs, [r["train_person"]["best_f1"] for r in history_data], label="Train Person Best F1")
        plt.plot(epochs, [r["val_person"]["best_f1"] for r in history_data], label="Val Person Best F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Person-Level F1")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"  Training plots saved to: {plot_path}")


# ─────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────

def train(cfg: Config, use_direct_dataset: bool = False):
    """
    Full training pipeline.

    Args:
        cfg: Configuration object
        use_direct_dataset: If True, use TTMGraphDatasetDirect (no preprocessing needed)
    """
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Feature mode: {cfg.feature_mode}")

    # ── Setup datasets ──
    print("\n" + "=" * 60)
    print("Loading datasets ...")
    print("=" * 60)

    ds_start = time.time()
    if use_direct_dataset:
        train_dataset = TTMGraphDatasetDirect(
            cfg.train_json_path, cfg,
            oversample=cfg.oversample_positive
        )
        val_dataset = TTMGraphDatasetDirect(
            cfg.val_json_path, cfg,
            oversample=False
        )
    else:
        train_dataset = TTMGraphDataset(
            "train", cfg,
            oversample=cfg.oversample_positive
        )
        val_dataset = TTMGraphDataset("val", cfg, oversample=False)
    print(f"  Dataset loading took: {_format_time(time.time() - ds_start)}")
    print(f"  Train graphs: {len(train_dataset)} | Val graphs: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Compute class weights ──
    if cfg.pos_weight <= 0:
        cfg.pos_weight = compute_class_weights(train_loader, device)
    print(f"Using pos_weight: {cfg.pos_weight:.2f}")

    # ── Initialize model ──
    print("\n" + "=" * 60)
    print("Initializing model ...")
    print("=" * 60)

    model = SpatioTemporalGAT(cfg).to(device)
    model_summary(model, cfg)

    # ── Loss function ──
    if cfg.use_focal_loss:
        criterion = FocalLossWithPosWeight(
            alpha=cfg.focal_alpha,
            gamma=cfg.focal_gamma,
            pos_weight=min(cfg.pos_weight, 50.0),  # cap to prevent instability
        )
        print(f"Using Focal Loss (α={cfg.focal_alpha}, γ={cfg.focal_gamma}, "
              f"pos_weight={min(cfg.pos_weight, 50.0):.1f})")
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([min(cfg.pos_weight, 50.0)]).to(device)
        )
        print(f"Using BCEWithLogitsLoss (pos_weight={min(cfg.pos_weight, 50.0):.1f})")

    # ── Optimizer ──
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # ── LR Scheduler: Linear warmup + Cosine annealing ──
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=cfg.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.num_epochs - cfg.warmup_epochs, T_mult=1,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.warmup_epochs],
    )

    # ── AMP Scaler ──
    scaler = torch.amp.GradScaler('cuda') if cfg.use_amp and device.type == "cuda" else None

    # ── TensorBoard ──
    writer = None
    if HAS_TB:
        log_path = os.path.join(cfg.log_dir, cfg.experiment_name,
                                datetime.now().strftime("%Y%m%d_%H%M%S"))
        writer = SummaryWriter(log_path)
        print(f"TensorBoard logs: {log_path}")

    # ── Training loop ──
    print("\n" + "=" * 60)
    print("Starting training ...")
    print(f"  Max epochs: {cfg.num_epochs} | Early stopping patience: {cfg.patience}")
    print("=" * 60)

    best_map = 0.0
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    epoch_times = []
    training_start = time.time()

    for epoch in range(1, cfg.num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_node_metrics, train_person_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, cfg, epoch
        )

        # Validate
        val_node_metrics, val_person_metrics = validate_epoch(
            model, val_loader, criterion, cfg, epoch
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # ── Timing calculations ──
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        total_elapsed = time.time() - training_start
        remaining_epochs = cfg.num_epochs - epoch
        eta_seconds = remaining_epochs * avg_epoch_time
        est_finish = datetime.now() + timedelta(seconds=eta_seconds)

        # ── Logging ──
        record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_node": train_node_metrics,
            "train_person": train_person_metrics,
            "val_node": val_node_metrics,
            "val_person": val_person_metrics,
            "time": epoch_time,
        }
        history.append(record)

        improved = "" 
        val_map = val_person_metrics["mAP"]
        val_auc = val_person_metrics["auc_roc"]
        if val_map > best_map:
            improved = " ★ NEW BEST"

        print(f"\n{'─'*60}")
        print(f"Epoch {epoch:03d}/{cfg.num_epochs} | "
              f"{epoch_time:.1f}s (avg {avg_epoch_time:.1f}s) | "
              f"Elapsed: {_format_time(total_elapsed)} | "
              f"ETA: {_format_time(eta_seconds)} (~{est_finish.strftime('%H:%M')}) | "
              f"LR: {current_lr:.2e}{improved}")
        
        print(f"  Train(N)| Loss: {train_node_metrics['loss']:.4f} | "
              f"AUC: {train_node_metrics['auc_roc']:.4f} | "
              f"mAP: {train_node_metrics['mAP']:.4f} | "
              f"F1: {train_node_metrics['best_f1']:.4f}@{train_node_metrics['best_threshold']:.2f}")
              
        print(f"  Train(P)| "
              f"AUC: {train_person_metrics['auc_roc']:.4f} | "
              f"mAP: {train_person_metrics['mAP']:.4f} | "
              f"F1: {train_person_metrics['best_f1']:.4f}")

        print(f"  Val(N)  | Loss: {val_node_metrics['loss']:.4f} | "
              f"AUC: {val_node_metrics['auc_roc']:.4f} | "
              f"mAP: {val_node_metrics['mAP']:.4f} | "
              f"F1: {val_node_metrics['best_f1']:.4f}@{val_node_metrics['best_threshold']:.2f}")
              
        print(f"  Val(P)  | "
              f"AUC: {val_person_metrics['auc_roc']:.4f} | "
              f"mAP: {val_person_metrics['mAP']:.4f} | "
              f"F1: {val_person_metrics['best_f1']:.4f} | "
              f"Patience: {patience_counter}/{cfg.patience}")

        if writer:
            writer.add_scalar("Loss/train", train_node_metrics["loss"], epoch)
            writer.add_scalar("Loss/val", val_node_metrics["loss"], epoch)
            writer.add_scalar("AUC/train_node", train_node_metrics["auc_roc"], epoch)
            writer.add_scalar("AUC/train_person", train_person_metrics["auc_roc"], epoch)
            writer.add_scalar("AUC/val_node", val_node_metrics["auc_roc"], epoch)
            writer.add_scalar("AUC/val_person", val_person_metrics["auc_roc"], epoch)
            writer.add_scalar("mAP/train_node", train_node_metrics["mAP"], epoch)
            writer.add_scalar("mAP/train_person", train_person_metrics["mAP"], epoch)
            writer.add_scalar("mAP/val_node", val_node_metrics["mAP"], epoch)
            writer.add_scalar("mAP/val_person", val_person_metrics["mAP"], epoch)
            writer.add_scalar("LR", current_lr, epoch)

        # ── Best model checkpoint ──
        if val_map > best_map:
            best_map = val_map
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch,
                           val_node_metrics, cfg, "best_model.pt")
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch,
                           val_node_metrics, cfg, f"checkpoint_epoch{epoch:03d}.pt")

        # ── Early stopping ──
        if patience_counter >= cfg.patience:
            print(f"\n⚠ Early stopping at epoch {epoch} (no improvement for "
                  f"{cfg.patience} epochs)")
            break

    # ── Final summary ──
    total_training_time = time.time() - training_start
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Total time:     {_format_time(total_training_time)}")
    print(f"  Epochs run:     {len(epoch_times)}/{cfg.num_epochs}")
    print(f"  Avg per epoch:  {_format_time(avg_epoch_time)}")
    print(f"  Best epoch:     {best_epoch}")
    print(f"  Best val mAP:   {best_map:.4f}")
    print(f"  Best val AUC:   {best_auc:.4f}")

    # Save training history
    history_path = os.path.join(cfg.checkpoint_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"  History saved:  {history_path}")

    # Plot history
    plot_training_history(history, cfg.checkpoint_dir)

    if writer:
        writer.close()

    return model, history


# ─────────────────────────────────────────────────────────────────
# Test evaluation
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(cfg: Config, checkpoint_path: str = None,
             use_direct_dataset: bool = False):
    """
    Full evaluation on validation set with detailed metrics.
    """
    eval_start = time.time()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = SpatioTemporalGAT(cfg).to(device)
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    # Load dataset
    ds_start = time.time()
    if use_direct_dataset:
        dataset = TTMGraphDatasetDirect(cfg.val_json_path, cfg, oversample=False)
    else:
        dataset = TTMGraphDataset("val", cfg, oversample=False)
    print(f"  Dataset loaded in {_format_time(time.time() - ds_start)}")

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers)
    print(f"  {len(loader)} batches to evaluate")

    # Collect predictions
    all_logits = []
    all_labels = []
    all_person_ids = []
    all_clip_uids = []

    infer_start = time.time()
    for batch in tqdm(loader, desc="Evaluating", unit="batch"):
        batch = batch.to(device)
        logits = model(batch)

        # BUG FIX: filter out virtual ego nodes (person_ids == -999) and
        # padding nodes (person_ids == -1), matching train/val behavior.
        # Without this, ego nodes (always label=0) inflate TN and corrupt metrics.
        valid_mask = (batch.person_ids != -1) & (batch.person_ids != -999)

        all_logits.append(logits[valid_mask].cpu())
        all_labels.append(batch.y[valid_mask].cpu())
        all_person_ids.append(batch.person_ids[valid_mask].cpu())

        valid_batch_indices = batch.batch[valid_mask].tolist()
        node_clip_uids = [batch.clip_uid[b_idx] for b_idx in valid_batch_indices]
        all_clip_uids.extend(node_clip_uids)

    infer_time = time.time() - infer_start
    print(f"  Inference took: {_format_time(infer_time)} "
          f"({len(loader)/max(infer_time,1e-3):.1f} batches/sec)")

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = 1 / (1 + np.exp(-all_logits))
    all_person_ids = torch.cat(all_person_ids).numpy()

    # Node-level metrics
    print("\n" + "=" * 60)
    print("Node-Level Metrics")
    print("=" * 60)
    node_metrics = compute_metrics(all_probs, all_labels, threshold=0.40)
    for k, v in node_metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    # Person-level metrics
    print("\n" + "=" * 60)
    print("Person-Level Metrics (aggregated)")
    print("=" * 60)
    person_metrics = compute_person_metrics(all_probs, all_labels, all_person_ids, all_clip_uids, 
                                            aggregate=cfg.eval_aggregate, threshold=0.20)
    for k, v in person_metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    # Confusion matrix at best threshold
    best_thresh = node_metrics["best_threshold"]
    preds = (all_probs >= best_thresh).astype(int)
    cm = confusion_matrix(all_labels, preds)
    print(f"\nConfusion Matrix (threshold={best_thresh:.2f}):")
    print(f"  TN={cm[0, 0]:6d}  FP={cm[0, 1]:6d}")
    print(f"  FN={cm[1, 0]:6d}  TP={cm[1, 1]:6d}")

    total_eval_time = time.time() - eval_start
    print(f"\n  Total evaluation time: {_format_time(total_eval_time)}")

    # Save evaluation results
    results = {
        "node_metrics": node_metrics,
        "person_metrics": person_metrics,
        "confusion_matrix": cm.tolist(),
        "best_threshold": best_thresh,
        "eval_time_seconds": total_eval_time,
    }
    results_path = os.path.join(cfg.checkpoint_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {results_path}")

    # Also try to plot training history if it exists
    history_path = os.path.join(cfg.checkpoint_dir, "training_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history_data = json.load(f)
            plot_training_history(history_data, cfg.checkpoint_dir)
        except Exception as e:
            print(f"  Could not plot training history: {e}")

    return results
