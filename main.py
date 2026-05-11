"""
Main entry point for the Spatio-Temporal GNN TTM pipeline.

Usage:
  # Step 1: Preprocess (extract features from videos)
  python main.py preprocess --split both --mode full

  # Step 1 (alt): If no video access, use lite mode (bbox features only)
  python main.py preprocess --split both --mode lite

  # Step 2: Train the model
  python main.py train

  # Step 2 (alt): Train directly from JSON (no preprocessing needed, lite mode)
  python main.py train --direct

  # Step 3: Evaluate
  python main.py evaluate

  # Step 4: Visualize
  python main.py visualize --history ./checkpoints/training_history.json

  # Step 5: EDA
  python main.py eda

  # Full pipeline (lite mode, for testing)
  python main.py full --mode lite
"""

import argparse
import os
import sys
import time
from datetime import datetime

from config import Config


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


def cmd_preprocess(args, cfg):
    """Run preprocessing."""
    from preprocess import preprocess_split

    start = time.time()
    print(f"\n⏱  Preprocessing started at {datetime.now().strftime('%H:%M:%S')}")

    cfg.feature_mode = args.mode
    cfg.device = args.device
    if args.data_root:
        cfg.data_root = args.data_root
        cfg.annotation_dir = os.path.join(args.data_root, "annotations")
        cfg.clips_dir = os.path.join(args.data_root, "clips")
        cfg.videos_dir = os.path.join(args.data_root, "full_scale")
        cfg.feature_dir = os.path.join(args.data_root, "preprocessed_features")
    cfg.__post_init__()

    if args.split in ("train", "both"):
        preprocess_split("train", cfg)
    if args.split in ("val", "both"):
        preprocess_split("val", cfg)

    elapsed = time.time() - start
    print(f"\n✓ Preprocessing complete! Total time: {_format_time(elapsed)}")
    return elapsed


def cmd_train(args, cfg):
    """Run training."""
    from train import train

    start = time.time()
    print(f"\n⏱  Training started at {datetime.now().strftime('%H:%M:%S')}")

    cfg.feature_mode = args.mode
    cfg.device = args.device
    cfg.batch_size = args.batch_size
    cfg.num_epochs = args.epochs
    cfg.lr = args.lr
    cfg.experiment_name = args.name
    if args.data_root:
        cfg.data_root = args.data_root
        cfg.annotation_dir = os.path.join(args.data_root, "annotations")
        cfg.feature_dir = os.path.join(args.data_root, "preprocessed_features")
    cfg.__post_init__()

    print("\n" + "=" * 60)
    print(f"  EXPERIMENT: {cfg.experiment_name}")
    print(f"  Feature mode: {cfg.feature_mode}")
    print(f"  Direct dataset: {args.direct}")
    print("=" * 60)

    model, history = train(cfg, use_direct_dataset=args.direct)

    # Auto-generate training plots
    try:
        from visualize import plot_training_curves
        history_path = os.path.join(cfg.checkpoint_dir, "training_history.json")
        if os.path.isfile(history_path):
            plot_training_curves(history_path)
    except Exception as e:
        print(f"  (Could not generate plots: {e})")

    elapsed = time.time() - start
    print(f"\n✓ Training complete! Total time: {_format_time(elapsed)}")
    return elapsed


def cmd_evaluate(args, cfg):
    """Run evaluation."""
    from train import evaluate

    start = time.time()
    print(f"\n⏱  Evaluation started at {datetime.now().strftime('%H:%M:%S')}")

    cfg.feature_mode = args.mode
    cfg.device = args.device
    if args.data_root:
        cfg.data_root = args.data_root
        cfg.annotation_dir = os.path.join(args.data_root, "annotations")
        cfg.feature_dir = os.path.join(args.data_root, "preprocessed_features")
    cfg.__post_init__()

    checkpoint_path = args.checkpoint or os.path.join(cfg.checkpoint_dir, "best_model.pt")
    results = evaluate(cfg, checkpoint_path, use_direct_dataset=args.direct)

    elapsed = time.time() - start
    print(f"\n✓ Evaluation complete! Total time: {_format_time(elapsed)}")
    return elapsed


def cmd_visualize(args, cfg):
    """Generate visualizations."""
    from visualize import plot_training_curves, plot_dataset_statistics

    if args.history and os.path.isfile(args.history):
        plot_training_curves(args.history, args.save_dir)

    if args.data and os.path.isfile(args.data):
        plot_dataset_statistics(args.data, args.save_dir)

    print("\n✓ Visualization complete!")


def cmd_eda(args, cfg):
    """Run exploratory data analysis on the dataset."""
    from visualize import plot_dataset_statistics

    if args.data_root:
        cfg.annotation_dir = os.path.join(args.data_root, "annotations")

    for split_file in [cfg.train_annotation, cfg.val_annotation]:
        json_path = os.path.join(cfg.annotation_dir, split_file)
        if os.path.isfile(json_path):
            print(f"\nAnalyzing {split_file} ...")
            plot_dataset_statistics(json_path, save_dir=args.save_dir)
        else:
            print(f"  Not found: {json_path}")

    print("\n✓ EDA complete!")


def cmd_full(args, cfg):
    """Run the full pipeline end-to-end."""
    pipeline_start = time.time()
    print("\n" + "#" * 60)
    print(f"  FULL PIPELINE — Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 60)

    step_times = {}

    # Step 1: Preprocess
    print("\n>>> Step 1/3: Preprocessing ...")
    args.split = "both"
    step_times["Preprocess"] = cmd_preprocess(args, cfg)

    # Step 2: Train
    print("\n>>> Step 2/3: Training ...")
    args.direct = (args.mode == "lite")
    step_times["Train"] = cmd_train(args, cfg)

    # Step 3: Evaluate
    print("\n>>> Step 3/3: Evaluation ...")
    args.checkpoint = None
    step_times["Evaluate"] = cmd_evaluate(args, cfg)

    total_time = time.time() - pipeline_start

    print("\n" + "#" * 60)
    print(f"  FULL PIPELINE COMPLETE!")
    print(f"  Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 60)
    print(f"\n  Time Breakdown:")
    for step, t in step_times.items():
        pct = (t / total_time * 100) if total_time > 0 else 0
        print(f"    {step:15s}: {_format_time(t):>10s}  ({pct:.1f}%)")
    print(f"    {'─'*40}")
    print(f"    {'TOTAL':15s}: {_format_time(total_time):>10s}")
    print()


# ─────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="Spatio-Temporal GNN for Ego4D Talking-To-Me",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (no video files needed):
  python main.py train --direct --mode lite --epochs 20

  # Full pipeline:
  python main.py preprocess --split both --mode full
  python main.py train --mode full --epochs 100

  # Evaluate:
  python main.py evaluate --checkpoint ./checkpoints/best_model.pt
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data_root", type=str, default=None,
                       help="Override data root directory")
    common.add_argument("--device", type=str, default="cuda")
    common.add_argument("--mode", type=str, default="full",
                       choices=["full", "lite"])
    common.add_argument("--save_dir", type=str, default="./plots")

    # Preprocess
    p_pre = subparsers.add_parser("preprocess", parents=[common],
                                   help="Extract features from videos")
    p_pre.add_argument("--split", type=str, default="both",
                       choices=["train", "val", "both"])

    # Train
    p_train = subparsers.add_parser("train", parents=[common],
                                     help="Train the ST-GAT model")
    p_train.add_argument("--direct", action="store_true",
                        help="Use direct JSON dataset (no preprocessing)")
    p_train.add_argument("--batch_size", type=int, default=32)
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--lr", type=float, default=5e-4)
    p_train.add_argument("--name", type=str, default="stgnn_ttm_v1",
                        help="Experiment name")

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", parents=[common],
                                    help="Evaluate trained model")
    p_eval.add_argument("--checkpoint", type=str, default=None)
    p_eval.add_argument("--direct", action="store_true")

    # Visualize
    p_vis = subparsers.add_parser("visualize", parents=[common],
                                   help="Generate plots")
    p_vis.add_argument("--history", type=str,
                       default="./checkpoints/training_history.json")
    p_vis.add_argument("--data", type=str, default=None,
                       help="Annotation JSON for EDA plots")

    # EDA
    p_eda = subparsers.add_parser("eda", parents=[common],
                                   help="Exploratory data analysis")

    # Full pipeline
    p_full = subparsers.add_parser("full", parents=[common],
                                    help="Run full pipeline end-to-end")
    p_full.add_argument("--batch_size", type=int, default=32)
    p_full.add_argument("--epochs", type=int, default=100)
    p_full.add_argument("--lr", type=float, default=5e-4)
    p_full.add_argument("--name", type=str, default="stgnn_ttm_v1")
    p_full.add_argument("--direct", action="store_true")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    cfg = Config()

    commands = {
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "visualize": cmd_visualize,
        "eda": cmd_eda,
        "full": cmd_full,
    }

    commands[args.command](args, cfg)


if __name__ == "__main__":
    main()
