"""
Configuration for Spatio-Temporal GNN TTM Pipeline.
All paths, hyperparameters, and feature dimensions are defined here.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Config:
    # ──────────────────────────────────────────────────────────────
    # Data paths  (matched to lab PC layout)
    # ──────────────────────────────────────────────────────────────
    data_root: str = "/DATA/DL_21/wttm/ego4d_data_full/v2"
    annotation_dir: str = "/DATA/DL_21/wttm/ego4d_data_full/v2/annotations"
    clips_dir: str = "/DATA/DL_21/wttm/ego4d_data_full/v2/clips"          # {clip_uid}.mp4
    # Fallback: if clips aren't pre-cut, try full videos
    videos_dir: str = "/DATA/DL_21/wttm/ego4d_data_full/v2/full_scale"    # {video_uid}.mp4

    train_annotation: str = "ttm_train_clean.json"
    val_annotation: str = "ttm_val_clean.json"

    # Where preprocessed features get saved
    feature_dir: str = "preprocessed_features"

    # ──────────────────────────────────────────────────────────────
    # Preprocessing
    # ──────────────────────────────────────────────────────────────
    face_size: int = 224          # resize face crops to this before ResNet
    audio_sr: int = 16000         # audio sample rate
    bbox_format: str = "xywh"     # "xywh" = top-left x, y, width, height
    bbox_pad_ratio: float = 0.15  # expand face bbox by this ratio for context

    # ──────────────────────────────────────────────────────────────
    # Graph construction
    # ──────────────────────────────────────────────────────────────
    spatial_threshold: float = 400.0   # pixel dist for spatial edges
    temporal_stride: int = 3           # connect frame t to t ± stride
    temporal_skip: int = 6             # also connect t to t ± skip (long range)
    frame_sample_rate: int = 3         # 10 FPS (keeps all temporal context without hitting the 24GB VRAM ceiling)
    min_frames_per_clip: int = 5       # skip clips with fewer frames

    # Edge type indices
    EDGE_SPATIAL: int = 0
    EDGE_TEMPORAL: int = 1
    EDGE_TEMPORAL_SKIP: int = 2
    EDGE_HEADPOSE: int = 3
    EDGE_EGO: int = 4
    EDGE_SELF: int = 5             # dedicated self-loop edge type
    NUM_EDGE_TYPES: int = 6

    headpose_feat_dim: int = 3    # pitch, yaw, roll

    # ──────────────────────────────────────────────────────────────
    # Model architecture
    face_feat_dim: int = 1024    # dinov2_vitl14 backbone output
    audio_feat_dim: int = 768     # HuBERT base output
    bbox_feat_dim: int = 6        # normalized bbox features
    node_input_dim: int = 0       # auto-calculated
    face_model: str = "dinov2_vitl14"  # dinov2_vitl14 or resnet50 or vit
    speech_context_sigma: float = 2.0  # Gaussian smoothing sigma for context-aware speech
    audio_visual_cross_attn: bool = True  # cross-modal attention in fusion
    hidden_dim: int = 256
    num_gat_layers: int = 4
    num_heads: int = 4
    gat_dropout: float = 0.4
    classifier_dropout: float = 0.5
    use_edge_type: bool = True    # encode edge types in GAT

    # ──────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────
    batch_size: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-3   # reduced from 4e-3 — was over-regularizing minority class
    num_epochs: int = 80         # increased from 50 — model needs more time with imbalanced data
    patience: int = 15           # increased from 10 — avoids early exit during slow minority class learning
    warmup_epochs: int = 5       # increased from 3 — smoother ramp into cosine schedule
    grad_accum_steps: int = 4    # effective batch = batch_size × grad_accum_steps (= 8 graphs)
    label_smooth_eps: float = 0.05  # smooth positive labels: 1 → 0.95 to prevent overconfidence

    # Class imbalance handling
    focal_alpha: float = 0.75     # up-weight positive (TTM=1) class — 0.25 was down-weighting minority
    focal_gamma: float = 2.0      # focusing parameter
    pos_weight: float = 10.0  # actual ratio: 4778407 neg / 272901 pos ≈ 17.5 (switcing to 10 , over-penalizing FP)
    use_focal_loss: bool = True   # Focal loss handles 17:1 imbalance better
    oversample_positive: bool = True
    oversample_ratio: float = 5.0  # repeat positive clips this many times

    grad_clip: float = 1.0
    use_amp: bool = True           # automatic mixed precision

    # ──────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────
    eval_aggregate: str = "max"    # "max" better for TTM: person is positive if ANY frame is positive
    tta: bool = False              # test-time augmentation

    # ──────────────────────────────────────────────────────────────
    # Misc
    # ──────────────────────────────────────────────────────────────
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    experiment_name: str = "stgnn_ttm_v1"

    # ──────────────────────────────────────────────────────────────
    # Feature extraction mode
    # ──────────────────────────────────────────────────────────────
    # "full"    = ResNet-50 visual + MFCC audio  (requires video files)
    # "lite"    = bbox metadata only             (no video files needed)
    feature_mode: str = "full"

    def __post_init__(self):
        """Auto-calculate derived fields and create directories."""
        if self.feature_mode == "full":
            self.node_input_dim = (self.face_feat_dim + self.audio_feat_dim +
                                   self.bbox_feat_dim + self.headpose_feat_dim)
        else:
            self.node_input_dim = self.bbox_feat_dim + self.headpose_feat_dim

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.feature_dir, exist_ok=True)

    @property
    def train_json_path(self) -> str:
        return os.path.join(self.annotation_dir, self.train_annotation)

    @property
    def val_json_path(self) -> str:
        return os.path.join(self.annotation_dir, self.val_annotation)


