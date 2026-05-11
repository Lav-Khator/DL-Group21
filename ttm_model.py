"""
Spatio-Temporal Graph Attention Network (ST-GAT) for TTM prediction.

Architecture:
  1. Input projection layer
  2. Multi-layer GAT with edge-type-aware attention + residual connections
  3. Node-level MLP classifier
  4. Optional per-person temporal pooling for clip-level prediction

Also includes Focal Loss for extreme class imbalance.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm, global_mean_pool
from torch_geometric.utils import softmax


# ═══════════════════════════════════════════════════════════════════
# Focal Loss
# ═══════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with extreme class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: weight for positive class (TTM=1), (1-alpha) for negative
        gamma: focusing parameter, higher = more focus on hard examples
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N,) raw logits (before sigmoid)
            targets: (N,) binary labels {0, 1}
        """
        probs = torch.sigmoid(logits)
        # p_t = p if y=1 else (1-p)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        # alpha_t = alpha if y=1 else (1-alpha)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal term
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # BCE loss (numerically stable)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        loss = focal_weight * bce
        return loss.mean()


class FocalLossWithPosWeight(nn.Module):
    """
    Stabilized Focal loss with pos_weight scaling.
    Applies pos_weight inside the BCE calculation for numerical stability.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
        # Use native PyTorch BCE for stabilized log-sum-exp trick
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction="none", 
            pos_weight=torch.tensor([pos_weight])
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Move pos_weight to correct device dynamically
        if self.bce_loss.pos_weight.device != logits.device:
            self.bce_loss.pos_weight = self.bce_loss.pos_weight.to(logits.device)

        bce = self.bce_loss(logits, targets)
        
        # Calculate p_t (probability of the target class)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # alpha_t: weight positive class by alpha, negative by (1 - alpha)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal scaling factor (with alpha weighting restored)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        loss = focal_weight * bce
        return loss.mean()


# ═══════════════════════════════════════════════════════════════════
# Edge-Type Embedding
# ═══════════════════════════════════════════════════════════════════

class EdgeTypeEmbedding(nn.Module):
    """Learnable edge type embeddings added to attention."""

    def __init__(self, num_edge_types: int = 4, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_edge_types, embed_dim)

    def forward(self, edge_type: torch.Tensor) -> torch.Tensor:
        return self.embedding(edge_type)


# ═══════════════════════════════════════════════════════════════════
# GAT Layer with Residual + LayerNorm
# ═══════════════════════════════════════════════════════════════════

class ResidualGATBlock(nn.Module):
    """
    Single GAT block with:
      - GATv2Conv (multi-head attention)
      - Residual connection (with projection if dims change)
      - Layer normalization
      - Dropout
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 4,
                 dropout: float = 0.2, concat: bool = True,
                 edge_dim: int = None):
        super().__init__()

        self.concat = concat
        actual_out = out_channels * heads if concat else out_channels

        self.gat = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=False,  # we already added self-loops in graph
        )

        self.norm = nn.LayerNorm(actual_out)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions don't match
        if in_channels != actual_out:
            self.residual_proj = nn.Linear(in_channels, actual_out)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        residual = self.residual_proj(x)
        out = self.gat(x, edge_index, edge_attr=edge_attr)
        out = self.dropout(out)
        out = self.norm(out + residual)
        out = F.elu(out)
        return out


class AudioVisualCrossAttentionFusion(nn.Module):
    """
    Enhanced Fusion with:
    1. Visual Modality Missing Awareness (VMMA)
    2. Visual Speaker Target Recognition (VSTR) with Lip Reading Confidence
    3. Headpose/Spatial Fallback
    """
    def __init__(self, cfg):
        super().__init__()
        self.face_dim = cfg.face_feat_dim
        self.audio_dim = cfg.audio_feat_dim
        self.spatial_dim = cfg.bbox_feat_dim + cfg.headpose_feat_dim
        self.hidden_dim = cfg.hidden_dim

        # ── 1. Visual Modality Missing Awareness (VMMA) Embeddings ──
        # Learnable tokens to replace zero-padded missing data
        # Small init scale prevents large activations in early training
        self.missing_face_token  = nn.Parameter(torch.randn(1, self.face_dim)  * 0.02)
        self.missing_audio_token = nn.Parameter(torch.randn(1, self.audio_dim) * 0.02)

        # Per-modality projections
        self.face_proj = nn.Sequential(
            nn.Linear(self.face_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ELU(),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(self.audio_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ELU(),
        )
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.spatial_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ELU(),
        )

        # ── 2. Visual Speaker Target Recognition (VSTR) ──
        # Evaluates Face + Audio to determine "Is this person speaking to me?" (Lip Reading)
        self.lip_reading_scorer = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim // 2),
            nn.ELU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
            nn.Sigmoid() # Outputs confidence [0, 1]
        )

        # ── 3. Fallback Mechanism ──
        # Evaluates Spatial + Headpose to determine "Are they looking at me?"
        self.fallback_scorer = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ELU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Final output projection
        self.out_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ELU(),
            nn.Dropout(cfg.gat_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.spatial_dim:
            return self.out_proj(self.spatial_proj(x))

        # Split modalities
        face_x = x[:, :self.face_dim]
        audio_x = x[:, self.face_dim : self.face_dim + self.audio_dim]
        spatial_x = x[:, self.face_dim + self.audio_dim :]

        # ── VMMA: Detect and replace missing modalities ──
        # If sum of absolute values is near zero, the modality is missing
        face_missing_mask = (face_x.abs().sum(dim=-1, keepdim=True) < 1e-6).float()
        audio_missing_mask = (audio_x.abs().sum(dim=-1, keepdim=True) < 1e-6).float()

        face_x = face_x * (1 - face_missing_mask) + self.missing_face_token * face_missing_mask
        audio_x = audio_x * (1 - audio_missing_mask) + self.missing_audio_token * audio_missing_mask

        # Project modalities
        f_feat = self.face_proj(face_x)
        a_feat = self.audio_proj(audio_x)
        s_feat = self.spatial_proj(spatial_x)

        # ── VSTR: Lip Reading Confidence ──
        # Predict if the audio matches the lip movements of this specific face
        av_context = torch.cat([f_feat, a_feat], dim=-1)
        lip_confidence = self.lip_reading_scorer(av_context) # (N, 1)

        # If face or audio is missing, force lip confidence to 0 (trigger fallback)
        missing_any = torch.max(face_missing_mask, audio_missing_mask)
        lip_confidence = lip_confidence * (1 - missing_any)

        # ── Fallback: Spatial/Headpose Confidence ──
        fallback_confidence = self.fallback_scorer(s_feat) # (N, 1)

        # ── Context-Aware Target Assignment ──
        # If lip reading is confident, heavily weight the audio-visual features.
        # If lip reading fails (or is missing), rely on the spatial/headpose features.
        fused_vstr = (f_feat + a_feat) * lip_confidence
        fused_fallback = s_feat * (1 - lip_confidence) * fallback_confidence

        combined = fused_vstr + fused_fallback + s_feat # Retain baseline spatial context

        return self.out_proj(combined)

# ═══════════════════════════════════════════════════════════════════
# Main ST-GAT Model
# ═══════════════════════════════════════════════════════════════════

class SpatioTemporalGAT(nn.Module):
    """
    Spatio-Temporal Graph Attention Network for TTM prediction.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ── Input projection (Audio-Visual Cross-Attention Fusion) ──
        self.input_proj = AudioVisualCrossAttentionFusion(cfg)

        # ── Edge type embedding ──
        self.use_edge_type = cfg.use_edge_type
        edge_dim = 64 if cfg.use_edge_type else None
        if cfg.use_edge_type:
            self.edge_embed = EdgeTypeEmbedding(num_edge_types=cfg.NUM_EDGE_TYPES, embed_dim=64)

        # ── GAT layers ──
        self.gat_layers = nn.ModuleList()

        in_dim = cfg.hidden_dim
        for i in range(cfg.num_gat_layers):
            if i < cfg.num_gat_layers - 1:
                # Intermediate layers: multi-head with concat
                out_dim = cfg.hidden_dim // cfg.num_heads
                layer = ResidualGATBlock(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=cfg.num_heads,
                    dropout=cfg.gat_dropout,
                    concat=True,
                    edge_dim=edge_dim,
                )
                in_dim = out_dim * cfg.num_heads  # = hidden_dim
            else:
                # Final layer: multi-head with mean (no concat)
                layer = ResidualGATBlock(
                    in_channels=in_dim,
                    out_channels=cfg.hidden_dim,
                    heads=cfg.num_heads,
                    dropout=cfg.gat_dropout,
                    concat=False,
                    edge_dim=edge_dim,
                )
                in_dim = cfg.hidden_dim

            self.gat_layers.append(layer)

        # ── Node-level classifier ──
        # Use LayerNorm (not BatchNorm1d) — BN is unsafe with AMP and variable
        # PyG batch sizes: zero-variance mini-batches produce NaN running stats.
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),            # normalize GAT output — stable regardless of batch size
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.LayerNorm(cfg.hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(cfg.hidden_dim // 2, cfg.hidden_dim // 4),
            nn.ELU(),
            nn.Dropout(cfg.classifier_dropout * 0.5),
            nn.Linear(cfg.hidden_dim // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: PyG Batch or Data object

        Returns:
            logits: (N,) per-node TTM logits
        """
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type if hasattr(data, "edge_type") else None

        # Input projection
        x = self.input_proj(x)

        # Edge type embeddings
        edge_attr = None
        if self.use_edge_type and edge_type is not None:
            edge_attr = self.edge_embed(edge_type)

        # GAT message passing
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_attr=edge_attr)

        # Node-level classification
        logits = self.classifier(x).squeeze(-1)  # (N,)

        return logits

    def predict_per_person(self, data, aggregate: str = "mean") -> dict:
        """
        Make per-person-per-clip predictions by aggregating node predictions.

        Returns:
            dict: {(clip_uid, person_id): (probability, label)}
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)
            probs = torch.sigmoid(logits)

        person_ids = data.person_ids.cpu().numpy()
        labels = data.y.cpu().numpy()
        probs = probs.cpu().numpy()

        # Handle batched data
        if hasattr(data, "batch"):
            batch_ids = data.batch.cpu().numpy()
            clip_uids = data.clip_uid if isinstance(data.clip_uid, list) else [data.clip_uid]
        else:
            batch_ids = np.zeros(len(probs), dtype=int)
            clip_uids = [data.clip_uid]

        results = {}
        from collections import defaultdict
        person_preds = defaultdict(list)
        person_labels = defaultdict(list)

        for i in range(len(probs)):
            bid = batch_ids[i]
            cuid = clip_uids[bid] if bid < len(clip_uids) else "unknown"
            pid = person_ids[i]
            key = (cuid, pid)
            person_preds[key].append(probs[i])
            person_labels[key].append(labels[i])

        for key in person_preds:
            preds = np.array(person_preds[key])
            lbls = np.array(person_labels[key])
            if aggregate == "mean":
                agg_prob = preds.mean()
            else:  # max
                agg_prob = preds.max()
            agg_label = int(lbls.max())  # if any frame is positive, person is positive
            results[key] = (float(agg_prob), agg_label)

        return results


# ═══════════════════════════════════════════════════════════════════
# Model with Temporal Attention Pooling (Enhanced version)
# ═══════════════════════════════════════════════════════════════════

class TemporalAttentionPooling(nn.Module):
    """
    Attention-based pooling across temporal dimension for each person.
    Learns to weight different frames differently.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor, person_ids: torch.Tensor,
                batch: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pool node features per person using learned attention.

        Args:
            x: (N, D) node features after GNN
            person_ids: (N,) person ID per node
            batch: (N,) batch index per node (for batched graphs)

        Returns:
            pooled: (P, D) pooled features per person
            pooled_person_ids: (P,) person IDs
            pooled_batch: (P,) batch indices
        """
        # Compute attention scores
        attn_scores = self.attention(x).squeeze(-1)  # (N,)

        # Use tuple-based unique keys to avoid integer arithmetic collisions
        # (e.g. batch * 100000 + person_id breaks when person_ids >= 100000)
        if batch is not None:
            unique_key_list = list(zip(batch.tolist(), person_ids.tolist()))
        else:
            unique_key_list = list(zip([0] * len(person_ids), person_ids.tolist()))

        unique_keys = list(dict.fromkeys(unique_key_list))  # preserves order, deduplicates
        pooled_list = []
        pid_list = []
        batch_list = []

        # O(N) dict-based grouping — avoids the O(N×P) list comprehension mask
        from collections import defaultdict
        groups = defaultdict(list)
        for node_i, uk in enumerate(unique_key_list):
            groups[uk].append(node_i)

        pooled_list = []
        pid_list = []
        batch_list = []

        for uk, indices in groups.items():
            idx_t = torch.tensor(indices, device=x.device, dtype=torch.long)
            node_feats = x[idx_t]            # (K, D)
            scores     = attn_scores[idx_t]  # (K,)

            weights = F.softmax(scores, dim=0).unsqueeze(-1)  # (K, 1)
            pooled_list.append((weights * node_feats).sum(dim=0))

            pid_list.append(person_ids[idx_t[0]])
            if batch is not None:
                batch_list.append(batch[idx_t[0]])

        pooled = torch.stack(pooled_list)
        pooled_pids = torch.stack(pid_list)
        pooled_batch = torch.stack(batch_list) if batch_list else None

        return pooled, pooled_pids, pooled_batch


class SpatioTemporalGATWithPooling(SpatioTemporalGAT):
    """
    Enhanced ST-GAT with temporal attention pooling for per-person prediction.
    Produces one prediction per person per clip instead of per node.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.temporal_pool = TemporalAttentionPooling(cfg.hidden_dim)

        # Override classifier for pooled features
        self.person_classifier = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.LayerNorm(cfg.hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns both node-level and person-level logits.

        Returns:
            node_logits: (N,) per-node logits
            person_logits: (P,) per-person logits (after temporal pooling)
        """
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type if hasattr(data, "edge_type") else None

        # Input projection
        x = self.input_proj(x)

        # Edge type embeddings
        edge_attr = None
        if self.use_edge_type and edge_type is not None:
            edge_attr = self.edge_embed(edge_type)

        # GAT message passing
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_attr=edge_attr)

        # Node-level logits
        node_logits = self.classifier(x).squeeze(-1)

        # Person-level pooling and classification
        batch = data.batch if hasattr(data, "batch") else None
        pooled, _, _ = self.temporal_pool(x, data.person_ids, batch)
        person_logits = self.person_classifier(pooled).squeeze(-1)

        return node_logits, person_logits


# ═══════════════════════════════════════════════════════════════════
# Utility: compute number of parameters
# ═══════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, cfg):
    """Print model summary."""
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"  Input dim:      {cfg.node_input_dim}")
    print(f"  Hidden dim:     {cfg.hidden_dim}")
    print(f"  GAT layers:     {cfg.num_gat_layers}")
    print(f"  Attention heads: {cfg.num_heads}")
    print(f"  Edge types:     {cfg.use_edge_type}")
    print(f"  Parameters:     {count_parameters(model):,}")
    print("=" * 60 + "\n")
