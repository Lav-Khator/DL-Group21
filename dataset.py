"""
PyTorch Geometric Dataset for Spatio-Temporal Graph TTM.

Builds graphs from preprocessed features:
  - Nodes: (person_id, frame) pairs
  - Spatial edges: between persons in same frame within distance threshold
  - Temporal edges: same person across consecutive frames
  - Temporal skip edges: same person across non-adjacent frames (long-range)
"""

import math
import os
import pickle
import json
import hashlib
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Sampler
from torch_geometric.data import Data, Dataset, InMemoryDataset
from tqdm import tqdm

from config import Config


# ─────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────

def build_graph(clip_data: dict, cfg: Config) -> Data | None:
    """
    Build a PyG Data object (spatio-temporal graph) for one clip.

    Node ordering: nodes are indexed by (person_id, frame) pairs,
    sorted by (frame, person_id) for consistent ordering.

    Returns:
        PyG Data with:
            x:          (N, node_input_dim) node features
            edge_index: (2, E) edge indices
            edge_type:  (E,) edge type labels
            y:          (N,) binary TTM labels
            person_ids: (N,) person ID for each node (for aggregation)
            frame_ids:  (N,) frame index for each node
            clip_uid:   string identifier
    """
    metadata = clip_data["metadata"]
    bbox_features = clip_data["bbox_features"]
    face_features = clip_data.get("face_features", {})
    audio_features = clip_data.get("audio_features", None)
    headpose_features = clip_data.get("headpose_features", {})

    labels = clip_data["labels"]
    bboxes = clip_data["bboxes"]

    person_ids = sorted(metadata["person_ids"])
    frame_indices = sorted(metadata["frame_indices"])

    # Hard ceiling to prevent 5+ GB OOM errors on outlier validation clips
    # 1000 frames @ 10 FPS = 100 seconds of continuous video (plenty of context)
    if len(frame_indices) > 1000:
        frame_indices = frame_indices[:1000]

    # Build node list: sorted by (frame, person_id)
    node_keys = []
    for frame in frame_indices:
        for pid in person_ids:
            if (pid, frame) in labels:
                node_keys.append((pid, frame))

    if len(node_keys) == 0:
        return None

    # ── Inject Virtual Ego Nodes ─────────────────────────────────
    unique_frames = sorted(list(set(frame for _, frame in node_keys)))
    ego_pid = "VIRTUAL_EGO"
    frame_to_ego_idx = {}
    
    for frame in unique_frames:
        node_keys.append((ego_pid, frame))
        # Ego bbox (center of screen)
        bbox_features[(ego_pid, frame)] = np.array([0.0, 0.0, 1.0, 1.0, 0.5, 0.5], dtype=np.float32)
        labels[(ego_pid, frame)] = 0
        headpose_features[(ego_pid, frame)] = np.zeros(cfg.headpose_feat_dim, dtype=np.float32)
        
    # Create node index mapping
    key_to_idx = {key: idx for idx, key in enumerate(node_keys)}
    num_nodes = len(node_keys)
    
    # Store ego idx mapping for easy edge construction
    for frame in unique_frames:
        frame_to_ego_idx[frame] = key_to_idx[(ego_pid, frame)]

    # ── Build node features ──────────────────────────────────────
    node_feats = []
    node_labels = []
    node_person_ids = []
    node_frame_ids = []

    for pid, frame in node_keys:
        feat_parts = []

        # Face features (ResNet-50 / DINO)
        if cfg.feature_mode == "full":
            if (pid, frame) in face_features:
                feat_parts.append(face_features[(pid, frame)])
            else:
                feat_parts.append(np.zeros(cfg.face_feat_dim, dtype=np.float32))

        # Audio features (MFCC / HuBERT)
        if cfg.feature_mode == "full":
            if audio_features is not None and frame < len(audio_features):
                feat_parts.append(audio_features[frame])
            else:
                feat_parts.append(np.zeros(cfg.audio_feat_dim, dtype=np.float32))

        # Bbox features (always included)
        feat_parts.append(bbox_features[(pid, frame)])

        # Headpose features
        hp_feat = headpose_features.get((pid, frame), np.zeros(cfg.headpose_feat_dim, dtype=np.float32))
        feat_parts.append(hp_feat)

        node_feats.append(np.concatenate(feat_parts))
        node_labels.append(labels[(pid, frame)])
        if pid == "VIRTUAL_EGO":
            node_person_ids.append(-999)
        else:
            if pid.isdigit():
                node_person_ids.append(int(pid))
            else:
                # Use stable hash (hashlib) instead of Python's built-in hash(),
                # which is non-deterministic across processes (hash randomization).
                # This prevents person_id corruption when num_workers > 0.
                stable_id = int(hashlib.md5(pid.encode()).hexdigest(), 16) % 10000
                node_person_ids.append(stable_id)
        node_frame_ids.append(frame)

    x = torch.tensor(np.stack(node_feats), dtype=torch.float32)
    y = torch.tensor(node_labels, dtype=torch.float32)

    # ── Build edges ──────────────────────────────────────────────
    src_list = []
    dst_list = []
    edge_types = []

    # 1) Spatial edges: connect persons within same frame
    frame_to_nodes = defaultdict(list)
    for idx, (pid, frame) in enumerate(node_keys):
        frame_to_nodes[frame].append((idx, pid, bboxes.get((pid, frame), None)))

    for frame, nodes_in_frame in frame_to_nodes.items():
        for i in range(len(nodes_in_frame)):
            for j in range(i + 1, len(nodes_in_frame)):
                idx_i, pid_i, bbox_i = nodes_in_frame[i]
                idx_j, pid_j, bbox_j = nodes_in_frame[j]

                if bbox_i is None or bbox_j is None:
                    continue

                # Compute distance between bbox centers
                cx_i = bbox_i[0] + bbox_i[2] / 2
                cy_i = bbox_i[1] + bbox_i[3] / 2
                cx_j = bbox_j[0] + bbox_j[2] / 2
                cy_j = bbox_j[1] + bbox_j[3] / 2
                dist = np.sqrt((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2)

                if dist < cfg.spatial_threshold:
                    # Bidirectional edges
                    src_list.extend([idx_i, idx_j])
                    dst_list.extend([idx_j, idx_i])
                    edge_types.extend([cfg.EDGE_SPATIAL, cfg.EDGE_SPATIAL])

    # 2) Temporal edges: same person across consecutive frames
    person_to_nodes = defaultdict(list)
    for idx, (pid, frame) in enumerate(node_keys):
        person_to_nodes[pid].append((frame, idx))

    for pid, frame_nodes in person_to_nodes.items():
        frame_nodes.sort(key=lambda x: x[0])  # sort by frame

        for i in range(len(frame_nodes)):
            for j in range(i + 1, len(frame_nodes)):
                frame_i, idx_i = frame_nodes[i]
                frame_j, idx_j = frame_nodes[j]
                diff = frame_j - frame_i

                if diff <= cfg.temporal_stride:
                    # Short-range temporal edge
                    src_list.extend([idx_i, idx_j])
                    dst_list.extend([idx_j, idx_i])
                    edge_types.extend([cfg.EDGE_TEMPORAL, cfg.EDGE_TEMPORAL])
                elif diff <= cfg.temporal_skip:
                    # Long-range temporal skip edge
                    src_list.extend([idx_i, idx_j])
                    dst_list.extend([idx_j, idx_i])
                    edge_types.extend([cfg.EDGE_TEMPORAL_SKIP, cfg.EDGE_TEMPORAL_SKIP])

    # 3) Headpose edges: directed edges based on yaw
    node_hp_yaw = {}
    for idx, (pid, frame) in enumerate(node_keys):
        hp = headpose_features.get((pid, frame), None)
        if hp is not None and len(hp) >= 2:
            node_hp_yaw[idx] = hp[1]  # yaw is index 1 of the 3-dim vector
        else:
            node_hp_yaw[idx] = None

    for frame, nodes_in_frame in frame_to_nodes.items():
        for i in range(len(nodes_in_frame)):
            for j in range(i + 1, len(nodes_in_frame)):
                idx_i, pid_i, bbox_i = nodes_in_frame[i]
                idx_j, pid_j, bbox_j = nodes_in_frame[j]

                if bbox_i is None or bbox_j is None:
                    continue

                cx_i = bbox_i[0] + bbox_i[2] / 2
                cy_i = bbox_i[1] + bbox_i[3] / 2
                cx_j = bbox_j[0] + bbox_j[2] / 2
                cy_j = bbox_j[1] + bbox_j[3] / 2

                # A→B: person i looking toward person j
                yaw_i = node_hp_yaw.get(idx_i)
                if yaw_i is not None:
                    angle_i_to_j = math.atan2(cy_j - cy_i, cx_j - cx_i)
                    yaw_i_rad = math.radians(yaw_i)
                    diff_i = (yaw_i_rad - angle_i_to_j + math.pi) % (2 * math.pi) - math.pi
                    if abs(diff_i) < 0.4:
                        src_list.append(idx_i)
                        dst_list.append(idx_j)
                        edge_types.append(cfg.EDGE_HEADPOSE)

                # B→A: person j looking toward person i
                yaw_j = node_hp_yaw.get(idx_j)
                if yaw_j is not None:
                    angle_j_to_i = math.atan2(cy_i - cy_j, cx_i - cx_j)
                    yaw_j_rad = math.radians(yaw_j)
                    diff_j = (yaw_j_rad - angle_j_to_i + math.pi) % (2 * math.pi) - math.pi
                    if abs(diff_j) < 0.4:
                        src_list.append(idx_j)
                        dst_list.append(idx_i)
                        edge_types.append(cfg.EDGE_HEADPOSE)

    # 4) Virtual Ego Edges
    for frame, ego_idx in frame_to_ego_idx.items():
        if frame in frame_to_nodes:
            for idx_i, pid_i, bbox_i in frame_to_nodes[frame]:
                # Connect Person <-> Ego
                src_list.extend([idx_i, ego_idx])
                dst_list.extend([ego_idx, idx_i])
                edge_types.extend([cfg.EDGE_EGO, cfg.EDGE_EGO])
                
    # 5) Virtual Ego Temporal Edges
    for i in range(len(unique_frames) - 1):
        frame_i = unique_frames[i]
        frame_j = unique_frames[i+1]
        if frame_j - frame_i <= cfg.temporal_stride:
            idx_i = frame_to_ego_idx[frame_i]
            idx_j = frame_to_ego_idx[frame_j]
            src_list.extend([idx_i, idx_j])
            dst_list.extend([idx_j, idx_i])
            edge_types.extend([cfg.EDGE_TEMPORAL, cfg.EDGE_TEMPORAL])

    # Add self-loops
    for i in range(num_nodes):
        src_list.append(i)
        dst_list.append(i)
        edge_types.append(cfg.EDGE_SELF)  # dedicated self-loop type (not temporal)

    if len(src_list) == 0:
        # No edges at all — shouldn't happen with self-loops, but safety check
        return None

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=y,
        person_ids=torch.tensor(node_person_ids, dtype=torch.long),
        frame_ids=torch.tensor(node_frame_ids, dtype=torch.long),
        num_nodes=num_nodes,
    )
    data.clip_uid = metadata["clip_uid"]

    return data


# ─────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────

class TTMGraphDataset(Dataset):
    """
    PyG Dataset that loads preprocessed clip features and builds graphs on-the-fly.
    Supports class-balanced oversampling of positive (TTM=1) clips.
    """

    def __init__(self, split: str, cfg: Config, oversample: bool = False):
        self.split = split
        self.cfg = cfg
        self.oversample = oversample

        feature_dir = os.path.join(cfg.feature_dir, split)

        # Find all preprocessed clip files
        self.clip_files = sorted([
            os.path.join(feature_dir, f)
            for f in os.listdir(feature_dir)
            if f.endswith(".pkl")
        ])

        if len(self.clip_files) == 0:
            raise RuntimeError(
                f"No preprocessed files found in {feature_dir}. "
                f"Run preprocess.py first!"
            )

        print(f"[{split}] Found {len(self.clip_files)} preprocessed clips")

        # Identify positive vs negative clips for oversampling
        if oversample:
            self._identify_positive_clips()

        super().__init__(root=None, transform=None)

    def _identify_positive_clips(self):
        """Scan clips to identify which ones contain TTM=1 labels."""
        print(f"  Scanning clips for class balance ...")
        positive_indices = []
        negative_indices = []

        for idx, path in enumerate(tqdm(self.clip_files, desc="Scanning")):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                has_positive = any(v == 1 for v in data["labels"].values())
                if has_positive:
                    positive_indices.append(idx)
                else:
                    negative_indices.append(idx)
            except Exception:
                negative_indices.append(idx)

        n_pos = len(positive_indices)
        n_neg = len(negative_indices)
        print(f"  Positive clips: {n_pos}, Negative clips: {n_neg}")

        # Oversample positive clips
        if n_pos > 0 and self.cfg.oversample_ratio > 1.0:
            repeat = int(self.cfg.oversample_ratio)
            oversampled_positives = positive_indices * repeat
            self._sample_indices = negative_indices + oversampled_positives
            print(f"  After oversampling: {len(self._sample_indices)} total clips "
                  f"({n_neg} neg + {len(oversampled_positives)} pos)")
        else:
            self._sample_indices = list(range(len(self.clip_files)))

    def len(self) -> int:
        if self.oversample and hasattr(self, "_sample_indices"):
            return len(self._sample_indices)
        return len(self.clip_files)

    def get(self, idx: int) -> Data:
        if self.oversample and hasattr(self, "_sample_indices"):
            actual_idx = self._sample_indices[idx]
        else:
            actual_idx = idx

        path = self.clip_files[actual_idx]

        with open(path, "rb") as f:
            clip_data = pickle.load(f)

        # Temporal frame dropout augmentation (training only)
        # Randomly drop 20% of frames with 50% probability — forces the model
        # to learn from partial temporal context and improves generalization.
        if self.split == "train" and random.random() < 0.5:
            all_frames = clip_data["metadata"]["frame_indices"]
            min_frames = max(self.cfg.min_frames_per_clip, 3)
            keep_n = max(min_frames, int(len(all_frames) * 0.8))
            if keep_n < len(all_frames):
                kept = sorted(random.sample(all_frames, keep_n))
                clip_data = dict(clip_data)  # shallow copy to avoid mutating cached data
                clip_data["metadata"] = dict(clip_data["metadata"])
                clip_data["metadata"]["frame_indices"] = kept

        graph = build_graph(clip_data, self.cfg)

        if graph is None:
            # Return a minimal dummy graph if construction fails
            graph = Data(
                x=torch.zeros((1, self.cfg.node_input_dim), dtype=torch.float32),
                edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                edge_type=torch.tensor([0], dtype=torch.long),
                y=torch.tensor([0.0]),
                person_ids=torch.tensor([-1], dtype=torch.long),
                frame_ids=torch.tensor([0], dtype=torch.long),
                num_nodes=1,
            )
            graph.clip_uid = "dummy"

        return graph


# ─────────────────────────────────────────────────────────────────
# Alternative: Direct JSON dataset (no preprocessing needed)
# Builds graphs from raw annotation JSON + bbox features only
# ─────────────────────────────────────────────────────────────────

class TTMGraphDatasetDirect(Dataset):
    """
    Builds graph dataset directly from annotation JSON without preprocessing.
    Uses only bbox metadata features (lite mode).
    Good for quick prototyping and testing.
    """

    def __init__(self, json_path: str, cfg: Config, oversample: bool = False):
        import copy
        self.cfg = copy.deepcopy(cfg)  # avoid mutating the caller's config
        self.cfg.feature_mode = "lite"
        self.cfg.node_input_dim = self.cfg.bbox_feat_dim + self.cfg.headpose_feat_dim
        cfg = self.cfg  # use local alias for the rest of __init__

        print(f"Loading annotations from {json_path} ...")
        with open(json_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for key in ["clips", "data", "annotations"]:
                if key in data:
                    data = data[key]
                    break
            if isinstance(data, dict):
                all_entries = []
                for clip_entries in data.values():
                    all_entries.extend(clip_entries)
                data = all_entries

        # Group by clip_uid
        self.clip_data = defaultdict(list)
        for entry in data:
            self.clip_data[entry["clip_uid"]].append(entry)

        self.clip_uids = sorted(self.clip_data.keys())
        print(f"  Loaded {len(self.clip_uids)} clips")

        # Class stats
        labels = [e["ttm_label"] for e in data]
        n_pos = sum(labels)
        print(f"  Labels: {n_pos} positive / {len(labels) - n_pos} negative")

        # Oversampling
        if oversample:
            positive_clips = []
            negative_clips = []
            for i, uid in enumerate(self.clip_uids):
                has_pos = any(e["ttm_label"] == 1 for e in self.clip_data[uid])
                if has_pos:
                    positive_clips.append(i)
                else:
                    negative_clips.append(i)
            repeat = int(cfg.oversample_ratio)
            self._sample_indices = negative_clips + positive_clips * repeat
            print(f"  Oversampled: {len(self._sample_indices)} clips")
        else:
            self._sample_indices = list(range(len(self.clip_uids)))

        super().__init__(root=None, transform=None)

    def len(self) -> int:
        return len(self._sample_indices)

    def get(self, idx: int) -> Data:
        actual_idx = self._sample_indices[idx]
        clip_uid = self.clip_uids[actual_idx]
        entries = self.clip_data[clip_uid]

        # Get all frame indices and apply frame_sample_rate (e.g. 3 FPS)
        all_frames = sorted(set(int(e["frame"]) for e in entries))
        if self.cfg.frame_sample_rate > 1:
            frame_set = set([f for f in all_frames if f % self.cfg.frame_sample_rate == 0])
            all_frames = sorted(list(frame_set))
            entries = [e for e in entries if int(e["frame"]) in frame_set]

        # Build a lite clip_data dict for graph construction
        clip_dict = {
            "face_features": {},
            "audio_features": None,
            "bbox_features": {},
            "headpose_features": {},
            "labels": {},
            "bboxes": {},
            "metadata": {
                "clip_uid": clip_uid,
                "video_uid": entries[0]["video_uid"],
                "person_ids": sorted(set(str(e["person_id"]) for e in entries)),
                "frame_indices": all_frames,
                "img_w": 1920,
                "img_h": 1080,
                "fps": 30.0,
            },
        }

        for e in entries:
            pid = str(e["person_id"])
            frame = int(e["frame"])
            bbox = e["bbox"]
            clip_dict["bbox_features"][(pid, frame)] = compute_bbox_features_static(bbox)
            clip_dict["labels"][(pid, frame)] = int(e["ttm_label"])
            clip_dict["bboxes"][(pid, frame)] = np.array(bbox, dtype=np.float32)
            clip_dict["headpose_features"][(pid, frame)] = np.zeros(3, dtype=np.float32)

        graph = build_graph(clip_dict, self.cfg)

        if graph is None:
            graph = Data(
                x=torch.zeros((1, self.cfg.node_input_dim), dtype=torch.float32),
                edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                edge_type=torch.tensor([0], dtype=torch.long),
                y=torch.tensor([0.0]),
                person_ids=torch.tensor([-1], dtype=torch.long),
                frame_ids=torch.tensor([0], dtype=torch.long),
                num_nodes=1,
            )
            graph.clip_uid = "dummy"

        return graph


def compute_bbox_features_static(bbox, img_w=1920, img_h=1080):
    """Static version for direct dataset (no module import needed)."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    aspect = w / max(h, 1e-6)
    area = (w * h) / (img_w * img_h)
    return np.array([cx, cy, nw, nh, aspect, area], dtype=np.float32)
