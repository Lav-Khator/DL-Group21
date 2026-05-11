"""
Preprocess Ego4D TTM data: extract face features (ResNet-50), audio MFCC,
and save per-clip feature bundles for fast graph dataset loading.

Usage:
    python preprocess.py --split train
    python preprocess.py --split val
    python preprocess.py --split train --mode lite   # bbox-only (no video needed)
"""

import argparse
import json
import os
import pickle
import warnings
from collections import defaultdict
from pathlib import Path

# Suppress repetitive librosa fallback warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm

from transformers import Wav2Vec2FeatureExtractor, HubertModel
import torch.nn.functional as F
from sixdrepnet import SixDRepNet
HAS_NEW_LIBS = True
from config import Config


# ─────────────────────────────────────────────────────────────────
# Feature extractor: ResNet-50 (replaces FaceNet for richer spatial/expression features)
# ─────────────────────────────────────────────────────────────────

class FaceFeatureExtractor:
    """
    Extract 1024-dim face/lip features using DINOv2-Large.
    DINOv2 provides exceptional dense patch-level features which are much 
    better for detecting fine-grained lip movements than ResNet.
    """
    def __init__(self, device: str = "cuda", output_dim: int = 1024):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dim = output_dim

        # Load DINOv2 Large
        print("Loading DINOv2-L...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(self.device).eval()
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)), # Multiple of 14 for DINOv2 patches
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_batch(self, face_crops: list[np.ndarray]) -> np.ndarray:
        if len(face_crops) == 0:
            return np.zeros((0, self.output_dim), dtype=np.float32)

        tensors = []
        for crop in face_crops:
            if crop is None or crop.size == 0:
                crop = np.zeros((224, 224, 3), dtype=np.uint8)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(crop_rgb))

        batch = torch.stack(tensors).to(self.device)
        # Extract CLS token features from DINOv2
        features = self.model(batch) 
        return features.cpu().numpy()


# ─────────────────────────────────────────────────────────────────
# Audio feature extractor — Context-Aware Speech Features
# ─────────────────────────────────────────────────────────────────

class AudioFeatureExtractor:
    """
    Extract per-frame 768-dim context-aware speech features using HuBERT
    + Gaussian temporal smoothing.

    Context-aware means: each frame's audio feature is a weighted blend of
    its neighbors (controlled by speech_context_sigma). This captures
    speech onset/offset patterns and phoneme transitions rather than
    instantaneous spectral snapshots.
    """

    def __init__(self, device: str = "cuda", context_sigma: float = 2.0):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.context_sigma = context_sigma
        if not HAS_NEW_LIBS:
            raise ImportError("transformers not installed")

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(self.device).eval()

    def extract(self, video_path: str, num_frames: int) -> np.ndarray:
        from scipy.ndimage import gaussian_filter1d

        try:
            y, sr = librosa.load(video_path, sr=16000, mono=True)
            if len(y) == 0:
                return np.zeros((num_frames, 768), dtype=np.float32)

            inputs = self.processor(y, sampling_rate=16000, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            feats = outputs.last_hidden_state  # (1, T_audio, 768)

            # Interpolate to match video frames
            feats = feats.transpose(1, 2)  # (1, 768, T_audio)
            feats = F.interpolate(feats, size=num_frames, mode='linear', align_corners=False)
            feats = feats.transpose(1, 2).squeeze(0)  # (num_frames, 768)
            feats = feats.cpu().numpy()

            # ── Context-aware smoothing ──
            # Gaussian filter along temporal axis makes each frame's feature
            # a weighted average of neighbors → captures speech patterns
            if self.context_sigma > 0 and num_frames > 1:
                feats = gaussian_filter1d(feats, sigma=self.context_sigma, axis=0)

            return feats

        except Exception as e:
            print(f"    [WARN] Audio extraction failed for {video_path}: {e}")
            return np.zeros((num_frames, 768), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
# Bbox utilities
# ─────────────────────────────────────────────────────────────────

def expand_bbox(x, y, w, h, pad_ratio: float, img_w: int, img_h: int):
    """Expand bbox by pad_ratio on each side, clamp to image bounds."""
    pad_w = w * pad_ratio
    pad_h = h * pad_ratio
    x1 = max(0, int(x - pad_w))
    y1 = max(0, int(y - pad_h))
    x2 = min(img_w, int(x + w + pad_w))
    y2 = min(img_h, int(y + h + pad_h))
    return x1, y1, x2, y2


def compute_bbox_features(bbox, img_w: int = 1920, img_h: int = 1080) -> np.ndarray:
    """
    Compute normalized bbox features: [cx, cy, w, h, aspect_ratio, area_ratio].
    All values normalized to [0, 1] range.
    """
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    aspect = w / max(h, 1e-6)
    area = (w * h) / (img_w * img_h)
    return np.array([cx, cy, nw, nh, aspect, area], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
# Video reader
# ─────────────────────────────────────────────────────────────────

def find_video_path(clip_uid: str, video_uid: str, cfg: Config) -> str | None:
    """Try to locate the video file for a given clip."""
    # Try clip-based path first
    candidates = [
        os.path.join(cfg.clips_dir, f"{clip_uid}.mp4"),
        os.path.join(cfg.clips_dir, clip_uid, f"{clip_uid}.mp4"),
        os.path.join(cfg.videos_dir, f"{video_uid}.mp4"),
        os.path.join(cfg.data_root, "clips", f"{clip_uid}.mp4"),
        os.path.join(cfg.data_root, "clips_hq", f"{clip_uid}.mp4"),
        os.path.join(cfg.data_root, "video_clips", f"{clip_uid}.mp4"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


import decord

def read_video_frames(video_path: str, frame_indices: list[int]) -> dict[int, np.ndarray]:
    """
    Lightning-fast random frame access using Decord.
    """
    try:
        # Load the video reader
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        
        # Ensure we don't ask for frames beyond the video length
        max_len = len(vr)
        valid_indices = [idx for idx in frame_indices if idx < max_len]
        
        if not valid_indices:
            return {}

        # Fetch all requested frames instantly in a single C++ batched call
        frames_batch = vr.get_batch(valid_indices).asnumpy()
        
        frames_dict = {}
        for i, idx in enumerate(valid_indices):
            # Decord loads in RGB. OpenCV loads in BGR.
            # We convert RGB -> BGR here ([::-1]) so we don't break your existing
            # cv2.cvtColor logic down below in the DINOv2 extractor!
            frames_dict[idx] = frames_batch[i][:, :, ::-1]
            
        return frames_dict
        
    except Exception as e:
        print(f"    [WARN] Decord failed for {video_path}: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────
# Main preprocessing
# ─────────────────────────────────────────────────────────────────

def load_annotations(json_path: str) -> dict:
    """
    Load TTM annotations and group by clip_uid.

    Returns:
        clip_data: dict[clip_uid] -> list of annotation entries
    """
    print(f"Loading annotations from {json_path} ...")
    with open(json_path, "r") as f:
        data = json.load(f)

    # Handle both list and dict formats
    if isinstance(data, dict):
        # Some versions wrap in a dict with a key
        for key in ["clips", "data", "annotations"]:
            if key in data:
                data = data[key]
                break
        if isinstance(data, dict):
            # Might be keyed by clip_uid already
            return data

    # Group by clip_uid
    clip_data = defaultdict(list)
    for entry in data:
        clip_data[entry["clip_uid"]].append(entry)

    print(f"  Found {len(clip_data)} clips, {len(data)} total entries")

    # Print class distribution
    labels = [e["ttm_label"] for e in data]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"  Class distribution: TTM=0: {n_neg} ({n_neg/len(labels)*100:.1f}%), "
          f"TTM=1: {n_pos} ({n_pos/len(labels)*100:.1f}%)")

    return dict(clip_data)


def preprocess_clip_full(clip_uid: str, entries: list, cfg: Config,
                         face_extractor: FaceFeatureExtractor,
                         headpose_model,
                         audio_extractor: AudioFeatureExtractor) -> dict | None:
    """
    Extract features for a single clip (full mode: visual + audio).

    Returns dict with:
        - face_features: dict[(person_id, frame)] -> (2048,) numpy array
        - audio_features: (num_frames, n_mfcc) numpy array
        - bbox_features: dict[(person_id, frame)] -> (6,) numpy array
        - labels: dict[(person_id, frame)] -> int
        - bboxes: dict[(person_id, frame)] -> (4,) [x, y, w, h]
        - metadata: dict with clip info
    """
    video_uid = entries[0]["video_uid"]
    video_path = find_video_path(clip_uid, video_uid, cfg)

    if video_path is None:
        return None

    # Organize entries by (person_id, frame)
    entry_map = {}
    frame_indices = set()
    person_ids = set()
    for e in entries:
        pid = str(e["person_id"])
        frame = int(e["frame"])
        entry_map[(pid, frame)] = e
        frame_indices.add(frame)
        person_ids.add(pid)

    frame_indices = sorted(frame_indices)
    person_ids = sorted(person_ids)

    if len(frame_indices) < cfg.min_frames_per_clip:
        return None

    # Apply frame_sample_rate (e.g. 10 = 3 FPS) to keep temporal story but reduce nodes
    if cfg.frame_sample_rate > 1:
        frame_indices = [f for f in frame_indices if f % cfg.frame_sample_rate == 0]
        # Filter entry_map
        frame_set = set(frame_indices)
        entry_map = {k: v for k, v in entry_map.items() if k[1] in frame_set}

    # Read video frames
    frames_dict = read_video_frames(video_path, frame_indices)
    if len(frames_dict) == 0:
        return None

    # Get video dimensions from first frame
    first_frame = next(iter(frames_dict.values()))
    img_h, img_w = first_frame.shape[:2]

    # Extract face crops and features
    face_features = {}
    bbox_features = {}
    labels = {}
    bboxes_out = {}

    # Collect face crops in batches for efficiency
    crop_keys = []
    crop_images = []
    headpose_features = {}

    for (pid, frame), entry in entry_map.items():
        if frame not in frames_dict:
            continue

        bbox = entry["bbox"]
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

        # Expand and clamp bbox
        x1, y1, x2, y2 = expand_bbox(x, y, w, h, cfg.bbox_pad_ratio, img_w, img_h)

        # Crop face
        img = frames_dict[frame]
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            crop = np.zeros((224, 224, 3), dtype=np.uint8)

        crop_keys.append((pid, frame))
        crop_images.append(crop)

        # Bbox features (always computed)
        bbox_features[(pid, frame)] = compute_bbox_features(bbox, img_w, img_h)
        labels[(pid, frame)] = int(entry["ttm_label"])
        bboxes_out[(pid, frame)] = np.array(bbox, dtype=np.float32)

        # Headpose features
        if headpose_model is not None:
            try:
                pitch, yaw, roll = headpose_model.predict(crop)
                hp_feat = np.array([pitch, yaw, roll], dtype=np.float32).flatten()
            except Exception:
                hp_feat = np.zeros(3, dtype=np.float32)
        else:
            hp_feat = np.zeros(3, dtype=np.float32)
        headpose_features[(pid, frame)] = hp_feat

    # Extract face features in batch
    if len(crop_images) > 0:
        # Process in sub-batches to avoid OOM
        batch_size = 64
        all_feats = []
        for i in range(0, len(crop_images), batch_size):
            batch = crop_images[i:i + batch_size]
            feats = face_extractor.extract_batch(batch)
            all_feats.append(feats)
        all_feats = np.concatenate(all_feats, axis=0)

        for idx, key in enumerate(crop_keys):
            face_features[key] = all_feats[idx]

    # Extract audio features
    num_frames = max(frame_indices) + 1
    # Get FPS from video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    audio_features = audio_extractor.extract(video_path, num_frames=num_frames)

    return {
        "face_features": face_features,
        "audio_features": audio_features,
        "bbox_features": bbox_features,
        "headpose_features": headpose_features,
        "labels": labels,
        "bboxes": bboxes_out,
        "metadata": {
            "clip_uid": clip_uid,
            "video_uid": video_uid,
            "person_ids": person_ids,
            "frame_indices": frame_indices,
            "img_w": img_w,
            "img_h": img_h,
            "fps": fps,
        },
    }


def preprocess_clip_lite(clip_uid: str, entries: list, cfg: Config) -> dict | None:
    """
    Lightweight preprocessing using only bbox metadata (no video files needed).
    Good for testing the pipeline and as a baseline.
    """
    video_uid = entries[0]["video_uid"]

    entry_map = {}
    frame_indices = set()
    person_ids = set()
    for e in entries:
        pid = str(e["person_id"])
        frame = int(e["frame"])
        entry_map[(pid, frame)] = e
        frame_indices.add(frame)
        person_ids.add(pid)

    frame_indices = sorted(frame_indices)
    person_ids = sorted(person_ids)

    if len(frame_indices) < cfg.min_frames_per_clip:
        return None

    bbox_features = {}
    headpose_features = {}
    labels = {}
    bboxes_out = {}

    for (pid, frame), entry in entry_map.items():
        bbox = entry["bbox"]
        bbox_features[(pid, frame)] = compute_bbox_features(bbox)
        labels[(pid, frame)] = int(entry["ttm_label"])
        bboxes_out[(pid, frame)] = np.array(bbox, dtype=np.float32)
        headpose_features[(pid, frame)] = np.zeros(3, dtype=np.float32)


    return {
        "face_features": {},       # empty in lite mode
        "audio_features": None,    # empty in lite mode
        "bbox_features": bbox_features,
        "headpose_features": headpose_features,
        "labels": labels,
        "bboxes": bboxes_out,
        "metadata": {
            "clip_uid": clip_uid,
            "video_uid": video_uid,
            "person_ids": person_ids,
            "frame_indices": frame_indices,
            "img_w": 1920,
            "img_h": 1080,
            "fps": 30.0,
        },
    }


def preprocess_split(split: str, cfg: Config):
    """Preprocess an entire data split (train or val)."""
    import time

    json_path = cfg.train_json_path if split == "train" else cfg.val_json_path
    clip_data = load_annotations(json_path)

    save_dir = os.path.join(cfg.feature_dir, split)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize feature extractor
    face_extractor = None
    headpose_model = None
    audio_extractor = None
    if cfg.feature_mode == "full":
        print(f"Initializing {cfg.face_model} face feature extractor ...")
        face_extractor = FaceFeatureExtractor(device=cfg.device, output_dim=cfg.face_feat_dim)
        print(f"Initializing HuBERT audio feature extractor (context σ={cfg.speech_context_sigma}) ...")
        audio_extractor = AudioFeatureExtractor(device=cfg.device, context_sigma=cfg.speech_context_sigma)
        print("Initializing 6DRepNet headpose model ...")
        if HAS_NEW_LIBS:
            headpose_model = SixDRepNet(gpu_id=0 if cfg.device == "cuda" else -1)
        else:
            headpose_model = None

    success = 0
    skipped = 0
    already_done = 0
    total = len(clip_data)
    clip_times = []

    split_start = time.time()
    print(f"\n{'='*60}")
    print(f"  Preprocessing {split} split: {total} clips")
    print(f"  Mode: {cfg.feature_mode} | Save dir: {save_dir}")
    print(f"{'='*60}\n")

    for i, (clip_uid, entries) in enumerate(clip_data.items()):
        save_path = os.path.join(save_dir, f"{clip_uid}.pkl")

        # Skip if already processed
        if os.path.isfile(save_path):
            already_done += 1
            success += 1
            continue

        clip_start = time.time()

        try:
            if cfg.feature_mode == "full":
                result = preprocess_clip_full(clip_uid, entries, cfg, face_extractor, headpose_model, audio_extractor)
            else:
                result = preprocess_clip_lite(clip_uid, entries, cfg)

            if result is None:
                skipped += 1
                continue

            with open(save_path, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            success += 1

        except Exception as e:
            print(f"  [ERROR] Failed on clip {clip_uid}: {e}")
            skipped += 1

        clip_elapsed = time.time() - clip_start
        clip_times.append(clip_elapsed)

        # Print progress every 10 clips or at milestones
        processed = i + 1 - already_done
        if processed > 0 and (processed % 10 == 0 or processed <= 3 or (i + 1) == total):
            avg_time = sum(clip_times) / len(clip_times)
            remaining = total - (i + 1)
            eta_seconds = remaining * avg_time
            elapsed = time.time() - split_start

            # Format times
            elapsed_str = _format_time(elapsed)
            eta_str = _format_time(eta_seconds)

            print(f"  [{i+1}/{total}] "
                  f"✓{success} ✗{skipped} ⏭{already_done} | "
                  f"Last: {clip_elapsed:.1f}s | Avg: {avg_time:.1f}s/clip | "
                  f"Elapsed: {elapsed_str} | ETA: {eta_str}")

    total_time = time.time() - split_start

    print(f"\n{'='*60}")
    print(f"  {split.upper()} PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Processed:     {success}/{total} clips")
    print(f"  Already cached: {already_done}")
    print(f"  Skipped:       {skipped}")
    print(f"  Total time:    {_format_time(total_time)}")
    if clip_times:
        print(f"  Avg per clip:  {sum(clip_times)/len(clip_times):.2f}s")
        print(f"  Throughput:    {len(clip_times)/max(total_time,1):.1f} clips/sec")
    print(f"{'='*60}\n")


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


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Ego4D TTM data")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "both"])
    parser.add_argument("--mode", type=str, default="full", choices=["full", "lite"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = Config()
    cfg.feature_mode = args.mode
    cfg.device = args.device
    if args.data_root:
        cfg.data_root = args.data_root
        cfg.annotation_dir = os.path.join(args.data_root, "annotations")
        cfg.clips_dir = os.path.join(args.data_root, "clips")
        cfg.videos_dir = os.path.join(args.data_root, "full_scale")
        cfg.feature_dir = os.path.join(args.data_root, "preprocessed_features")
        os.makedirs(cfg.feature_dir, exist_ok=True)

    if args.split in ("train", "both"):
        preprocess_split("train", cfg)
    if args.split in ("val", "both"):
        preprocess_split("val", cfg)

    print("\nDone! Features saved to:", cfg.feature_dir)
