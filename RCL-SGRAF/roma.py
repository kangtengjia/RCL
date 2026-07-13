"""Shared RoMa dataset protocol for the comparison-method submodules."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Sequence

import numpy as np
from torch.utils.data import Sampler


ROMA_DATASETS = {"scenedepict", "scanrefer", "nr3d", "3dllm"}
DATASET_ALIASES = {
    "our_data": "scenedepict",
    "3d_text_retrv": "scenedepict",
    "scenedepict-3d2t": "scenedepict",
    "3d_llm": "3dllm",
    "llm-3d-scene": "3dllm",
}


@dataclass(frozen=True)
class RoMaBundle:
    captions: List[str]
    scene_ids: List[str]
    scene_indices: List[int]
    feature_scene_ids: List[str]
    features: np.ndarray


def canonical_dataset_name(data_name: str) -> str:
    lowered = str(data_name).strip().lower()
    return DATASET_ALIASES.get(lowered, lowered)


def is_roma_dataset(data_name: str) -> bool:
    return canonical_dataset_name(data_name) in ROMA_DATASETS


def _split_tag(data_split: str) -> str:
    return "train" if str(data_split).lower() == "train" else "val"


def _json(path: Path) -> List[Dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list: {path}")
    return data


def _jsonl(path: Path) -> List[Dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ordered_unique(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(values))


def _rows(root: Path, dataset: str, split: str) -> tuple[List[str], List[str]]:
    if dataset == "scanrefer":
        rows = _jsonl(root / f"scanrefer_{split}.jsonl")
        return [str(row["description"]) for row in rows], [str(row["scene_id"]) for row in rows]
    if dataset == "nr3d":
        rows = _jsonl(root / f"nr3d_{split}.jsonl")
        return [str(row["description"]) for row in rows], [str(row.get("scan_id") or row["scene_id"]) for row in rows]
    if dataset == "scenedepict":
        rows = _json(root / f"3D_Text_Retrv_{split}_final_sorted.json")
        return [str(row["description"]) for row in rows], [str(row["scene_id"]) for row in rows]
    if dataset == "3dllm":
        rows = _json(root / f"3d_llm_scene_description_{split}_sorted.json")
        captions = [str(row["answers"][0]) for row in rows if isinstance(row.get("answers"), list) and row["answers"]]
        if len(captions) != len(rows):
            raise ValueError("every 3dllm row must contain a non-empty answers list")
        return captions, [str(row["scene_id"]) for row in rows]
    raise ValueError(f"unsupported RoMa dataset: {dataset}")


def _scanrefer_pool(root: Path) -> tuple[List[str], np.ndarray]:
    scene_ids, arrays = [], []
    for split in ("train", "val"):
        _, rows_scene_ids = _rows(root, "scanrefer", split)
        order = _ordered_unique(rows_scene_ids)
        array = np.load(root / f"pt2vec_200_random_{split}.npy")
        if len(order) != len(array):
            raise ValueError(f"ScanRefer {split} feature scene count mismatch")
        scene_ids.extend(order)
        arrays.append(array)
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("ScanRefer train/val feature scene orders overlap")
    return scene_ids, np.concatenate(arrays, axis=0)


def _scenedepict_pool(root: Path) -> tuple[List[str], np.ndarray]:
    scene_ids, arrays = [], []
    for split in ("train", "val"):
        _, rows_scene_ids = _rows(root, "scenedepict", split)
        order = _ordered_unique(rows_scene_ids)
        array = np.load(root / f"3D_Text_Retrv_grid_{split}.npy")
        if len(order) != len(array):
            raise ValueError(f"SceneDepict {split} feature scene count mismatch")
        scene_ids.extend(order)
        arrays.append(array)
    return scene_ids, np.concatenate(arrays, axis=0)


def load_roma_bundle(data_root: str | Path, data_name: str, data_split: str) -> RoMaBundle:
    root = Path(data_root)
    dataset = canonical_dataset_name(data_name)
    if dataset not in ROMA_DATASETS:
        raise ValueError(f"unsupported RoMa dataset: {dataset}")
    split = _split_tag(data_split)
    captions, scene_ids = _rows(root, dataset, split)
    feature_scene_ids = _ordered_unique(scene_ids)
    if dataset == "nr3d":
        scan_ids, scan_features = _scanrefer_pool(root)
        depict_ids, depict_features = _scenedepict_pool(root)
        features_by_scene = dict(zip(scan_ids, scan_features))
        features_by_scene.update({scene_id: feature for scene_id, feature in zip(depict_ids, depict_features) if scene_id not in features_by_scene})
        missing = sorted(set(feature_scene_ids) - set(features_by_scene))
        if missing:
            raise ValueError(f"Nr3D scenes missing from feature pools: {missing[:5]}")
        features = np.stack([features_by_scene[scene_id] for scene_id in feature_scene_ids])
    else:
        feature_path = {
            "scanrefer": root / f"pt2vec_200_random_{split}.npy",
            "scenedepict": root / f"3D_Text_Retrv_grid_{split}.npy",
            "3dllm": root / f"3d_llm_grid_{split}.npy",
        }[dataset]
        features = np.load(feature_path)
    if features.ndim != 3 or features.shape[1:] != (200, 1024):
        raise ValueError(f"expected RoMa features (N, 200, 1024), found {features.shape}")
    if len(feature_scene_ids) != len(features):
        raise ValueError(f"feature scene count mismatch for {dataset}/{split}")
    index_by_scene = {scene_id: index for index, scene_id in enumerate(feature_scene_ids)}
    return RoMaBundle(captions, scene_ids, [index_by_scene[scene_id] for scene_id in scene_ids], feature_scene_ids, features)


class SceneUniqueBatchSampler(Sampler[List[int]]):
    """Samples captions without repeating a scene inside a training batch."""

    def __init__(self, scene_indices: Sequence[int], batch_size: int, *, shuffle: bool = True, seed: int = 2022):
        self.scene_indices = list(scene_indices)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size <= 0 or self.batch_size > len(set(self.scene_indices)):
            raise ValueError("batch_size must be in [1, unique scene count]")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        grouped: Dict[int, Deque[int]] = defaultdict(deque)
        for index, scene_index in enumerate(self.scene_indices):
            grouped[scene_index].append(index)
        active = list(grouped)
        while active:
            if len(active) < 2:
                break
            if self.shuffle:
                rng.shuffle(active)
            selected = active[: self.batch_size]
            yield [grouped[scene_index].popleft() for scene_index in selected]
            active = [scene_index for scene_index in active if grouped[scene_index]]

    def __len__(self) -> int:
        return math.ceil(len(self.scene_indices) / self.batch_size)


def text_to_scene_metrics(similarities: np.ndarray, caption_scene_indices: Sequence[int]) -> Dict[str, float]:
    """Compute RoMa text-to-scene retrieval metrics from a scene-by-caption matrix."""
    scores = np.asarray(similarities)
    target_indices = np.asarray(caption_scene_indices, dtype=np.int64)
    if scores.ndim != 2 or scores.shape[1] != len(target_indices):
        raise ValueError("similarities must be (scenes, captions) with one target per caption")
    if target_indices.size and (target_indices.min() < 0 or target_indices.max() >= scores.shape[0]):
        raise ValueError("caption scene index is outside the similarity matrix")
    ranks = []
    for caption_index, target_index in enumerate(target_indices):
        ranked = np.argsort(-scores[:, caption_index], kind="stable")
        ranks.append(int(np.flatnonzero(ranked == target_index)[0]) + 1)
    ranks = np.asarray(ranks, dtype=np.float64)
    if not len(ranks):
        return {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "R@30": 0.0, "Rsum": 0.0, "MedR": 0.0, "MeanR": 0.0}
    recalls = {f"R@{k}": float((ranks <= k).mean() * 100.0) for k in (1, 5, 10, 30)}
    return {
        **recalls,
        "Rsum": recalls["R@1"] + recalls["R@5"] + recalls["R@10"],
        "MedR": float(np.median(ranks)),
        "MeanR": float(ranks.mean()),
    }
