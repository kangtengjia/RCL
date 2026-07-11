from __future__ import annotations

import statistics
from typing import Dict, Sequence

import numpy as np


DEFAULT_KS = (1, 5, 10, 30)


def text_to_scene_metrics(similarities: np.ndarray, caption_scene_indices: Sequence[int], *, ks: Sequence[int] = DEFAULT_KS) -> Dict[str, float]:
    scores = np.asarray(similarities)
    if scores.ndim != 2 or scores.shape[0] != len(caption_scene_indices):
        raise ValueError("similarities must have one row per caption")
    ranks = []
    for row, target in enumerate(caption_scene_indices):
        if not 0 <= int(target) < scores.shape[1]:
            raise ValueError(f"scene index {target} is outside candidates")
        order = np.argsort(scores[row])[::-1]
        ranks.append(int(np.where(order == int(target))[0][0]) + 1)
    metrics: Dict[str, float] = {"queries": len(ranks)}
    for cutoff in ks:
        metrics[f"R@{cutoff}"] = 100.0 * sum(rank <= cutoff for rank in ranks) / max(len(ranks), 1)
    metrics["Rsum"] = sum(metrics[f"R@{cutoff}"] for cutoff in ks)
    metrics["MedR"] = float(statistics.median(ranks)) if ranks else 0.0
    metrics["MeanR"] = float(np.mean(ranks)) if ranks else 0.0
    return metrics
