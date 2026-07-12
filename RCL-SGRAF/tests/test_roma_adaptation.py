from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import EncoderSimilarity
from roma_evaluation import text_to_scene_metrics


def test_sgraf_accepts_roma_200_patch_features():
    model = EncoderSimilarity(1024, 256, "SGR", 3, num_regions=200)

    similarities = model(
        torch.randn(2, 200, 1024),
        torch.randn(2, 7, 1024),
        [7, 7],
    )

    assert similarities.shape == (2, 2)


def test_roma_metrics_include_mrr_and_rsum_through_30():
    similarities = np.array(
        [
            [0.9, 0.1, 0.2],
            [0.1, 0.3, 0.8],
            [0.2, 0.7, 0.6],
        ]
    )

    metrics = text_to_scene_metrics(similarities, [0, 2, 1])

    assert metrics["MRR"] == 100.0
    assert metrics["Rsum"] == metrics["R@1"] + metrics["R@5"] + metrics["R@10"] + metrics["R@30"]


def test_checkpoint_options_are_configured_for_roma_evaluation():
    from eval_roma import configure_roma_options

    checkpoint_options = SimpleNamespace(data_name="f30k_precomp", data_path="old", vocab_path="old")
    arguments = SimpleNamespace(
        data_name="scanrefer",
        data_root="/roma/data",
        vocab_path="/roma/vocab",
        text_enc_type="bert",
        bert_path="/roma/bert",
        batch_size=16,
        workers=0,
    )

    options = configure_roma_options(checkpoint_options, arguments)

    assert options.data_name == "scanrefer"
    assert options.data_path == "/roma/data"
    assert options.data_root == "/roma/data"
    assert options.vocab_path == "/roma/vocab"
    assert options.text_enc_type == "bert"
    assert options.bert_path == "/roma/bert"
    assert options.num_regions == 200
