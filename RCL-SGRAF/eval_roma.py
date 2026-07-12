"""Evaluate an RCL-SGRAF checkpoint with the RoMa text-to-scene protocol."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os

import torch
from transformers import BertTokenizer

from data import get_test_loader
from evaluation import encode_data, shard_attn_scores
from model import SGRAF
from roma_evaluation import text_to_scene_metrics
from vocab import deserialize_vocab


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to model_best checkpoint.")
    parser.add_argument("--data_name", required=True, choices=("scenedepict", "scanrefer", "nr3d", "3dllm"))
    parser.add_argument("--data_root", required=True, help="RoMa data directory.")
    parser.add_argument("--vocab_path", required=True, help="RoMa vocabulary directory.")
    parser.add_argument("--text_enc_type", required=True, choices=("bigru", "bert"))
    parser.add_argument("--bert_path", default="", help="Local bert-base-uncased directory.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def configure_roma_options(checkpoint_options, arguments):
    options = copy.deepcopy(checkpoint_options)
    options.data_name = arguments.data_name
    options.data_path = arguments.data_root
    options.data_root = arguments.data_root
    options.vocab_path = arguments.vocab_path
    options.text_enc_type = arguments.text_enc_type
    options.bert_path = arguments.bert_path
    options.batch_size = arguments.batch_size
    options.workers = arguments.workers
    options.img_dim = 1024
    options.num_regions = 200
    return options


def load_vocabulary(options):
    if options.text_enc_type == "bert":
        if not options.bert_path:
            raise ValueError("--bert_path is required for --text_enc_type bert")
        vocabulary = BertTokenizer.from_pretrained(options.bert_path, local_files_only=True)
        options.vocab_size = len(vocabulary.vocab)
        return vocabulary

    vocabulary_file = "nr3d_vocab.json" if options.data_name == "nr3d" else "my_data_vocab.json"
    vocabulary = deserialize_vocab(os.path.join(options.vocab_path, vocabulary_file))
    vocabulary.add_word("<mask>")
    options.vocab_size = len(vocabulary)
    return vocabulary


def evaluate(options, checkpoint):
    vocabulary = load_vocabulary(options)
    model = SGRAF(options)
    model.load_state_dict(checkpoint["model"])
    loader = get_test_loader("dev", options.data_name, vocabulary, options.batch_size, options.workers, options)

    image_embeddings, caption_embeddings, caption_lengths = encode_data(model, loader, logging.info)
    scene_rows = [loader.dataset.scene_indices.index(scene) for scene in dict.fromkeys(loader.dataset.scene_indices)]
    similarities = shard_attn_scores(
        model,
        image_embeddings[scene_rows],
        caption_embeddings,
        caption_lengths,
        options,
        shard_size=100,
    )
    return text_to_scene_metrics(similarities.T, loader.dataset.scene_indices)


def main():
    arguments = parse_args()
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu")
    options = configure_roma_options(checkpoint["opt"], arguments)
    metrics = evaluate(options, checkpoint)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
