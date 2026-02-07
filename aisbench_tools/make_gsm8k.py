#!/usr/bin/env python3
"""Generate GSM8K dataset for benchmarking."""

import json
import os
import subprocess
from pathlib import Path

import click
from loguru import logger
from modelscope import snapshot_download
from transformers import AutoTokenizer


@click.command()
@click.option("--input-len", default=64000, show_default=True, help="Input token length")
@click.option("--batch-size", default=2800, show_default=True, help="Batch size")
@click.option("--model-id", default="deepseek-ai/DeepSeek-V3", help="Model ID from modelscope")
@click.option("--cache-dir", default="./tokenizer_cache", help="Tokenizer cache directory")
@click.option("--zip-path", default="./gsm8k.zip", help="Path to GSM8K zip file")
@click.option("--gsm8k-dir", default="./gsm8k", help="GSM8K extracted directory")
def main(
    input_len: int,
    batch_size: int,
    model_id: str,
    cache_dir: str,
    zip_path: str,
    gsm8k_dir: str,
):
    """Generate GSM8K dataset with specified input length and batch size."""
    cache_dir = Path(cache_dir)
    zip_path = Path(zip_path)
    gsm8k_file = Path(gsm8k_dir) / "train.jsonl"

    output_file = Path(f"GSM8K-in{input_len}-bs{batch_size}.jsonl")

    if output_file.exists():
        logger.info(f"Dataset already exists: {output_file}")
        return

    tokenizer_path = download_tokenizer_only(model_id, str(cache_dir))
    logger.success(f"Tokenizer downloaded to: {tokenizer_path}")

    if not gsm8k_file.exists():
        if not zip_path.exists():
            logger.error(f"{zip_path} not found")
            return
        logger.info(f"Unzipping {zip_path}...")
        subprocess.run(["unzip", "-o", str(zip_path), "-d", str(zip_path.parent)], check=True)

    if not gsm8k_file.exists():
        logger.error(f"Still not found after unzip: {gsm8k_file}")
        return

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    logger.info(f"Loading GSM8K from {gsm8k_file}...")

    dataset = []
    with open(gsm8k_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data["question"])

    logger.info(f"Processing {len(dataset)} questions...")

    dataset_2k = []
    for sentence in dataset:
        words = tokenizer.tokenize(sentence)
        if not words:
            continue
        len_num = len(words)

        if len_num < input_len:
            multiplier = input_len // len_num + 1
            words = (words * multiplier)[:input_len]
        else:
            words = words[:input_len]

        decoded_text = tokenizer.convert_tokens_to_string(words)
        dataset_2k.append(decoded_text)

    batch_num = len(dataset_2k) // batch_size
    if batch_num == 0:
        multiplier = batch_size // len(dataset_2k) + 1
        dataset_2k = (dataset_2k * multiplier)[:batch_size]
    else:
        dataset_2k = dataset_2k[:batch_size]

    logger.info(f"Writing {len(dataset_2k)} samples to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset_2k:
            f.write(json.dumps({"question": item, "answer": "none"}, ensure_ascii=False) + "\n")

    logger.success(f"Done: {output_file}")


def download_tokenizer_only(model_id: str, cache_dir: str) -> str:
    """Download tokenizer files only from modelscope."""
    tokenizer_files = [
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "chat_template.json",
        "config.json",
    ]

    model_path = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        ignore_patterns=["*.bin", "*.safetensors", "*.pth", "*.model", "*.gguf"],
        allow_patterns=tokenizer_files,
    )
    return model_path


if __name__ == "__main__":
    main()
