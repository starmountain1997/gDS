import json
import os
import subprocess

from modelscope import snapshot_download
from transformers import AutoTokenizer


def unzip_gsm8k(zip_path="./gsm8k.zip", output_dir="."):
    """解压 gsm8k.zip 文件"""
    if not os.path.exists(zip_path):
        print(f"{zip_path} 不存在...")
        return False
    print(f"正在解压 {zip_path}...")
    subprocess.run(["unzip", "-o", zip_path, "-d", output_dir], check=True)
    print("解压完成")
    return True


def download_tokenizer_only(model_id, cache_dir="./tokenizer_cache"):
    """只从 modelscope 下载 tokenizer 文件，不下载模型权重"""
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
        # 只下载 tokenizer 相关文件
        allow_patterns=tokenizer_files,
    )
    return model_path


def create_data(input_len, batch_size, model_path=None, save_path="."):
    # 如果没有指定 model_path，自动从 modelscope 下载 tokenizer
    if model_path is None:
        print("正在从 modelscope 下载 DeepSeek-V3 的 tokenizer...")
        model_path = download_tokenizer_only("deepseek-ai/DeepSeek-V3")
        print(f"Tokenizer 已下载到: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if os.path.exists(f"GSM8K-in{input_len}-bs{batch_size}.jsonl"):
        print("dataset already exists...")
        exit(0)

    gsm8k_file = "./gsm8k/train.jsonl"
    if not os.path.exists(gsm8k_file):
        print("gsm8k dataset not exists...")
        unzip_gsm8k()
        if not os.path.exists(gsm8k_file):
            print(f"解压后仍未找到 {gsm8k_file}")
            exit(1)

    dataset = []
    with open(gsm8k_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data["question"])

    dataset_2k = []
    for sentence in dataset:
        words = tokenizer.tokenize(sentence)
        if len(words) == 0:
            continue
        len_num = len(words)

        if len_num < input_len:
            # token数量不足时，重复文本直到达到input_len
            multiplier = (input_len) // len_num + 1
            repeated_len = words * multiplier
            words = repeated_len[:input_len]
        else:
            # token数量足够时，截断到input_len
            words = words[:input_len]

        decoded_text = tokenizer.convert_tokens_to_string(words)
        dataset_2k.append(decoded_text)

    batch_num = len(dataset_2k) // batch_size
    if batch_num == 0:
        multiplier = (batch_size // len(dataset_2k)) + 1

        repeated_batch = dataset_2k * multiplier
        dataset_2k = repeated_batch[:batch_size]
    else:
        dataset_2k = dataset_2k[:batch_size]

    json_str = json.dumps(dataset_2k, ensure_ascii=False, indent=4)

    with open(
        os.path.join(save_path, f"GSM8K-{input_len}-bs{batch_size}.jsonl"),
        "w",
        encoding="utf-8",
    ) as f:
        for i in range(len(dataset_2k)):
            f.write(
                json.dumps(
                    {"question": dataset_2k[i], "answer": "none"}, ensure_ascii=False
                )
            )
            f.write("\n")


if __name__ == "__main__":
    create_data(input_len=64000, batch_size=1000)
