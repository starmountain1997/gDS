#!/usr/bin/env python3
"""
从 GitHub 拉取 DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml 配置并生成双机部署脚本
"""

from pathlib import Path
from urllib.request import urlopen

import yaml

CONFIG_URL_TEMPLATE = "https://raw.githubusercontent.com/starmountain1997/vllm-ascend/{branch}/tests/e2e/nightly/multi_node/config/DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml"


def fetch_config(branch: str = "main") -> dict:
    """从 GitHub 拉取 YAML 配置"""
    url = CONFIG_URL_TEMPLATE.format(branch=branch)
    with urlopen(url, timeout=30) as response:
        content = response.read().decode("utf-8")
        return yaml.safe_load(content)


def generate_node_scripts(output_dir: str = "./dual_ds32_w8a8", branch: str = "main"):
    """拉取配置并生成 node0.sh 和 node1.sh"""
    config = fetch_config(branch)
    env_common = config.get("env_common", {})
    deployments = config.get("deployment", [])

    if len(deployments) != 2:
        raise ValueError(f"期望2个 deployment 配置，实际找到 {len(deployments)} 个")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成 Node 0 脚本
    node0_script = generate_script(deployments[0], env_common, is_master=True)
    (output_dir / "node0.sh").write_text(node0_script)

    # 生成 Node 1 脚本
    node1_script = generate_script(deployments[1], env_common, is_master=False)
    (output_dir / "node1.sh").write_text(node1_script)

    print(f"✓ 已生成 node0.sh")
    print(f"✓ 已生成 node1.sh")
    print("\n请配置以下变量后执行:")
    print("  Node 0: export LOCAL_IP=<本机IP>")
    print("  Node 1: export MASTER_IP=<Node0的IP>")


def generate_script(deploy: dict, env_common: dict, is_master: bool) -> str:
    """生成单个节点的启动脚本"""
    cmd_block = deploy.get("server_cmd", "")
    cmd_lines = [line.strip() for line in cmd_block.strip().split("\n") if line.strip()]
    full_cmd = " \\\n    ".join(cmd_lines)

    env_lines = ["# ==================== 环境变量 ===================="]
    for key, value in env_common.items():
        if isinstance(value, bool):
            value_str = "true" if value else "false"
        else:
            value_str = str(value)
        env_lines.append(f'export {key}="{value_str}"')

    node_type = "主节点" if is_master else "从节点"
    ip_hint = (
        "LOCAL_IP 需要替换为本机实际 IP"
        if is_master
        else "MASTER_IP 需要替换为 Node 0 的 IP"
    )

    return f"""#!/bin/bash
# Node {{node_idx}} ({node_type})

{chr(10).join(env_lines)}

# ==================== 节点配置 ====================
# {ip_hint}

# ==================== 启动命令 ====================
{full_cmd}
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成双机部署脚本")
    parser.add_argument("-o", "--output", default="./dual_ds32_w8a8", help="输出目录")
    parser.add_argument(
        "-b", "--branch", default="main", help="Git 分支名 (默认: main)"
    )

    args = parser.parse_args()
    generate_node_scripts(args.output, args.branch)
