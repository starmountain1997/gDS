#!/usr/bin/env python3
"""
Generate dual-node deployment scripts from GitHub config.
"""

import shlex
import socket
from pathlib import Path
from urllib.request import urlopen

import click
import yaml
from loguru import logger

CONFIG_URL_TEMPLATE = "https://raw.githubusercontent.com/starmountain1997/vllm-ascend/{branch}/tests/e2e/nightly/multi_node/config/DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml"


def get_local_ip() -> str:
    """Get the local IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@click.command()
@click.option(
    "--output",
    "-o",
    default="./dual_ds32_w8a8",
    show_default=True,
    help="Output directory",
)
@click.option(
    "--branch", "-b", default="main", show_default=True, help="Git branch name"
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model path (default: read from config)",
)
@click.option(
    "--served-model-name",
    default="dsv3",
    show_default=True,
    help="Served model name for OpenAI API",
)
@click.option(
    "--master-ip",
    default=None,
    help="Master node IP (for node1 script)",
)
def main(
    output: str,
    branch: str,
    model: str | None,
    served_model_name: str,
    master_ip: str | None,
):
    """Fetch DeepSeek-V3 config and generate node0.sh / node1.sh scripts."""
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = fetch_config(branch)
    env_common = config.get("env_common", {})
    deployments = config.get("deployment", [])

    if len(deployments) != 2:
        raise click.ClickException(f"Expected 2 deployments, found {len(deployments)}")

    default_model = extract_model_from_cmd(deployments[0].get("server_cmd", ""))
    model_path = model or default_model

    # 如果 master_ip 未指定，自动获取本机 IP
    if master_ip is None:
        master_ip = get_local_ip()

    node0_script = generate_script(
        deployments[0],
        env_common,
        model_path,
        served_model_name,
        master_ip,
        is_master=True,
    )
    node1_script = generate_script(
        deployments[1],
        env_common,
        model_path,
        served_model_name,
        master_ip,
        is_master=False,
    )

    (output_dir / "node0.sh").write_text(node0_script)
    (output_dir / "node1.sh").write_text(node1_script)

    logger.success(f"Generated: {output_dir / 'node0.sh'}")
    logger.success(f"Generated: {output_dir / 'node1.sh'}")

    click.echo("\nConfiguration:")
    click.echo(f"  Master IP (node0): {master_ip}")
    click.echo(f"  Master IP (node1): {master_ip}")


def fetch_config(branch: str = "main") -> dict:
    """Fetch YAML config from GitHub."""
    url = CONFIG_URL_TEMPLATE.format(branch=branch)
    with urlopen(url, timeout=30) as response:
        content = response.read().decode("utf-8")
        return yaml.safe_load(content)


def extract_model_from_cmd(cmd_block: str) -> str:
    """Extract model path from command (third token: vllm serve <model>)."""
    tokens = shlex.split(cmd_block)
    if len(tokens) >= 3:
        return tokens[2]
    return ""


def generate_script(
    deploy: dict,
    env_common: dict,
    model_path: str,
    served_model_name: str,
    master_ip: str | None,
    is_master: bool,
) -> str:
    """Generate startup script for a single node."""
    cmd_block = deploy.get("server_cmd", "")
    tokens = shlex.split(cmd_block)
    result = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if i == 0:
            # 第一个命令是 vllm，第二个是 serve，合并成 "vllm serve"
            result.append(f"{token} {tokens[i + 1]}")
        elif i == 1:
            # 第二个 token（serve）已被上面处理，跳过
            pass
        elif i == 2:
            # 第三个 token 是 model path，使用用户指定的
            result.append(model_path)
        elif token.startswith("--"):
            # 长参数（以 -- 开头）
            if "=" in token:
                result.append(token)
            else:
                result.append(token)
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    result[-1] += f" {tokens[i + 1]}"
                    i += 1
        elif token.startswith("-"):
            # 短参数如 -v，直接拼接下一个值
            result.append(f"{token} {tokens[i + 1]}")
            i += 1
        else:
            result.append(token)

        i += 1

    # 找到第一个 -- 参数的位置，插入 --served-model-name
    first_flag_idx = next((i for i, r in enumerate(result) if r.startswith("--")), 3)
    result.insert(first_flag_idx, f"--served-model-name {served_model_name}")

    # 如果是 worker 节点且指定了 master_ip，替换 $MASTER_IP
    if not is_master and master_ip:
        result = [r.replace("$MASTER_IP", master_ip) for r in result]

    full_cmd = " \\\n    ".join(result)

    env_lines = ["# ==================== Environment Variables ===================="]
    for key, value in env_common.items():
        value_str = "true" if isinstance(value, bool) else str(value)
        env_lines.append(f'export {key}="{value_str}"')

    node_type = "Master Node" if is_master else "Worker Node"
    ip_hint = (
        "LOCAL_IP needs to be replaced with actual machine IP"
        if is_master
        else "MASTER_IP needs to be replaced with Node 0's IP"
    )

    return f"""#!/bin/bash
# Node {{node_idx}} ({node_type})

{chr(10).join(env_lines)}

# ==================== Node Configuration ====================
# {ip_hint}

# ==================== Startup Command ====================
{full_cmd}
"""


if __name__ == "__main__":
    main()
