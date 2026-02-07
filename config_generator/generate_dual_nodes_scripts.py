#!/usr/bin/env python3
"""
Generate dual-node deployment scripts from GitHub config.
"""

from pathlib import Path
from urllib.request import urlopen

import click
import yaml
from loguru import logger

CONFIG_URL_TEMPLATE = "https://raw.githubusercontent.com/starmountain1997/vllm-ascend/{branch}/tests/e2e/nightly/multi_node/config/DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml"


@click.command()
@click.option("--output", "-o", default="./dual_ds32_w8a8", show_default=True, help="Output directory")
@click.option("--branch", "-b", default="main", show_default=True, help="Git branch name")
def main(output: str, branch: str):
    """Fetch DeepSeek-V3 config and generate node0.sh / node1.sh scripts."""
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = fetch_config(branch)
    env_common = config.get("env_common", {})
    deployments = config.get("deployment", [])

    if len(deployments) != 2:
        raise click.ClickException(f"Expected 2 deployments, found {len(deployments)}")

    node0_script = generate_script(deployments[0], env_common, is_master=True)
    node1_script = generate_script(deployments[1], env_common, is_master=False)

    (output_dir / "node0.sh").write_text(node0_script)
    (output_dir / "node1.sh").write_text(node1_script)

    logger.success(f"Generated: {output_dir / 'node0.sh'}")
    logger.success(f"Generated: {output_dir / 'node1.sh'}")

    click.echo("\nConfigure before execution:")
    click.echo("  Node 0: export LOCAL_IP=<this_machine_ip>")
    click.echo("  Node 1: export MASTER_IP=<node0_ip>")


def fetch_config(branch: str = "main") -> dict:
    """Fetch YAML config from GitHub."""
    url = CONFIG_URL_TEMPLATE.format(branch=branch)
    with urlopen(url, timeout=30) as response:
        content = response.read().decode("utf-8")
        return yaml.safe_load(content)


def generate_script(deploy: dict, env_common: dict, is_master: bool) -> str:
    """Generate startup script for a single node."""
    cmd_block = deploy.get("server_cmd", "")
    cmd_lines = [line.strip() for line in cmd_block.strip().split("\n") if line.strip()]
    full_cmd = " \\\n    ".join(cmd_lines)

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
