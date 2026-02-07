#!/usr/bin/env python3
"""
Generate single-node vLLM server startup script from GitHub config.
"""

import ast
from pathlib import Path
from urllib.request import urlopen

import click
from loguru import logger

CONFIG_URL = "https://raw.githubusercontent.com/starmountain1997/vllm-ascend/{branch}/tests/e2e/nightly/single_node/models/test_deepseek_v3_2_w8a8.py"


@click.command()
@click.option(
    "--output",
    "-o",
    default="./test_single_node",
    show_default=True,
    help="Output directory",
)
@click.option(
    "--branch", "-b", default="main", show_default=True, help="Git branch name"
)
@click.option(
    "--model-path",
    default=None,
    help="Model path (default: read from config)",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Server port (default: read from config)",
)
def main(output: str, branch: str, model_path: str | None, port: int | None):
    """Fetch single-node test config and generate vLLM server startup script."""
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = fetch_config(branch)

    default_model = config.get("MODELS", ["vllm-ascend/DeepSeek-V3.2-W8A8"])[0]
    model = model_path or default_model

    tp_size = config.get("TENSOR_PARALLELS", [1])[0]
    dp_size = config.get("DATA_PARALLELS", [1])[0]
    default_port = config.get("PORT", 8087)
    server_port = port if port is not None else default_port

    script = generate_script(
        model,
        config.get("server_args", []),
        tp_size,
        dp_size,
        server_port,
        config.get("env_dict", {}),
    )
    (output_dir / "start_server.sh").write_text(script)

    logger.success(f"Generated: {output_dir / 'start_server.sh'}")

    click.echo("\nUsage:")
    click.echo(f"  cd {output_dir}")
    click.echo("  ./start_server.sh")


def fetch_config(branch: str) -> dict:
    """Fetch Python test config from GitHub."""
    url = CONFIG_URL.format(branch=branch)
    with urlopen(url, timeout=30) as response:
        content = response.read().decode("utf-8")
        return parse_python_config(content)


def parse_python_config(content: str) -> dict:
    """Parse Python config file using AST."""
    tree = ast.parse(content)
    result = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    if isinstance(node.value, ast.List):
                        result[name] = [
                            get_list_element_value(elt) for elt in node.value.elts
                        ]
                    elif isinstance(node.value, ast.Dict):
                        result[name] = {
                            get_value(k): get_value(v)
                            for k, v in zip(node.value.keys, node.value.values)
                        }
                    elif isinstance(node.value, ast.Constant):
                        result[name] = node.value.value
                    elif (
                        isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == "str"
                    ):
                        # Handle str(xxx) calls - extract the variable name
                        arg = node.value.args[0]
                        if isinstance(arg, ast.Name):
                            result[name] = f"STR_{arg.id.upper()}"
                        else:
                            result[name] = get_value(arg)

    return result


def get_list_element_value(node: ast.AST) -> str | int | bool | None:
    """Extract value from a list element AST node, handling str() calls."""
    if isinstance(node, ast.Constant):
        return node.value
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "str"
    ):
        # Handle str(xxx) calls - extract the variable name
        arg = node.args[0]
        if isinstance(arg, ast.Name):
            return f"STR_{arg.id.upper()}"
    elif isinstance(node, ast.Name):
        return node.id
    return None


def get_value(node: ast.AST) -> str | int | bool | None:
    """Extract value from AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        return node.id
    return None


def format_args(args: list, tp_size: int, dp_size: int, port: int) -> str:
    """Format server args, merging flag + value pairs."""
    if not args:
        return ""

    # 替换占位符
    tokens = [
        str(tp_size)
        if a == "STR_TP_SIZE"
        else str(dp_size)
        if a == "STR_DP_SIZE"
        else str(port)
        if a == "STR_PORT"
        else a
        for a in args
    ]
    result = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.startswith("--"):
            if "=" in token:
                result.append(token)
            else:
                result.append(token)
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    result[-1] += f" {tokens[i + 1]}"
                    i += 1
        elif token.startswith("-"):
            result.append(f"{token} {tokens[i + 1]}")
            i += 1
        else:
            result.append(token)

        i += 1

    return " \\\n    ".join(result)


def generate_script(
    model: str, args: list, tp_size: int, dp_size: int, server_port: int, env: dict
) -> str:
    """Generate vLLM server startup script."""
    formatted_args = format_args(args, tp_size, dp_size, server_port)

    env_lines = []
    for key, value in env.items():
        env_lines.append(f'export {key}="{value}"')

    return f"""#!/bin/bash
# vLLM Server Startup Script

{chr(10).join(env_lines)}

# ==================== Server Configuration ====================
# Model: {model}
# Tensor Parallel: {tp_size}
# Data Parallel: {dp_size}
# Port: {server_port}

# ==================== Startup Command ====================
python -m vllm.entrypoints.openai.api_server \\
    {model} \\
    {formatted_args}
"""


if __name__ == "__main__":
    main()
