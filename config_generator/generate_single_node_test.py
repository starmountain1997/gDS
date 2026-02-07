#!/usr/bin/env python3
"""
Generate single-node vLLM server startup script from GitHub config.
"""

import ast
import shlex
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
@click.option(
    "--served-model-name",
    default="dsv3",
    show_default=True,
    help="Served model name for OpenAI API",
)
def main(
    output: str,
    branch: str,
    model_path: str | None,
    port: int | None,
    served_model_name: str,
):
    """Fetch single-node test config and generate vLLM server startup script."""
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = fetch_config(branch)

    default_model = config.get("MODELS", ["vllm-ascend/DeepSeek-V3.2-W8A8"])[0]
    model = model_path or default_model

    tp_size = config.get("TENSOR_PARALLELS", [1])[0]
    dp_size = config.get("DATA_PARALLELS", [1])[0]
    server_port = port if port is not None else config.get("PORT", 8087)

    script = generate_script(
        model,
        served_model_name,
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


def _get_ast_node_value(node: ast.AST) -> object | None:
    """Recursively extract value from an AST node, handling various types and str() calls."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.List):
        return [_get_ast_node_value(elt) for elt in node.elts]
    elif isinstance(node, ast.Dict):
        return {
            _get_ast_node_value(k): _get_ast_node_value(v)
            for k, v in zip(node.keys, node.values)
        }
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "str"
    ):
        # Handle str(xxx) calls - extract the variable name
        arg = node.args[0]
        if isinstance(arg, ast.Name):
            return f"STR_{arg.id.upper()}"
        else:
            return _get_ast_node_value(arg)
    return None


def parse_python_config(content: str) -> dict:
    """Parse Python config file using AST."""
    tree = ast.parse(content)
    result = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    result[name] = _get_ast_node_value(node.value)
    return result


def format_args(args: list, tp_size: int, dp_size: int, port: int) -> str:
    """Format server args, merging flag + value pairs and quoting special chars."""
    if not args:
        return ""

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

    formatted_parts = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.startswith("--"):
            if "=" in token:
                # Handle --flag=value format
                flag, value = token.split("=", 1)
                formatted_parts.append(f"{flag}={shlex.quote(value)}")
            else:
                # Handle --flag value or --flag (boolean)
                # Check if the next token is a value (not starting with '-')
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    value = tokens[i + 1]
                    formatted_parts.append(f"{token} {shlex.quote(value)}")
                    i += 1  # Consume the value token
                else:
                    # Boolean flag, no separate value
                    formatted_parts.append(token)
        else:
            # This case means a token that does not start with '--' is encountered.
            # This should ideally not happen if `server_args` is well-formed.
            # For robustness, we quote it.
            formatted_parts.append(shlex.quote(token))
        i += 1

    return " \\\n    ".join(formatted_parts)


def generate_script(
    model: str,
    served_model_name: str,
    args: list,
    tp_size: int,
    dp_size: int,
    server_port: int,
    env: dict,
) -> str:
    """Generate vLLM server startup script."""
    formatted_args = format_args(args, tp_size, dp_size, server_port)

    env_lines = [f'export {key}="{value}"' for key, value in env.items()]

    return f"""#!/bin/bash
# vLLM Server Startup Script

{chr(10).join(env_lines)}

# ==================== Server Configuration ====================
# Model: {model}
# Served Model Name: {served_model_name}
# Tensor Parallel: {tp_size}
# Data Parallel: {dp_size}
# Port: {server_port}

# ==================== Startup Command ====================
vllm serve {model} \\
    --served-model-name {served_model_name} \\
    {formatted_args}
"""


if __name__ == "__main__":
    main()
