# gDS Config Generator Tools

Tools for generating vLLM deployment and test scripts from GitHub configs.

## Installation

```bash
pip install -e .
```

## Commands

### gen-single-node

Generate single-node vLLM server startup script from GitHub test config.

```bash
gen-single-node -o ./test_single_node
```

Options:
- `-o, --output`: Output directory (default: `./test_single_node`)
- `-b, --branch`: Git branch name (default: `main`)
- `--model-path`: Model path (default: read from config)
- `--port`: Server port (default: read from config)
- `--served-model-name`: Served model name for OpenAI API (default: `dsv3`)

### gen-dual-nodes

Generate dual-node deployment scripts from GitHub config.

```bash
gen-dual-nodes -o ./dual_ds32_w8a8
```

Options:
- `-o, --output`: Output directory (default: `./dual_ds32_w8a8`)
- `-b, --branch`: Git branch name (default: `main`)
- `-m, --model`: Model path (default: read from config)
- `--served-model-name`: Served model name for OpenAI API (default: `dsv3`)
- `--master-ip`: Master node IP (auto-detected if not specified)

## Examples

### Single Node

```bash
# Default settings
gen-single-node -o /tmp/test

# Custom model and port
gen-single-node -o /tmp/test --model-path /path/to/model --port 9000
```

### Dual Nodes

```bash
# Auto-detect local IP for master
gen-dual-nodes -o /tmp/dual

# Specify master IP manually
gen-dual-nodes -o /tmp/dual --master-ip 192.168.1.100

# All custom options
gen-dual-nodes -o /tmp/dual --master-ip 192.168.1.100 --model /path/to/model
```

## Output Files

### Single Node

```
./test_single_node/
└── start_server.sh   # vLLM server startup script
```

### Dual Nodes

```
./dual_ds32_w8a8/
├── node0.sh  # Master node startup script
└── node1.sh  # Worker node startup script
```

## Usage

### Single Node

```bash
# Terminal 1: Start server
cd /tmp/test_single_node
./start_server.sh
```

### Dual Nodes

```bash
# Terminal 1 (Node 0): Start master
cd /tmp/dual_ds32_w8a8
./node0.sh
# Configure: export LOCAL_IP=<this_machine_ip>

# Terminal 2 (Node 1): Start worker
cd /tmp/dual_ds32_w8a8
./node1.sh
# Configure: export MASTER_IP=<node0_ip>
```
