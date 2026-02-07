# gDS - DeepSeek-V3.2-W8A8 A3 双节点部署工具

## 数据集

### GSM8K 数据集下载

http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip

## 双节点部署脚本生成器

用于从 vllm-ascend 仓库自动生成 DeepSeek-V3.2-W8A8 模型的双节点部署脚本。

### 使用方法

```bash
# 生成部署脚本(默认输出到 ./dual_ds32_w8a8/ 目录)
python3 generate_dual_nodes_scripts.py

# 指定输出目录
python3 generate_dual_nodes_scripts.py -o /path/to/output

# 指定 Git 分支
python3 generate_dual_nodes_scripts.py -b dev
```

### 生成文件说明

- `node0.sh` - 主节点(Node 0)启动脚本
  - 需设置环境变量 `LOCAL_IP` 为本机实际 IP
- `node1.sh` - 从节点(Node 1)启动脚本
  - 需设置环境变量 `MASTER_IP` 为 Node 0 的 IP

### 部署步骤

1. 在两台机器上生成脚本
2. Node 0 执行: `export LOCAL_IP=<本机IP> && bash dual_ds32_w8a8/node0.sh`
3. Node 1 执行: `export MASTER_IP=<Node0的IP> && bash dual_ds32_w8a8/node1.sh`
