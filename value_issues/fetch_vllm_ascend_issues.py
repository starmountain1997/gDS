#!/usr/bin/env python3
"""
获取 vllm-project/vllm-ascend 仓库的 issue 列表并保存为 JSON
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta


def run_gh_command(cmd):
    """执行 gh 命令并返回结果"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def main():
    repo = "vllm-project/vllm-ascend"

    # 动态计算4个月前的日期
    search_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")

    # 获取符合条件的所有 issues
    print(f"正在从 {repo} 获取 {search_date} 之后的 issues...")

    # 先检查总数
    count_cmd = f'gh issue list --repo {repo} --state all --search "created:>={search_date}" --limit 1000 | wc -l'
    count = int(run_gh_command(count_cmd).strip())
    print(f"符合条件的问题总数: {count}")

    # 获取完整列表
    json_cmd = f'gh issue list --repo {repo} --state all --search "created:>={search_date}" --limit 1000 --json number,title,state,createdAt,author,labels'
    json_output = run_gh_command(json_cmd)

    # 解析 JSON
    issues = json.loads(json_output)

    # 保存为 JSON
    json_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "vllm_ascend_issues.json"
    )
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(issues, f, ensure_ascii=False, indent=2)

    print(f"JSON 文件已保存到: {json_file}")
    print(f"共导出 {len(issues)} 条 issues")

    open_count = sum(1 for i in issues if i["state"] == "OPEN")
    closed_count = sum(1 for i in issues if i["state"] == "CLOSED")
    print(f"OPEN: {open_count}, CLOSED: {closed_count}")


if __name__ == "__main__":
    main()
