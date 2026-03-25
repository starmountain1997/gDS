#!/usr/bin/env python3
"""
获取 GitCode ascend-tribe 组织下所有模型的下载量并保存为 CSV
"""

import argparse
import csv
import os
import sys

import requests


def get_org_repos(org: str, token: str) -> list:
    """获取组织下的模型仓库"""
    url = f"https://gitcode.com/api/v5/orgs/{org}/repos"
    headers = {"private-token": token} if token else {}
    repos = []
    page = 1
    per_page = 100

    while True:
        params = {"page": page, "per_page": per_page, "cat": "model"}
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        repos.extend(data)
        page += 1

    return repos


def get_download_statistics(owner: str, repo: str, token: str) -> dict:
    """获取仓库的下载统计"""
    url = f"https://gitcode.com/api/v5/repos/{owner}/{repo}/download_statistics"
    headers = {"private-token": token} if token else {}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  Warning: 获取 {owner}/{repo} 下载统计失败: {e}", file=sys.stderr)
        return {}


def main():
    parser = argparse.ArgumentParser(description="获取 GitCode 组织模型的下载量统计")
    parser.add_argument(
        "--token", "-t", help="GitCode API Token (或设置环境变量 GITCODE_TOKEN)"
    )
    parser.add_argument("--org", "-o", default="ascend-tribe", help="组织名称")
    parser.add_argument(
        "--output", "-f", default="model_downloads.csv", help="输出CSV文件名"
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("GITCODE_TOKEN")
    if not token:
        print(
            "Error: 请通过 --token 参数或 GITCODE_TOKEN 环境变量提供 API Token",
            file=sys.stderr,
        )
        sys.exit(1)

    org = args.org

    print(f"正在获取 {org} 组织的模型仓库...")
    repos = get_org_repos(org, token)
    print(f"共找到 {len(repos)} 个模型仓库")

    results = []
    for i, repo in enumerate(repos):
        repo_name = repo.get("path") or repo.get("name", "")
        owner = (
            repo.get("namespace", {}).get("path")
            or repo.get("full_name", "").split("/")[0]
        )
        full_name = repo.get("full_name")
        print(f"[{i + 1}/{len(repos)}] 正在处理: {full_name}")

        stats = get_download_statistics(owner, repo_name, token)
        # 不同API返回的字段可能不同，尝试多个可能的字段名
        downloads = (
            stats.get("download_statistics_history_total")
            or stats.get("downloads")
            or stats.get("download_count")
            or stats.get("total_downloads")
            or stats.get("count")
            or 0
        )

        results.append(
            {
                "repo_name": full_name,
                "downloads": downloads,
                "description": repo.get("description", ""),
                "updated_at": repo.get("updated_at", ""),
            }
        )

    # 按下载量降序排序
    results.sort(key=lambda x: x["downloads"], reverse=True)

    # 保存为 CSV
    csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["repo_name", "downloads", "description", "updated_at"]
        )
        writer.writeheader()
        writer.writerows(results)

    total = sum(r["downloads"] for r in results)
    print(f"\nCSV 文件已保存到: {csv_file}")
    print(f"共导出 {len(results)} 个仓库")
    print(f"总下载量: {total:,}")


if __name__ == "__main__":
    main()
