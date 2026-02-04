#!/usr/bin/env python3
"""
Save GitHub Actions workflow run logs to files.

Targets: vllm-project/vllm-ascend / Nightly-A3 / multi-node & single-node DeepSeek-V3 tests
"""

import json
import subprocess
from pathlib import Path


REPO = "vllm-project/vllm-ascend"
WORKFLOW_NAME = "Nightly-A3"
WORKFLOW_ID = "215695701"  # Nightly-A3 workflow ID
TARGET_JOBS = [
    "multi-node-dpsk3.2-2node",  # DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml
    "test_deepseek_v3_2_w8a8",  # single-node test
]


def run_gh(args: list[str]) -> subprocess.CompletedProcess:
    """Run gh CLI command."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True,
        text=True,
    )
    return result


def get_recent_runs(limit: int = 10) -> list[dict]:
    """Get recent workflow runs for Nightly-A3."""
    result = run_gh([
        "run", "list",
        "-R", REPO,
        "-w", WORKFLOW_ID,
        "-L", str(limit),
        "--json", "number,databaseId,name,status,conclusion,createdAt,headBranch,headSha",
    ])
    if result.returncode != 0:
        print(f"Failed to list runs: {result.stderr}")
        return []

    return json.loads(result.stdout)


def get_run_jobs(run_id: int) -> list[dict]:
    """Get jobs for a specific run."""
    result = run_gh([
        "run", "view",
        "-R", REPO,
        str(run_id),
        "--json", "jobs",
    ])
    if result.returncode != 0:
        print(f"Failed to get run jobs: {result.stderr}")
        return []

    data = json.loads(result.stdout)
    return data.get("jobs", [])


def find_target_jobs(run_id: int) -> list[dict]:
    """Find all target jobs in a run."""
    jobs = get_run_jobs(run_id)
    found = []
    for job in jobs:
        for keyword in TARGET_JOBS:
            if keyword in job.get("name", ""):
                found.append(job)
                break
    return found


def get_job_log(run_id: int, job_id: int) -> str:
    """Get log content for a specific job."""
    result = run_gh([
        "run", "view",
        "-R", REPO,
        "--log",
        "--job", str(job_id),
        str(run_id),
    ])
    if result.returncode != 0:
        print(f"Failed to get job log: {result.stderr}")
        return ""

    return result.stdout


def save_log(run_date: str, run_id: int, job: dict, log_content: str, keyword: str, commit_sha: str = ""):
    """Save log content to file."""
    log_dir = Path("logs") / run_date
    log_dir.mkdir(parents=True, exist_ok=True)

    # Filename: {date}_{commit_sha}_{keyword}.log
    short_sha = commit_sha[:7] if commit_sha else str(run_id)
    filename = f"{run_date}_{short_sha}_{keyword}.log"
    log_path = log_dir / filename

    # Write metadata header
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Run ID: {run_id}\n")
        f.write(f"# Commit: {commit_sha}\n")
        f.write(f"# Job: {job['name']}\n")
        f.write(f"# Date: {run_date}\n")
        f.write("=" * 60 + "\n\n")
        f.write(log_content)

    print(f"Saved: {log_path}")
    return log_path


def main():
    print(f"Target: {REPO} / {WORKFLOW_NAME}")
    print(f"Job keywords: {TARGET_JOBS}")
    print("-" * 50)

    # Get recent runs
    runs = get_recent_runs(limit=10)
    if not runs:
        print("No runs found")
        return

    saved_count = 0

    # Find runs with any target job
    for run in runs:
        run_id = run["databaseId"]
        created_at = run["createdAt"]
        conclusion = run.get("conclusion", "unknown")

        print(f"Checking run #{run['number']} (databaseId: {run_id}, {created_at[:10]}) - {conclusion}")

        jobs = find_target_jobs(run_id)
        if not jobs:
            print(f"  No target jobs found")
            continue

        commit_sha = run.get("headSha", "")
        run_date = created_at[:10]  # YYYY-MM-DD

        for job in jobs:
            # Find which keyword matched this job
            matched_keyword = ""
            for keyword in TARGET_JOBS:
                if keyword in job.get("name", ""):
                    matched_keyword = keyword
                    break

            print(f"  Found job: {job['name']}")
            print(f"  Job ID: {job['databaseId']}")

            log_content = get_job_log(run_id, job["databaseId"])
            if log_content:
                save_log(run_date, run_id, job, log_content, matched_keyword, commit_sha)
                saved_count += 1

        if saved_count >= len(TARGET_JOBS):
            print("\nAll target jobs saved")
            break

    else:
        print("No runs with target jobs found")


if __name__ == "__main__":
    main()
