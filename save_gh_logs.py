#!/usr/bin/env python3
"""
Save GitHub Actions workflow run logs to files.

Targets: vllm-project/vllm-ascend / Nightly-A3 / multi-node & single-node DeepSeek-V3 tests
"""

import csv
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
LOGS_DIR = Path("logs")


def get_csv_path(keyword: str) -> Path:
    """Get CSV path for a specific task."""
    return LOGS_DIR / f"{keyword}_results.csv"


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
    result = run_gh(
        [
            "run",
            "list",
            "-R",
            REPO,
            "-w",
            WORKFLOW_ID,
            "-b",
            "main",
            "-L",
            str(limit),
            "-e",
            "schedule",
            "--json",
            "number,databaseId,name,status,conclusion,createdAt,headBranch,headSha",
        ]
    )
    if result.returncode != 0:
        print(f"Failed to list runs: {result.stderr}")
        return []

    return json.loads(result.stdout)


def get_run_jobs(run_id: int) -> list[dict]:
    """Get jobs for a specific run."""
    result = run_gh(
        [
            "run",
            "view",
            "-R",
            REPO,
            str(run_id),
            "--json",
            "jobs",
        ]
    )
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
    """Get log content for a specific job, removing job/step prefix."""
    result = run_gh(
        [
            "run",
            "view",
            "-R",
            REPO,
            "--log",
            "--job",
            str(job_id),
            str(run_id),
        ]
    )
    if result.returncode != 0:
        print(f"Failed to get job log: {result.stderr}")
        return ""

    # Remove job name and step prefix from each line
    # Format: "{job}\t{step}\t{timestamp}\t{content}"
    lines = result.stdout.split("\n")
    cleaned_lines = []
    for line in lines:
        # Find first tab, then second tab, keep from third tab onwards
        first_tab = line.find("\t")
        if first_tab == -1:
            cleaned_lines.append(line)
        else:
            second_tab = line.find("\t", first_tab + 1)
            if second_tab == -1:
                cleaned_lines.append(line)
            else:
                # Keep from second tab (timestamp) onwards
                cleaned_lines.append(line[second_tab + 1 :])

    return "\n".join(cleaned_lines)


def append_result_to_csv(
    keyword: str, conclusion: str, commit_sha: str, run_time: str, job_id: int
):
    """Append monitoring result to CSV file, avoiding duplicates."""
    csv_path = get_csv_path(keyword)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["任务运行时间", "是否成功", "最后commit", "job_id"]
    existing_job_ids = set()
    rows_to_write = []

    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            file_header = next(reader, None)  # Read header
            if file_header == header:
                for row in reader:
                    if len(row) > 3:  # Ensure job_id column exists
                        existing_job_ids.add(row[3])
                        rows_to_write.append(row)
            else:
                # If header doesn't match, re-write the file. This handles cases where the script
                # or CSV format might have changed.
                print(
                    f"Warning: CSV header for {csv_path} does not match expected. Rewriting file."
                )

    if str(job_id) in existing_job_ids:
        print(f"Result for job {job_id} already exists in {csv_path}, skipping.")
        return

    rows_to_write.append([run_time, conclusion, commit_sha, job_id])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)  # Always write header first
        writer.writerows(rows_to_write)
    print(f"Result recorded to {csv_path}")


def save_log(
    run_date: str,
    run_id: int,
    job: dict,
    log_content: str,
    keyword: str,
    job_id: int,
    commit_sha: str = "",
):
    """Save log content to file."""
    log_dir = Path("logs") / run_date
    log_dir.mkdir(parents=True, exist_ok=True)

    # Filename: {date}_{commit_sha}_{keyword}_{job_id}.log
    short_sha = commit_sha[:7] if commit_sha else str(run_id)
    filename = f"{run_date}_{short_sha}_{keyword}_{job_id}.log"
    log_path = log_dir / filename

    if log_path.exists():
        print(f"Log already exists, skipping: {log_path}")
        return None

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

    # Ensure logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Get recent runs
    runs = get_recent_runs(limit=10)
    if not runs:
        print("No runs found")
        return

    # Find runs with any target job
    for run in runs:
        run_id = run["databaseId"]
        created_at = run["createdAt"]
        conclusion = run.get("conclusion", "unknown")

        print(
            f"Checking run #{run['number']} (databaseId: {run_id}, {created_at[:10]}) - {conclusion}"
        )

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
            print(f"  Conclusion: {job.get('conclusion', 'N/A')}")

            log_content = get_job_log(run_id, job["databaseId"])
            if log_content:
                log_path = save_log(
                    run_date,
                    run_id,
                    job,
                    log_content,
                    matched_keyword,
                    job["databaseId"],
                    commit_sha,
                )
                if log_path is None:
                    print(f"  Log skipped (already exists)")
                else:
                    print(f"  Log saved: {log_path}")

            # 记录 job 的实际运行状态
            emoji_status = {
                "success": "✅",
                "failure": "❌",
                "cancelled": "⚪",
                "skipped": "🚫",
            }
            emoji_conclusion = emoji_status.get(job.get("conclusion", ""), "?")
            append_result_to_csv(
                matched_keyword,
                emoji_conclusion,
                commit_sha,
                created_at,
                job["databaseId"],
            )


if __name__ == "__main__":
    main()
