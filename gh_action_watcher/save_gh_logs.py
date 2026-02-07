#!/usr/bin/env python3
"""
Save GitHub Actions workflow run logs to files.

Targets: vllm-project/vllm-ascend / Nightly-A3 / multi-node & single-node DeepSeek-V3 tests
"""

import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import click
from loguru import logger

REPO = "vllm-project/vllm-ascend"
WORKFLOW_NAME = "schedule_nightly_test_a3.yaml"
TARGET_JOBS = [
    "multi-node-dpsk3.2-2node",
    "test_deepseek_v3_2_w8a8",
]


@click.command()
@click.option("--runs", "-n", default=4, show_default=True, help="Number of recent runs to check")
def main(runs: int):
    SCRIPT_DIR = Path(__file__).parent.resolve()
    LOGS_DIR = SCRIPT_DIR / "logs"

    logger.info(f"Target: {REPO} / {WORKFLOW_NAME}")
    logger.info(f"Job keywords: {TARGET_JOBS}")
    logger.info(f"Checking last {runs} runs")
    logger.info("-" * 50)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    runs_data = get_recent_runs(runs)
    if not runs_data:
        logger.error("No runs found")
        return

    for run in runs_data:
        run_id = run["databaseId"]
        created_at = run["createdAt"]
        conclusion = run.get("conclusion", "unknown")

        logger.info(
            f"Run #{run['number']} (databaseId: {run_id}, {created_at[:10]}) - {conclusion}"
        )

        jobs = find_target_jobs(run_id)
        if not jobs:
            logger.info("  No target jobs found")
            continue

        commit_sha = run.get("headSha", "")
        run_date = created_at[:10]

        for job in jobs:
            matched_keyword = next(
                (k for k in TARGET_JOBS if k in job.get("name", "")), ""
            )

            logger.info(f"  Found job: {job['name']}")
            logger.info(f"  Job ID: {job['databaseId']}")
            logger.info(f"  Conclusion: {job.get('conclusion', 'N/A')}")

            log_content = get_job_log(run_id, job["databaseId"])
            output_token_throughput = None

            if log_content:
                log_path = save_log(
                    LOGS_DIR, run_date, run_id, job, log_content, matched_keyword, commit_sha
                )
                if log_path is None:
                    logger.info("  Log skipped (already exists)")
                else:
                    logger.success(f"  Log saved: {log_path}")

                output_token_throughput = extract_output_token_throughput(log_content)

            emoji_conclusion = {
                "success": "✅",
                "failure": "❌",
                "cancelled": "⚪",
                "skipped": "🚫",
            }.get(job.get("conclusion", ""), "?")

            append_result_to_csv(
                LOGS_DIR, matched_keyword, emoji_conclusion, commit_sha, created_at, job["databaseId"], output_token_throughput
            )


def get_recent_runs(limit: int = 4) -> list[dict]:
    """Get recent workflow runs for Nightly-A3."""
    result = run_gh(
        [
            "run", "list", "-R", REPO, "-w", WORKFLOW_NAME, "-b", "main",
            "-L", str(limit), "-e", "schedule",
            "--json", "number,databaseId,name,status,conclusion,createdAt,headBranch,headSha",
        ]
    )
    if result.returncode != 0:
        logger.error(f"Failed to list runs: {result.stderr}")
        return []
    return json.loads(result.stdout)


def get_run_jobs(run_id: int) -> list[dict]:
    """Get jobs for a specific run."""
    result = run_gh(["run", "view", "-R", REPO, str(run_id), "--json", "jobs"])
    if result.returncode != 0:
        logger.error(f"Failed to get run jobs: {result.stderr}")
        return []
    return json.loads(result.stdout).get("jobs", [])


def find_target_jobs(run_id: int) -> list[dict]:
    """Find all target jobs in a run."""
    jobs = get_run_jobs(run_id)
    return [j for j in jobs if any(k in j.get("name", "") for k in TARGET_JOBS)]


def get_job_log(run_id: int, job_id: int) -> str:
    """Get log content for a specific job, removing job/step prefix."""
    result = run_gh(["run", "view", "-R", REPO, "--log", "--job", str(job_id), str(run_id)])
    if result.returncode != 0:
        logger.error(f"Failed to get job log: {result.stderr}")
        return ""

    lines = result.stdout.split("\n")
    cleaned_lines = []
    for line in lines:
        first_tab = line.find("\t")
        if first_tab == -1:
            cleaned_lines.append(line)
            continue
        second_tab = line.find("\t", first_tab + 1)
        if second_tab == -1:
            cleaned_lines.append(line)
        else:
            cleaned_lines.append(line[second_tab + 1:])
    return "\n".join(cleaned_lines)


def extract_output_token_throughput(log_content: str) -> float | None:
    """Extract output token throughput from log content."""
    match = re.search(r"Output Token Throughput │ total\s+│ (\d+\.\d+) token/s", log_content)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def save_log(
    LOGS_DIR: Path,
    run_date: str,
    run_id: int,
    job: dict,
    log_content: str,
    keyword: str,
    commit_sha: str = "",
) -> Path | None:
    """Save log content to file."""
    log_dir = LOGS_DIR / run_date
    log_dir.mkdir(parents=True, exist_ok=True)

    short_sha = commit_sha[:7] if commit_sha else str(run_id)
    filename = f"{run_date}_{short_sha}_{keyword}_{job['databaseId']}.log"
    log_path = log_dir / filename

    if log_path.exists():
        return None

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Run ID: {run_id}\n")
        f.write(f"# Commit: {commit_sha}\n")
        f.write(f"# Job: {job['name']}\n")
        f.write(f"# Date: {run_date}\n")
        f.write("=" * 60 + "\n\n")
        f.write(log_content)

    return log_path


def append_result_to_csv(
    LOGS_DIR: Path,
    keyword: str,
    conclusion: str,
    commit_sha: str,
    run_time: str,
    job_id: int,
    output_token_throughput: float | None = None,
):
    """Append monitoring result to CSV file, avoiding duplicates."""
    csv_path = LOGS_DIR / f"{keyword}_results.csv"

    header = [
        "任务运行时间",
        "是否成功",
        "最后commit",
        "job_id",
        "Output Token Throughput",
    ]

    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            file_header = next(reader, None)
            if file_header and file_header[:-1] == header[:-1]:
                existing_job_ids = {row[3] for row in reader if len(row) > 3}
                if str(job_id) in existing_job_ids:
                    logger.info(f"Result for job {job_id} already exists, skipping")
                    return
            else:
                logger.warning(f"CSV header mismatch, rewriting {csv_path}")
                csv_path.unlink()

    new_row = [
        run_time,
        conclusion,
        commit_sha,
        job_id,
        output_token_throughput if output_token_throughput is not None else "",
    ]

    existing_rows = []
    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) < len(header):
                    row.extend([""] * (len(header) - len(row)))
                existing_rows.append(row)
    existing_rows.append(new_row)

    # Sort by run_time
    try:
        existing_rows.sort(
            key=lambda x: datetime.fromisoformat(x[0].replace("Z", "+00:00"))
        )
    except ValueError as e:
        logger.warning(f"Could not sort CSV: {e}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(existing_rows)
    logger.success(f"Result recorded to {csv_path}")


def run_gh(args: list[str]) -> subprocess.CompletedProcess:
    """Run gh CLI command."""
    return subprocess.run(
        ["gh"] + args,
        capture_output=True,
        text=True,
    )


if __name__ == "__main__":
    main()
