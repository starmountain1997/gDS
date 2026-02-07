#!/usr/bin/env python3
"""
Runs save_gh_logs.py, then commits and pushes changes to git.
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


def run_command(
    command: list[str], cwd: Path | None = None
) -> subprocess.CompletedProcess:
    """Helper to run general shell commands."""
    logger.info(f"Running command: {' '.join(command)}")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,  # We'll check returncode manually for specific git commands
    )
    if result.stdout:
        logger.debug(f"Stdout:\n{result.stdout.strip()}")
    if result.stderr:
        logger.error(f"Stderr:\n{result.stderr.strip()}")
    return result


def _run_gh_cli(args: list[str]) -> subprocess.CompletedProcess:
    """Run gh CLI command."""
    return run_command(
        ["gh"] + args,
    )


def get_recent_runs(limit: int = 4) -> list[dict]:
    """Get recent workflow runs for Nightly-A3."""
    result = _run_gh_cli(
        [
            "run",
            "list",
            "-R",
            REPO,
            "-w",
            WORKFLOW_NAME,
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
        logger.error(f"Failed to list runs: {result.stderr}")
        return []
    return json.loads(result.stdout)


def get_run_jobs(run_id: int) -> list[dict]:
    """Get jobs for a specific run."""
    result = _run_gh_cli(["run", "view", "-R", REPO, str(run_id), "--json", "jobs"])
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
    result = _run_gh_cli(
        ["run", "view", "-R", REPO, "--log", "--job", str(job_id), str(run_id)]
    )
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
            cleaned_lines.append(line[second_tab + 1 :])
    return "\n".join(cleaned_lines)


def extract_output_token_throughput(log_content: str) -> float | None:
    """Extract output token throughput from log content."""
    match = re.search(
        r"Output Token Throughput │ total\s+│ (\d+\.\d+) token/s", log_content
    )
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
    """Append monitoring result to CSV file, avoiding duplicates, and ensuring sorted order."""
    csv_path = LOGS_DIR / f"{keyword}_results.csv"

    header = [
        "任务运行时间",
        "是否成功",
        "最后commit",
        "job_id",
        "Output Token Throughput",
    ]

    all_rows: list[dict] = []
    existing_job_ids = set()

    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Check if header matches, if not, treat as new file
            if reader.fieldnames != header:
                logger.warning(
                    f"CSV header mismatch in {csv_path}. Rewriting with new header."
                )
                # If header mismatch, we don't load existing rows, effectively rewriting.
            else:
                for row in reader:
                    # Ensure all header keys are present in the row, fill missing with ''
                    for h in header:
                        row.setdefault(h, "")
                    all_rows.append(row)
                    existing_job_ids.add(row.get("job_id"))  # Use .get() for safety

    # Prepare new row as a dictionary
    new_row = {
        "任务运行时间": run_time,
        "是否成功": conclusion,
        "最后commit": commit_sha,
        "job_id": str(
            job_id
        ),  # Ensure job_id is string for consistent comparison with set
        "Output Token Throughput": (
            str(output_token_throughput) if output_token_throughput is not None else ""
        ),
    }

    if new_row["job_id"] in existing_job_ids:
        logger.info(f"Result for job {job_id} already exists in {csv_path}, skipping.")
    else:
        all_rows.append(new_row)

    # Sort all_rows by '任务运行时间'
    try:
        all_rows.sort(
            key=lambda x: datetime.fromisoformat(
                x["任务运行时间"].replace("Z", "+00:00")
            )
        )
    except Exception as e:
        logger.warning(f"Could not sort CSV: {e}")

    # Write all rows back to the CSV file
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)
    logger.success(f"Result recorded to {csv_path}")


@click.command()
@click.option(
    "--runs", "-n", default=4, show_default=True, help="Number of recent runs to check"
)
def main(runs: int):
    script_dir = Path(__file__).parent.resolve()
    LOGS_DIR = script_dir / "logs"

    logger.info(f"Target: {REPO} / {WORKFLOW_NAME}")
    logger.info(f"Job keywords: {TARGET_JOBS}")
    logger.info(f"Checking last {runs} runs")
    logger.info("-" * 50)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    runs_data = get_recent_runs(runs)
    if not runs_data:
        logger.error("No runs found")
        # Do not return here if no runs found, still proceed to git ops
        # if there are existing logs to commit/push.

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
                    LOGS_DIR,
                    run_date,
                    run_id,
                    job,
                    log_content,
                    matched_keyword,
                    commit_sha,
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
                LOGS_DIR,
                matched_keyword,
                emoji_conclusion,
                commit_sha,
                created_at,
                job["databaseId"],
                output_token_throughput,
            )

    logger.info("Step 2: Adding changes to git")
    # The logs directory is relative to the git repository root.
    # Assuming the current working directory is the repository root when this script is run,
    # or that gh_action_watcher_dir is directly inside the repo root.
    # Let's adjust for the project structure. The logs directory is gh_action_watcher/logs/
    repo_root = (
        script_dir.parent
    )  # Assuming gh_action_watcher is directly under repo root

    add_target = LOGS_DIR  # Changed from gh_action_watcher_dir / "logs"
    result = run_command(["git", "add", str(add_target)], cwd=repo_root)
    if result.returncode != 0:
        logger.error(f"Git add failed with exit code {result.returncode}")
        return

    logger.info("Step 3: Creating a commit")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"track ci data {timestamp}"

    result = run_command(["git", "commit", "-m", commit_message], cwd=repo_root)
    if result.returncode != 0:
        if "nothing to commit" in result.stderr:
            logger.info("No changes to commit. Skipping commit and push.")
            return
        logger.error(f"Git commit failed with exit code {result.returncode}")
        return

    logger.info("Step 4: Pushing changes to git")
    result = run_command(["git", "push"], cwd=repo_root)
    if result.returncode != 0:
        logger.error(f"Git push failed with exit code {result.returncode}")
        return

    logger.success("Script finished. Changes committed and pushed.")


if __name__ == "__main__":
    main()
