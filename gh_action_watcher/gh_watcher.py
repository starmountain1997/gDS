#!/usr/bin/env python3
"""
Runs save_gh_logs.py, then commits and pushes changes to git.
"""

import csv
import json
import re
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

REPO = "vllm-project/vllm-ascend"
WORKFLOW_NAME = "schedule_nightly_test_a3.yaml"
TARGET_JOBS = [
    "multi-node-dpsk3.2-2node",
    "deepseek-v3.2-W8A8-EP",
    "deepseek-v3-2-w8a8",
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


def init_db(logs_dir: Path) -> None:
    """Initialize SQLite database."""
    db_path = logs_dir / "results.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            keyword     TEXT NOT NULL,
            job_id      TEXT NOT NULL,
            run_time    TEXT NOT NULL,
            success     TEXT NOT NULL,
            commit_sha  TEXT NOT NULL,
            throughput  REAL,
            run_id      TEXT,
            PRIMARY KEY (keyword, job_id)
        )
    """)
    # Migrate existing DBs
    cols = {r[1] for r in conn.execute("PRAGMA table_info(results)").fetchall()}
    if "run_id" not in cols:
        conn.execute("ALTER TABLE results ADD COLUMN run_id TEXT")
    if "error_type" not in cols:
        conn.execute("ALTER TABLE results ADD COLUMN error_type TEXT")
    conn.commit()
    conn.close()


def upsert_result(
    conn: sqlite3.Connection,
    keyword: str,
    conclusion: str,
    commit_sha: str,
    run_time: str,
    run_id: int,
    job_id: int,
    output_token_throughput: float | None = None,
):
    """Insert result into SQLite, ignoring duplicates."""
    conn.execute(
        "INSERT OR IGNORE INTO results (keyword, job_id, run_time, success, commit_sha, throughput, run_id) VALUES (?,?,?,?,?,?,?)",
        (
            keyword,
            str(job_id),
            run_time,
            conclusion,
            commit_sha,
            output_token_throughput,
            str(run_id),
        ),
    )
    conn.commit()
    logger.success(f"Result recorded for job {job_id}")


def export_csv(logs_dir: Path, conn: sqlite3.Connection):
    """Export SQLite results to per-keyword CSV files."""
    keywords = [
        r[0] for r in conn.execute("SELECT DISTINCT keyword FROM results").fetchall()
    ]
    for keyword in keywords:
        rows = conn.execute(
            "SELECT run_time, success, commit_sha, job_id, throughput, run_id, error_type "
            "FROM results WHERE keyword = ? ORDER BY run_time",
            (keyword,),
        ).fetchall()
        csv_path = logs_dir / f"{keyword}_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "任务运行时间",
                    "是否成功",
                    "最后commit",
                    "job_id",
                    "Output Token Throughput",
                    "job_link",
                    "错误类型",
                ]
            )
            for (
                run_time,
                success,
                commit_sha,
                job_id,
                throughput,
                run_id,
                error_type,
            ) in rows:
                job_link = (
                    f"https://github.com/{REPO}/actions/runs/{run_id}/job/{job_id}"
                    if run_id
                    else ""
                )
                writer.writerow(
                    [
                        run_time,
                        success,
                        commit_sha,
                        job_id,
                        throughput if throughput is not None else "",
                        job_link,
                        error_type or "",
                    ]
                )
        logger.info(f"Exported {len(rows)} rows → {csv_path.name}")


EMOJI = {"success": "✅", "failure": "❌", "cancelled": "⚪", "skipped": "🚫"}


@click.group()
def main():
    pass


@main.command()
@click.option(
    "--runs", "-n", default=4, show_default=True, help="Number of recent runs to check"
)
@click.option(
    "--workers", "-w", default=8, show_default=True, help="Concurrent workers"
)
def fetch(runs: int, workers: int):
    script_dir = Path(__file__).parent.resolve()
    LOGS_DIR = script_dir / "logs"

    logger.info(f"Target: {REPO} / {WORKFLOW_NAME}")
    logger.info(f"Job keywords: {TARGET_JOBS}")
    logger.info(f"Checking last {runs} runs with {workers} workers")
    logger.info("-" * 50)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    db_path = LOGS_DIR / "results.db"
    init_db(LOGS_DIR)

    runs_data = get_recent_runs(runs)
    if not runs_data:
        logger.error("No runs found")

    # Phase 1: fetch jobs for all runs concurrently
    run_jobs: list[tuple[dict, list[dict]]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(find_target_jobs, r["databaseId"]): r for r in runs_data}
        for fut in as_completed(futures):
            run = futures[fut]
            jobs = fut.result()
            logger.info(
                f"Run #{run['number']} ({run['createdAt'][:10]}) "
                f"- {run.get('conclusion', '?')} - {len(jobs)} target jobs"
            )
            if jobs:
                run_jobs.append((run, jobs))

    # Phase 2: fetch logs for all (run, job) pairs concurrently
    def _process_job(run: dict, job: dict, _db_path: Path = db_path):
        commit_sha = run.get("headSha", "")
        run_date = run["createdAt"][:10]
        matched_keyword = next((k for k in TARGET_JOBS if k in job.get("name", "")), "")
        short_sha = commit_sha[:7] if commit_sha else str(run["databaseId"])
        expected_log = (
            LOGS_DIR
            / run_date
            / f"{run_date}_{short_sha}_{matched_keyword}_{job['databaseId']}.log"
        )

        if expected_log.exists():
            logger.info(f"  {job['name']} log skipped (already exists)")
            log_content = expected_log.read_text(encoding="utf-8", errors="replace")
        else:
            log_content = get_job_log(run["databaseId"], job["databaseId"])
            if log_content:
                log_path = save_log(
                    LOGS_DIR,
                    run_date,
                    run["databaseId"],
                    job,
                    log_content,
                    matched_keyword,
                    commit_sha,
                )
                logger.success(f"  Log saved: {log_path}")

        output_token_throughput = (
            extract_output_token_throughput(log_content) if log_content else None
        )
        thread_conn = sqlite3.connect(_db_path)
        upsert_result(
            thread_conn,
            matched_keyword,
            EMOJI.get(job.get("conclusion", ""), "?"),
            commit_sha,
            run["createdAt"],
            run["databaseId"],
            job["databaseId"],
            output_token_throughput,
        )
        thread_conn.close()

    tasks = [(run, job) for run, jobs in run_jobs for job in jobs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_job, run, job): (run, job) for run, job in tasks
        }
        for fut in as_completed(futures):
            fut.result()

    export_csv(LOGS_DIR, sqlite3.connect(db_path))

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


ANALYZE_PROMPT = """\
你是 CI 故障分析师。下面是一份以失败告终的 GitHub Actions 日志。

第一行必须是以下三种之一：
类型: 代码问题
类型: 性能精度不达标
类型: 设备环境问题

然后给出根本原因（≤5 条）和最可能的修复方案，用中文输出。

日志：
{log}
"""


def _parse_error_type(analysis: str) -> str:
    match = re.search(r"类型[:：]\s*(.+)", analysis)
    return match.group(1).strip() if match else ""


def _find_log_file(logs_dir: Path, job_id: str) -> Path | None:
    for log_file in logs_dir.rglob(f"*_{job_id}.log"):
        return log_file
    return None


def _call_ai(client: OpenAI, model: str, log_tail: str) -> str:
    import time

    from openai import RateLimitError

    for attempt in range(6):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": ANALYZE_PROMPT.format(log=log_tail)}
                ],
            )
            content = response.choices[0].message.content or ""
            return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        except RateLimitError:
            wait = 2**attempt * 5  # 5, 10, 20, 40, 80, 160s
            logger.warning(
                f"Rate limited, retrying in {wait}s (attempt {attempt + 1}/6)..."
            )
            time.sleep(wait)
    raise RuntimeError("AI call failed after 6 retries due to rate limit")


def _analyze_job(
    client: OpenAI,
    model: str,
    logs_dir: Path,
    kw: str,
    job_id: str,
    run_time: str,
    commit_sha: str,
    tail: int,
) -> tuple[str, str, str, str, str | None]:
    """Analyze a single job; returns (kw, job_id, run_time, commit_sha, analysis_or_None)."""
    log_file = _find_log_file(logs_dir, job_id)
    if not log_file:
        return kw, job_id, run_time, commit_sha, None

    analysis_file = log_file.with_suffix(log_file.suffix + ".analysis")
    if analysis_file.exists():
        return (
            kw,
            job_id,
            run_time,
            commit_sha,
            analysis_file.read_text(encoding="utf-8"),
        )

    log_lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    log_tail = "\n".join(log_lines[-tail:])
    analysis = _call_ai(client, model, log_tail)
    analysis_file.write_text(analysis, encoding="utf-8")
    return kw, job_id, run_time, commit_sha, analysis


@main.command()
@click.option(
    "--date",
    "-d",
    default=None,
    help="Filter by date (YYYY-MM-DD); defaults to latest available",
)
@click.option("--keyword", "-k", default=None, help="Filter by job keyword")
@click.option(
    "--tail", "-t", default=300, show_default=True, help="Last N log lines sent to AI"
)
@click.option(
    "--workers", "-w", default=2, show_default=True, help="Concurrent AI calls"
)
def analyze(date: str | None, keyword: str | None, tail: int, workers: int):
    """Analyze failed CI logs with AI."""
    script_dir = Path(__file__).parent.resolve()
    env_path = script_dir.parent / ".env"
    load_dotenv(env_path)

    import os

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")

    if not api_key:
        logger.error(f".env not found or OPENAI_API_KEY missing (looked in {env_path})")
        raise SystemExit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)
    logs_dir = script_dir / "logs"
    db_path = logs_dir / "results.db"

    if not db_path.exists():
        logger.error("results.db not found — run `gh-watcher fetch` first")
        raise SystemExit(1)

    conn = sqlite3.connect(db_path)
    # migrate
    cols = {r[1] for r in conn.execute("PRAGMA table_info(results)").fetchall()}
    if "error_type" not in cols:
        conn.execute("ALTER TABLE results ADD COLUMN error_type TEXT")
        conn.commit()

    query = (
        "SELECT keyword, job_id, run_time, commit_sha FROM results WHERE success = '❌'"
    )
    params: list = []
    if keyword:
        query += " AND keyword = ?"
        params.append(keyword)
    if date:
        query += " AND run_time LIKE ?"
        params.append(f"{date}%")

    query += " ORDER BY run_time DESC"
    rows = conn.execute(query, params).fetchall()

    if not rows:
        logger.info("No failed jobs found for the given filters.")
        conn.close()
        return

    logger.info(f"Found {len(rows)} failed job(s) to analyze with {workers} workers")

    results: list[tuple[str, str, str, str, str | None]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _analyze_job,
                client,
                model,
                logs_dir,
                kw,
                job_id,
                run_time,
                commit_sha,
                tail,
            ): job_id
            for kw, job_id, run_time, commit_sha in rows
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: r[2], reverse=True)

    for kw, job_id, run_time, commit_sha, analysis in results:
        log_file = _find_log_file(logs_dir, job_id)
        click.echo(f"\n{'=' * 70}")
        click.echo(f"Job:    {kw}")
        click.echo(f"JobID:  {job_id}")
        click.echo(f"Date:   {run_time[:10]}  Commit: {commit_sha[:7]}")
        if log_file:
            click.echo(f"Log:    {log_file}")
        click.echo("─" * 70)
        if analysis is None:
            click.echo("(log file not found — skipped)")
        else:
            error_type = _parse_error_type(analysis)
            conn.execute(
                "UPDATE results SET error_type = ? WHERE job_id = ?",
                (error_type, job_id),
            )
            click.echo(analysis)

    conn.commit()
    export_csv(logs_dir, conn)
    conn.close()

    logger.info("Step 3: Adding changes to git")
    repo_root = script_dir.parent

    result = run_command(["git", "add", str(logs_dir)], cwd=repo_root)
    if result.returncode != 0:
        logger.error(f"Git add failed with exit code {result.returncode}")
        return

    logger.info("Step 4: Creating a commit")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"prompt(gh-watcher): analyze {timestamp}"

    result = run_command(["git", "commit", "-m", commit_message], cwd=repo_root)
    if result.returncode != 0:
        if "nothing to commit" in result.stderr:
            logger.info("No changes to commit. Skipping commit and push.")
            return
        logger.error(f"Git commit failed with exit code {result.returncode}")
        return

    logger.info("Step 5: Pushing changes to git")
    result = run_command(["git", "push"], cwd=repo_root)
    if result.returncode != 0:
        logger.error(f"Git push failed with exit code {result.returncode}")
        return

    logger.success("Analyze finished. Changes committed and pushed.")


if __name__ == "__main__":
    main()
