"""
run_all.py

Run the full notebook pipeline end-to-end in the correct order.
Each run gets a timestamped output folder, and `output/latest` is updated
to point to it (Windows junction) so you always have a stable path to the
latest results.

Usage:
    python run_all.py
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = NOTEBOOK_DIR / "output"


def run_script(name: str, env: dict) -> int:
    print(f"\n{'=' * 60}")
    print(f"Running {name}")
    print(f"{'=' * 60}")
    result = subprocess.run(
        [sys.executable, str(NOTEBOOK_DIR / name)],
        cwd=NOTEBOOK_DIR,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        print(f"ERROR: {name} failed with exit code {result.returncode}")
    return result.returncode


def _win_path(path: Path) -> str:
    """Return a Windows-style path with backslashes for cmd.exe."""
    return os.fspath(path).replace("/", "\\")


def update_latest_junction(run_name: str) -> None:
    """Point output/latest to the new run folder using a Windows junction."""
    latest = OUTPUT_DIR / "latest"
    target = OUTPUT_DIR / run_name

    # Remove any existing latest junction/directory (junction removal does NOT
    # delete the target folder when using rmdir on a reparse point).
    if latest.exists() or latest.is_symlink():
        subprocess.run(
            ["cmd", "/c", f'rmdir /s /q "{_win_path(latest)}"'],
            check=False,
        )

    # Create a junction. Falls back to a plain directory copy if junctions fail.
    result = subprocess.run(
        ["cmd", "/c", f'mklink /J "{_win_path(latest)}" "{_win_path(target)}"'],
        check=False,
    )
    if result.returncode != 0:
        print("WARNING: Could not create latest junction. Leaving timestamped folder only.")


def main() -> int:
    run_name = datetime.now().strftime("Transcript_Intelligence_%Y-%m-%d_%H%M%S")
    print(f"Pipeline run: {run_name}")

    env = os.environ.copy()
    env["OUTPUT_RUN_NAME"] = run_name

    scripts = [
        "01_explore.py",
        "02_call_types.py",
        "03_topic_modeling.py",
        "04_sentiment.py",
        "05_bonus_insights.py",
        "06_generate_ppt.py",
        "export_pdf.py",
    ]

    for script in scripts:
        rc = run_script(script, env)
        if rc != 0:
            print(f"\nPipeline stopped because {script} failed.")
            return rc

    update_latest_junction(run_name)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"Run folder:  output/{run_name}")
    print(f"Latest link: output/latest")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
