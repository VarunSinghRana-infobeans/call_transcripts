"""
start.py

One script to run the entire analysis pipeline.

Usage:
    python start.py                         # Run everything with auto provider
    python start.py --provider=mock         # Test without API costs (default if no key)
    python start.py --provider=openai       # Use OpenAI (requires OPENAI_API_KEY)
    python start.py --provider=ollama       # Use local Ollama model
    python start.py --skip-ppt              # Run analysis only, skip PowerPoint
    python start.py --scripts 01,02         # Run only data load (01) and call-type classification (02)

Script map:
    01  Load and validate data
    02  Classify call types
    03  Discover topics
    04  Analyze sentiment trends
    05  Churn, features, escalations
    06  Generate PowerPoint

Why this exists:
  Instead of running 6 commands manually (and forgetting the order),
  one command runs everything. It validates each step, reports progress,
  and stops on critical errors.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPTS = [
    ("01_explore.py", "Load and validate data"),
    ("02_call_types.py", "Classify call types"),
    ("03_topic_modeling.py", "Discover topics"),
    ("04_sentiment.py", "Analyze sentiment trends"),
    ("05_bonus_insights.py", "Churn, features, escalations"),
    ("06_generate_ppt.py", "Generate PowerPoint"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def preflight_checks(dataset_dir: Path) -> bool:
    """Validate environment before running pipeline."""
    all_ok = True
    print("Pre-flight checks...")

    # 1. Dataset exists
    if not dataset_dir.exists():
        print(f"  FAIL: Dataset not found: {dataset_dir}")
        all_ok = False
    else:
        call_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        print(f"  OK:   Dataset found ({len(call_dirs)} call folders)")
        if len(call_dirs) == 0:
            print(f"  FAIL: No call folders in dataset")
            all_ok = False
        elif len(call_dirs) < 10:
            print(f"  WARN: Only {len(call_dirs)} calls found (expected ~100)")

    # 2. Dependencies
    required = ["pandas", "numpy", "matplotlib", "sklearn", "hdbscan", "sentence_transformers", "pptx"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"  FAIL: Missing packages: {', '.join(missing)}")
        print(f"  FIX:  pip install {' '.join(missing)}")
        all_ok = False
    else:
        print(f"  OK:   All required packages installed")

    # 3. Scripts exist
    script_dir = Path(__file__).parent
    for script_name, _ in SCRIPTS:
        if not (script_dir / script_name).exists():
            print(f"  FAIL: Script missing: {script_name}")
            all_ok = False
    if all_ok:
        print(f"  OK:   All {len(SCRIPTS)} scripts found")

    # 4. Output writable
    output_dir = script_dir / "output"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        print(f"  OK:   Output directory writable")
    except Exception as e:
        print(f"  FAIL: Cannot write to output directory: {e}")
        all_ok = False

    return all_ok


def run_script(script_name: str, description: str, provider: str, run_name: str) -> bool:
    """Run a single script. Return True if success, False if failed."""
    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False

    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"Task:    {description}")
    print(f"{'='*60}")

    start_time = time.time()

    # Build command with provider override
    # Use os.environ.copy() to inherit all system env vars (including HOME/USERPROFILE)
    import os
    env = os.environ.copy()
    env["AI_PROVIDER"] = provider
    # Pass shared output folder name so all scripts write to the same folder
    env["OUTPUT_RUN_NAME"] = run_name

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path(__file__).parent,
            env=env,
            check=False,
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"OK: {script_name} completed in {elapsed:.1f}s")
            return True
        else:
            print(f"ERROR: {script_name} failed with code {result.returncode}")
            return False

    except Exception as e:
        print(f"ERROR: Could not run {script_name}: {e}")
        return False


def update_latest_junction(run_dir: Path) -> None:
    """Point output/latest to the most recent run so users can always find it."""
    output_dir = run_dir.parent
    latest_link = output_dir / "latest"
    try:
        # Remove existing junction / symlink / directory
        if latest_link.exists() or latest_link.is_symlink():
            if latest_link.is_dir() and not latest_link.is_symlink():
                import shutil
                shutil.rmtree(latest_link)
            else:
                latest_link.unlink()

        # Create a Windows directory junction (works without admin)
        import subprocess as sp
        sp.run(["cmd", "/c", "mklink", "/J", str(latest_link), str(run_dir)], check=True, capture_output=True)
        print(f"  latest -> {run_dir.name}")
    except Exception as e:
        print(f"  WARN: Could not update output/latest junction: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the transcript analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Provider options:
  --provider=auto     Auto-detect OpenAI API key, fallback to mock (default)
  --provider=mock     No API calls; fast, deterministic, no cost
  --provider=openai   Use OpenAI GPT models (set OPENAI_API_KEY env var)
  --provider=ollama   Use local Ollama endpoint (configure in .env)

Examples:
  python start.py                       # Full pipeline, auto provider
  python start.py --provider=mock       # Full pipeline, mock AI
  python start.py --provider=openai     # Full pipeline, OpenAI
  python start.py --skip-ppt            # Analysis only, auto provider
  python start.py --scripts 01,02       # Data load + call-type classification
  python start.py --scripts 05,06       # Bonus insights + PowerPoint

Script numbers:
  01 = Load and validate data
  02 = Classify call types
  03 = Discover topics
  04 = Analyze sentiment trends
  05 = Churn, features, escalations
  06 = Generate PowerPoint
        """
    )

    parser.add_argument(
        "--provider",
        choices=["openai", "ollama", "mock", "auto"],
        default="auto",
        help="AI provider to use: auto | mock | openai | ollama (default: auto-detect)",
    )

    parser.add_argument(
        "--skip-ppt",
        action="store_true",
        help="Skip PowerPoint generation (analysis only)",
    )

    parser.add_argument(
        "--scripts",
        type=str,
        default="all",
        help="Comma-separated script numbers to run, e.g. '01,02,03' (default: all)",
    )

    args = parser.parse_args()

    # Determine which scripts to run
    if args.scripts == "all":
        scripts_to_run = SCRIPTS.copy()
    else:
        requested = [s.strip().zfill(2) for s in args.scripts.split(",")]
        scripts_to_run = [(name, desc) for name, desc in SCRIPTS
                          if any(name.startswith(r) for r in requested)]

    # Skip PPT if requested
    if args.skip_ppt:
        scripts_to_run = [(n, d) for n, d in scripts_to_run
                          if n != "06_generate_ppt.py"]

    # Auto-detect provider
    provider = args.provider
    if provider == "auto":
        import os
        if os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "mock"
            print("No OPENAI_API_KEY found. Using mock provider.")
            print("Set AI_PROVIDER=openai and OPENAI_API_KEY for real AI.")

    # Create a shared run folder name (one timestamp for the whole pipeline)
    from datetime import datetime
    run_name = "Transcript_Intelligence_" + datetime.now().strftime("%Y-%m-%d_%H%M%S")

    print("=" * 60)
    print("TRANSCRIPT INTELLIGENCE PIPELINE")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"Run:      {run_name}")
    print(f"Scripts:  {len(scripts_to_run)}")
    print(f"Output:   {Path(__file__).parent / 'output' / run_name}")
    print("=" * 60)

    # Pre-flight validation
    dataset_dir = Path(__file__).parent.parent.parent / "interview-assignment" / "dataset"
    # Fallback: try dataset at repo root
    if not dataset_dir.exists():
        dataset_dir = Path(__file__).parent.parent.parent / "dataset"
    if not preflight_checks(dataset_dir):
        print("\nPre-flight checks failed. Fix issues above and re-run.")
        sys.exit(1)

    # Run pipeline
    total_start = time.time()
    results = []

    for script_name, description in scripts_to_run:
        success = run_script(script_name, description, provider, run_name)
        results.append((script_name, success))

        if not success:
            print(f"\n{'='*60}")
            print("PIPELINE STOPPED")
            print(f"{'='*60}")
            print(f"{script_name} failed. Fix the error and re-run.")
            print(f"You can resume from this script:")
            print(f"  python start.py --scripts {script_name[:2]}")
            sys.exit(1)

    total_elapsed = time.time() - total_start

    # Point output/latest to this run
    run_dir = Path(__file__).parent / "output" / run_name
    update_latest_junction(run_dir)

    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Scripts run: {len(results)}")
    print("")
    print("Output files:")
    ppt_path = run_dir / "Transcript_Intelligence_Report.pptx"
    charts_dir = run_dir / "charts"
    if ppt_path.exists():
        print(f"  Report:   output/{run_name}/Transcript_Intelligence_Report.pptx")
    print(f"  Charts:   output/{run_name}/charts/ ({len(list(charts_dir.glob('*.png')))} files)")
    print(f"  Data:     output/{run_name}/*.csv, output/{run_name}/*.json")
    print(f"  Latest:   output/latest/ -> output/{run_name}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
