"""
Shared utilities for transcript analysis scripts.
No Jupyter needed. Just Python.

WHY THIS FILE EXISTS:
  Common operations (JSON loading, chart saving, path resolution) are
  extracted here so each analysis script focuses on its own logic.
  DRY principle: define once, use in 01-06.

WHY RELATIVE PATHS:
  Hardcoded absolute paths (e.g., "d:/call_trannscript/...") only work
  on the author's machine. We use Path(__file__) navigation so the
  code works on any machine, any OS, in any parent directory.
"""

import json
import os
from pathlib import Path
from typing import Any
import pandas as pd
from datetime import datetime


# ---------------------------------------------------------------------------
# Paths (relative to this file's location)
# ---------------------------------------------------------------------------

# This file is at: assignment/notebook/scripts/utils.py
# Dataset is at:    interview-assignment/dataset/
# Output goes to:   assignment/notebook/output/<run_name>/

SCRIPT_DIR = Path(__file__).parent.resolve()
NOTEBOOK_DIR = SCRIPT_DIR.parent.resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent.parent.resolve()

DATASET_DIR = PROJECT_ROOT / "interview-assignment" / "dataset"

# Output directory: use env var if set (for pipeline runs), else timestamped
_RUN_NAME = os.environ.get("OUTPUT_RUN_NAME", "")
if not _RUN_NAME:
    # Default: use "latest" for manual single-script runs
    _RUN_NAME = "latest"

OUTPUT_DIR = NOTEBOOK_DIR / "output" / _RUN_NAME
CHARTS_DIR = OUTPUT_DIR / "charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output folder: {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# JSON Loading
# ---------------------------------------------------------------------------

def load_json_file(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, return None if missing or invalid."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def load_all_calls() -> list[dict[str, Any]]:
    """
    Scan dataset directory and load all call data.
    Returns list of dicts with all files per call.
    """
    calls = []

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}\n"
                                f"Expected at: {DATASET_DIR}\n"
                                f"Please ensure the dataset is in the correct location.")

    for meeting_dir in sorted(DATASET_DIR.iterdir()):
        if not meeting_dir.is_dir():
            continue
        if meeting_dir.name.startswith("."):
            continue

        call_data = {
            "meeting_id": meeting_dir.name,
            "meeting_info": load_json_file(meeting_dir / "meeting-info.json"),
            "transcript": load_json_file(meeting_dir / "transcript.json"),
            "summary": load_json_file(meeting_dir / "summary.json"),
            "speakers": load_json_file(meeting_dir / "speakers.json"),
            "events": load_json_file(meeting_dir / "events.json"),
            "speaker_meta": load_json_file(meeting_dir / "speaker-meta.json"),
        }
        calls.append(call_data)

    return calls


# ---------------------------------------------------------------------------
# Data Extraction Helpers
# ---------------------------------------------------------------------------

def extract_meeting_title(call: dict) -> str:
    """Get meeting title from meeting_info."""
    info = call.get("meeting_info") or {}
    return info.get("title", "Unknown")


def extract_meeting_duration_minutes(call: dict) -> float:
    """Calculate duration in minutes from meeting_info."""
    info = call.get("meeting_info") or {}
    # Some meeting-info.json has a 'duration' field directly
    if "duration" in info:
        return float(info["duration"])

    start = info.get("startTime", "")
    end = info.get("endTime", "")

    if not start or not end:
        return 0.0

    try:
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        return (end_dt - start_dt).total_seconds() / 60.0
    except (ValueError, TypeError):
        return 0.0


def extract_meeting_date(call: dict) -> str:
    """Get meeting date string from meeting_info."""
    info = call.get("meeting_info") or {}
    start = info.get("startTime", "")
    if not start:
        return ""
    try:
        dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return ""


def extract_summary_text(call: dict) -> str:
    """Get summary text from summary.json."""
    summary = call.get("summary") or {}
    # summary.json has key "summary" for the text, not "text"
    return summary.get("summary", "") or summary.get("text", "")


def count_transcript_sentences(call: dict) -> int:
    """Count sentences in transcript."""
    transcript = call.get("transcript") or {}
    sentences = transcript.get("data", [])
    return len(sentences)


def extract_speaker_count(call: dict) -> int:
    """Count unique speakers."""
    speakers = call.get("speakers") or []
    if isinstance(speakers, dict):
        speaker_list = speakers.get("speakers", [])
    else:
        # speakers.json is a list of speaker turns
        speaker_list = speakers
    unique_names = set(s.get("speakerName", "") for s in speaker_list)
    return len(unique_names)


# ---------------------------------------------------------------------------
# LLM Helper - Unified AI Provider
# ---------------------------------------------------------------------------
# Import from ai_config.py which supports OpenAI, Ollama (local), and Mock.
# See ai_config.py for setup instructions.
# ---------------------------------------------------------------------------

from ai_config import llm_classify, llm_generate


# ---------------------------------------------------------------------------
# Chart Styling
# ---------------------------------------------------------------------------

def set_chart_style(base_size: int = 11):
    """Apply a presentation-friendly matplotlib style with larger text."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": base_size,
        "axes.titlesize": base_size + 2,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1,
        "figure.titlesize": base_size + 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Chart Saving
# ---------------------------------------------------------------------------

def save_chart(fig, filename: str) -> Path:
    """Save matplotlib figure to charts directory."""
    path = CHARTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Data Export
# ---------------------------------------------------------------------------

def save_csv(df: pd.DataFrame, filename: str) -> Path:
    """Save DataFrame to output directory."""
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"Data saved: {path}")
    return path


def save_json(data: Any, filename: str) -> Path:
    """Save data as JSON to output directory."""
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Data saved: {path}")
    return path
