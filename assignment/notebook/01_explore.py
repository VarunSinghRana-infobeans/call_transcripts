"""
01_explore.py

Load all 100 calls, validate schema, print distributions.
Exports: calls_summary.csv, distribution charts

WHY THIS SCRIPT EXISTS:
  Before analyzing data, you must understand it. This script is the
  "look before you leap" step. It tells us: how many calls, how long,
  what dates, what sentiment, are there missing files?

WHY PANDAS (NOT POSTGRESQL):
  100 rows fit in memory (~5 MB). Pandas is faster for exploration than
  writing SQL queries. PostgreSQL is overkill for this scale. See
  notebook_decisions.md for the full storage comparison.

WHY MATPLOTLIB:
  It is the standard for static charts in Python. The output is PNG files
  that go directly into the PowerPoint. No interactivity needed here.
"""

import sys
from pathlib import Path

# Add scripts folder to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter

from utils import (
    load_all_calls,
    extract_meeting_title,
    extract_meeting_duration_minutes,
    extract_meeting_date,
    extract_summary_text,
    count_transcript_sentences,
    extract_speaker_count,
    create_chart_fig,
    save_chart,
    save_csv,
    save_json,
    set_chart_style,
    OUTPUT_DIR,
)


def main():
    set_chart_style()
    print("=" * 60)
    print("01 EXPLORE: Loading and validating all calls")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load all calls
    # ------------------------------------------------------------------
    calls = load_all_calls()
    print(f"\nTotal calls loaded: {len(calls)}")

    # ------------------------------------------------------------------
    # Build summary DataFrame
    # ------------------------------------------------------------------
    rows = []
    for call in calls:
        info = call.get("meeting_info") or {}
        summary = call.get("summary") or {}
        transcript = call.get("transcript") or {}

        # Count sentiment types
        sentences = transcript.get("data", [])
        sentiment_counts = Counter(s.get("sentimentType", "unknown") for s in sentences)

        row = {
            "meeting_id": call["meeting_id"],
            "title": extract_meeting_title(call),
            "date": extract_meeting_date(call),
            "duration_minutes": info.get("duration", extract_meeting_duration_minutes(call)),
            "speaker_count": extract_speaker_count(call),
            "sentence_count": len(sentences),
            "overall_sentiment": summary.get("overallSentiment", "unknown"),
            "sentiment_score": summary.get("sentimentScore", 0),
            "neutral_count": sentiment_counts.get("neutral", 0),
            "positive_count": sentiment_counts.get("positive", 0),
            "negative_count": sentiment_counts.get("negative", 0),
            "summary_text": extract_summary_text(call),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------------
    print("\n--- Validation ---")
    print(f"Calls with missing meeting_info: {df['title'].eq('Unknown').sum()}")
    print(f"Calls with missing transcript: {df['sentence_count'].eq(0).sum()}")
    print(f"Calls with missing summary: {df['summary_text'].eq('').sum()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # ------------------------------------------------------------------
    # Distributions
    # ------------------------------------------------------------------
    print("\n--- Duration Distribution ---")
    print(df["duration_minutes"].describe())

    print("\n--- Sentiment Score Distribution ---")
    print(df["sentiment_score"].describe())

    print("\n--- Overall Sentiment Counts ---")
    print(df["overall_sentiment"].value_counts())

    print("\n--- Speaker Count Distribution ---")
    print(df["speaker_count"].value_counts().sort_index())

    print("\n--- Top 10 Longest Calls ---")
    print(df.nlargest(10, "duration_minutes")[["title", "duration_minutes", "overall_sentiment"]].to_string())

    print("\n--- Top 10 Most Negative Calls ---")
    print(df.nsmallest(10, "sentiment_score")[["title", "sentiment_score", "overall_sentiment"]].to_string())

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    print("\n--- Generating charts ---")

    # Duration histogram
    fig, ax = create_chart_fig("01_duration_distribution.png")
    ax.hist(df["duration_minutes"], bins=20, edgecolor="black")
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Number of calls")
    ax.set_title("Call Duration Distribution")
    save_chart(fig, "01_duration_distribution.png")
    plt.close(fig)

    # Sentiment score histogram
    fig, ax = create_chart_fig("01_sentiment_distribution.png")
    ax.hist(df["sentiment_score"], bins=20, edgecolor="black", color="coral")
    ax.set_xlabel("Sentiment Score (0=negative, 5=positive)")
    ax.set_ylabel("Number of calls")
    ax.set_title("Sentiment Score Distribution")
    save_chart(fig, "01_sentiment_distribution.png")
    plt.close(fig)

    # Overall sentiment bar chart
    fig, ax = create_chart_fig("01_overall_sentiment.png")
    sentiment_counts = df["overall_sentiment"].value_counts()
    ax.bar(sentiment_counts.index, sentiment_counts.values, color=["green", "orange", "red", "gray"])
    ax.set_xlabel("Overall Sentiment")
    ax.set_ylabel("Number of calls")
    ax.set_title("Overall Sentiment Counts")
    save_chart(fig, "01_overall_sentiment.png")
    plt.close(fig)

    # Call volume over time
    df["date"] = pd.to_datetime(df["date"])
    daily_counts = df.groupby(df["date"].dt.date).size()

    fig, ax = create_chart_fig("01_call_volume_over_time.png")
    ax.plot(daily_counts.index, daily_counts.values, marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of calls")
    ax.set_title("Call Volume Over Time")
    fig.autofmt_xdate()
    save_chart(fig, "01_call_volume_over_time.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Save data
    # ------------------------------------------------------------------
    save_csv(df, "01_calls_summary.csv")
    save_json({
        "total_calls": len(calls),
        "date_range": {"min": str(df["date"].min().date()), "max": str(df["date"].max().date())},
        "duration": {"min": float(df["duration_minutes"].min()), "max": float(df["duration_minutes"].max()), "mean": float(df["duration_minutes"].mean())},
        "sentiment": {"min": float(df["sentiment_score"].min()), "max": float(df["sentiment_score"].max()), "mean": float(df["sentiment_score"].mean())},
        "overall_sentiment_counts": df["overall_sentiment"].value_counts().to_dict(),
    }, "01_explore_stats.json")

    print("\n" + "=" * 60)
    print("01 EXPLORE: Done")
    print(f"Output saved to: {OUTPUT_DIR}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
