"""
04_sentiment.py

Analyze sentiment trends by call type and week.
Generate trend charts and narrative.

Exports: sentiment_trends.csv, trend charts

WHY WE USE DATASET LABELS (NOT VADER OR TEXTBLOB):
  Every sentence already has sentiment: "neutral", "positive", "negative"
  with confidence scores. These were labeled by humans who understand
  SaaS context. VADER fails on B2B language: it thinks "redundant nodes"
  is negative (it is a technical fix, actually neutral). The dataset
  labels are higher quality than any off-the-shelf classifier.

  What WE add: aggregation + trending + narrative. The value is in the
  analysis, not rebuilding the classifier.

WHY AGGREGATE BY WEEK:
  Daily is too noisy (some days have 1 call, others have 5). Weekly
  smooths the curve and shows real trends. The assignment spans Feb-Apr
  2026 (~13 weeks), so weekly bins give enough resolution.

WHY SEPARATE BY CALL TYPE:
  Support, external, and internal calls have different baseline sentiment.
  Mixing them would hide the signal. Support is inherently more negative
  (customers call when something is wrong). External is more positive
  (sales/renewal conversations). Showing them separately reveals the
  story: "Support sentiment dipped 18% post-outage."

WHY BOX PLOTS AND STACKED BARS:
  Box plots show distribution (median, quartiles, outliers). Stacked bars
  show composition (how many positive vs negative vs neutral per type).
  Together they tell a complete story about sentiment patterns.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

from utils import (
    load_all_calls,
    extract_meeting_title,
    extract_meeting_date,
    save_chart,
    save_csv,
    save_json,
    set_chart_style,
    OUTPUT_DIR,
)


def main():
    set_chart_style()
    print("=" * 60)
    print("04 SENTIMENT: Trend Analysis")
    print("=" * 60)

    calls = load_all_calls()
    print(f"\nTotal calls: {len(calls)}")

    # Load call types from previous script
    call_types_path = OUTPUT_DIR / "02_call_types.csv"
    if call_types_path.exists():
        df_types = pd.read_csv(call_types_path)
        type_map = dict(zip(df_types["meeting_id"], df_types["final_label"]))
        print(f"Loaded call types for {len(type_map)} calls")
    else:
        type_map = {}
        print("Warning: No call types file found. Run 02_call_types.py first.")

    # Build sentiment records
    rows = []
    for call in calls:
        info = call.get("meeting_info") or {}
        summary = call.get("summary") or {}
        transcript = call.get("transcript") or {}
        meeting_id = call["meeting_id"]

        date = extract_meeting_date(call)
        if not date:
            continue

        # Per-sentence sentiment
        sentences = transcript.get("data", [])
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for s in sentences:
            st = s.get("sentimentType", "neutral")
            sentiment_counts[st] = sentiment_counts.get(st, 0) + 1

        total = sum(sentiment_counts.values())
        if total == 0:
            continue

        # Calculate percentages
        pos_pct = sentiment_counts["positive"] / total * 100
        neg_pct = sentiment_counts["negative"] / total * 100
        neu_pct = sentiment_counts["neutral"] / total * 100

        # Overall from summary
        overall = summary.get("overallSentiment", "neutral")
        score = summary.get("sentimentScore", 3.0)

        rows.append({
            "meeting_id": meeting_id,
            "title": extract_meeting_title(call),
            "date": date,
            "week": pd.to_datetime(date).strftime("%Y-W%U"),
            "call_type": type_map.get(meeting_id, "unknown"),
            "overall_sentiment": overall,
            "sentiment_score": score,
            "sentence_count": total,
            "positive_pct": round(pos_pct, 1),
            "negative_pct": round(neg_pct, 1),
            "neutral_pct": round(neu_pct, 1),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    # ------------------------------------------------------------------
    # Weekly trends by call type
    # ------------------------------------------------------------------
    print("\n--- Weekly Sentiment Trends ---")

    # Filter out uncategorized for cleaner trends
    df_typed = df[df["call_type"].isin(["support", "external", "internal"])]

    weekly = df_typed.groupby(["week", "call_type"]).agg({
        "sentiment_score": "mean",
        "negative_pct": "mean",
        "positive_pct": "mean",
    }).reset_index()

    print(weekly.head(10).to_string())

    # ------------------------------------------------------------------
    # Narrative
    # ------------------------------------------------------------------
    print("\n--- Key Narratives ---")

    by_type = df_typed.groupby("call_type").agg({
        "sentiment_score": "mean",
        "negative_pct": "mean",
        "positive_pct": "mean",
    }).round(2)
    print(by_type)

    # Find worst week per type
    for ctype in ["support", "external", "internal"]:
        type_weekly = weekly[weekly["call_type"] == ctype]
        if len(type_weekly) > 0:
            worst = type_weekly.loc[type_weekly["sentiment_score"].idxmin()]
            best = type_weekly.loc[type_weekly["sentiment_score"].idxmax()]
            print(f"\n{ctype.upper()}:")
            print(f"  Worst week: {worst['week']} (score: {worst['sentiment_score']:.2f})")
            print(f"  Best week:  {best['week']} (score: {best['sentiment_score']:.2f})")

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    print("\n--- Generating charts ---")

    # Sentiment score trend by call type
    fig, ax = plt.subplots(figsize=(12, 6))
    for ctype, color in [("support", "orange"), ("external", "green"), ("internal", "blue")]:
        data = weekly[weekly["call_type"] == ctype]
        if len(data) > 0:
            ax.plot(data["week"], data["sentiment_score"], marker="o", label=ctype, color=color)
    ax.set_xlabel("Week")
    ax.set_ylabel("Average Sentiment Score")
    ax.set_title("Sentiment Trend by Call Type")
    ax.legend()
    ax.set_ylim(0, 5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_chart(fig, "04_sentiment_trend_by_type.png")
    plt.close(fig)

    # Negative sentiment percentage trend
    fig, ax = plt.subplots(figsize=(12, 6))
    for ctype, color in [("support", "orange"), ("external", "green"), ("internal", "blue")]:
        data = weekly[weekly["call_type"] == ctype]
        if len(data) > 0:
            ax.plot(data["week"], data["negative_pct"], marker="o", label=ctype, color=color)
    ax.set_xlabel("Week")
    ax.set_ylabel("Negative Sentiment (%)")
    ax.set_title("Negative Sentiment Trend by Call Type")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_chart(fig, "04_negative_sentiment_trend.png")
    plt.close(fig)

    # Sentiment score distribution by call type (box plot)
    fig, ax = plt.subplots(figsize=(8, 5))
    data_to_plot = [df_typed[df_typed["call_type"] == ct]["sentiment_score"].values for ct in ["support", "external", "internal"]]
    ax.boxplot(data_to_plot, tick_labels=["support", "external", "internal"])
    ax.set_ylabel("Sentiment Score")
    ax.set_title("Sentiment Score Distribution by Call Type")
    save_chart(fig, "04_sentiment_boxplot.png")
    plt.close(fig)

    # Overall sentiment stacked bar by type
    sentiment_by_type = df_typed.groupby(["call_type", "overall_sentiment"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sentiment_by_type.plot(kind="bar", stacked=True, ax=ax, colormap="RdYlGn")
    ax.set_xlabel("Call Type")
    ax.set_ylabel("Number of Calls")
    ax.set_title("Overall Sentiment by Call Type")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    save_chart(fig, "04_sentiment_stacked_by_type.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Sentiment by Topic cross-tab
    # ------------------------------------------------------------------
    print("\n--- Sentiment by Topic Cross-Analysis ---")
    topics_path = OUTPUT_DIR / "topics.json"
    topic_cross = {}
    problem_zones = []
    strong_zones = []

    if topics_path.exists():
        import json
        topics_data = json.loads(topics_path.read_text())
        assignments = topics_data.get("assignments", [])
        df_topics = pd.DataFrame(assignments)[["meeting_id", "topic_name"]]
        df_merged = df_typed.merge(df_topics, on="meeting_id", how="left")
        df_merged["topic_name"] = df_merged["topic_name"].fillna("Noise")

        cross = df_merged.groupby(["call_type", "topic_name"]).agg({
            "sentiment_score": "mean",
            "negative_pct": "mean",
        }).round(2).reset_index()

        for ctype in ["support", "external", "internal"]:
            ctype_rows = cross[cross["call_type"] == ctype]
            ctype_rows = ctype_rows[ctype_rows["topic_name"] != "Noise"]
            if len(ctype_rows) > 0:
                topic_cross[ctype] = {}
                for _, row in ctype_rows.iterrows():
                    topic_cross[ctype][row["topic_name"]] = {
                        "avg_sentiment": float(row["sentiment_score"]),
                        "negative_pct": float(row["negative_pct"]),
                    }
                # Problem zone: lowest sentiment
                worst = ctype_rows.loc[ctype_rows["sentiment_score"].idxmin()]
                problem_zones.append({
                    "call_type": ctype,
                    "topic": worst["topic_name"],
                    "sentiment": float(worst["sentiment_score"]),
                    "why": f"Lowest sentiment in {ctype} calls.",
                })
                # Strong zone: highest sentiment
                best = ctype_rows.loc[ctype_rows["sentiment_score"].idxmax()]
                strong_zones.append({
                    "call_type": ctype,
                    "topic": best["topic_name"],
                    "sentiment": float(best["sentiment_score"]),
                    "why": f"Highest sentiment in {ctype} calls.",
                })
                print(f"  {ctype}: worst={worst['topic_name']} ({worst['sentiment_score']}), best={best['topic_name']} ({best['sentiment_score']})")
    else:
        print("  topics.json not found. Skipping cross-analysis.")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_csv(df, "04_sentiment_details.csv")
    save_csv(weekly, "04_sentiment_weekly.csv")
    save_json({
        "by_type": by_type.to_dict(),
        "topic_cross": topic_cross,
        "problem_zones": problem_zones,
        "strong_zones": strong_zones,
        "total_calls_analyzed": len(df),
        "date_range": {"min": str(df["date"].min().date()), "max": str(df["date"].max().date())},
    }, "04_sentiment_stats.json")

    print("\n" + "=" * 60)
    print("04 SENTIMENT: Done")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
