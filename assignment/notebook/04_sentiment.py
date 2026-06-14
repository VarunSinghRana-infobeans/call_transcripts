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
    fig, ax = create_chart_fig("04_sentiment_trend_by_type.png")
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
    fig, ax = create_chart_fig("04_negative_sentiment_trend.png")
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
    fig, ax = create_chart_fig("04_sentiment_boxplot.png")
    data_to_plot = [df_typed[df_typed["call_type"] == ct]["sentiment_score"].values for ct in ["support", "external", "internal"]]
    ax.boxplot(data_to_plot, tick_labels=["support", "external", "internal"])
    ax.set_ylabel("Sentiment Score")
    ax.set_title("Sentiment Score Distribution by Call Type")
    save_chart(fig, "04_sentiment_boxplot.png")
    plt.close(fig)

    # Overall sentiment stacked bar by type
    sentiment_by_type = df_typed.groupby(["call_type", "overall_sentiment"]).size().unstack(fill_value=0)
    fig, ax = create_chart_fig("04_sentiment_stacked_by_type.png")
    sentiment_by_type.plot(kind="bar", stacked=True, ax=ax, colormap="RdYlGn")
    ax.set_xlabel("Call Type")
    ax.set_ylabel("Number of Calls")
    ax.set_title("Overall Sentiment by Call Type")
    # Legend on the right; the target figure is wide enough to accommodate it.
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    save_chart(fig, "04_sentiment_stacked_by_type.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Sentiment scale interpretation
    # ------------------------------------------------------------------
    def interpret_score(score: float) -> str:
        if score >= 4.5:
            return "Very positive"
        if score >= 3.5:
            return "Positive"
        if score >= 2.5:
            return "Neutral / mixed"
        if score >= 1.5:
            return "Negative"
        return "Very negative"

    scale = {
        "1.0-1.9": "Very negative — urgent intervention",
        "2.0-2.9": "Negative — issue driving dissatisfaction",
        "3.0-3.9": "Neutral / mixed — resolved or ambiguous",
        "4.0-4.9": "Positive — relationship or product strength",
        "5.0": "Very positive — strong advocacy moment",
    }

    overall_mean = float(df["sentiment_score"].mean())
    sentiment_interpretation = {
        "scale": scale,
        "overall_label": interpret_score(overall_mean),
        "overall_score": round(overall_mean, 2),
    }

    # ------------------------------------------------------------------
    # Sentiment by Business Taxonomy cross-tab
    # ------------------------------------------------------------------
    print("\n--- Sentiment by Business Taxonomy Cross-Analysis ---")
    topics_path = OUTPUT_DIR / "topics.json"
    topic_cross = {}
    problem_zones = []
    strong_zones = []

    if topics_path.exists():
        import json
        topics_data = json.loads(topics_path.read_text())
        biz_assignments = topics_data.get("business_taxonomy", {}).get("assignments", [])
        if biz_assignments:
            df_topics = pd.DataFrame(biz_assignments)[["meeting_id", "primary"]]
            df_topics = df_topics.rename(columns={"primary": "topic_name"})
        else:
            assignments = topics_data.get("assignments", [])
            df_topics = pd.DataFrame(assignments)[["meeting_id", "topic_name"]]
        df_merged = df_typed.merge(df_topics, on="meeting_id", how="left")
        df_merged["topic_name"] = df_merged["topic_name"].fillna("Other")

        cross = df_merged.groupby(["call_type", "topic_name"]).agg({
            "sentiment_score": ["mean", "count"],
            "negative_pct": "mean",
        }).round(2).reset_index()
        cross.columns = ["call_type", "topic_name", "sentiment_score", "call_count", "negative_pct"]

        # Build a lookup of example titles per zone for richer "why" text
        title_lookup = dict(zip(df["meeting_id"], df["title"]))

        # Topic-level context so cards do not all read identically
        ZONE_INSIGHTS = {
            "Platform Reliability": {
                "low": "Incident and outage calls directly impact customers. SLA pressure and repeated downtime erode trust.",
                "high": "Stable reliability conversations (post-fix reviews) show customers respond well to transparency.",
            },
            "Threat Detection": {
                "low": "Alert tuning gaps and false positives create security anxiety; customers question coverage.",
                "high": "Detection reviews that resolve tuning issues are received positively by security teams.",
            },
            "Billing & Contracts": {
                "low": "Seat overages, invoice disputes and renewal terms generate pushback and pricing friction.",
                "high": "Clean renewal and expansion conversations reflect strong value communication.",
            },
            "Compliance & Audit": {
                "low": "Audit evidence gaps and auditor tooling mismatches slow down compliance workflows.",
                "high": "Compliance prep and certification calls score well; customers value the audit support.",
            },
            "Identity & Access": {
                "low": "SSO, MFA and LDAP failures block users and create urgent, high-friction support cases.",
                "high": "Successful identity rollouts and provisioning reviews land as product strengths.",
            },
            "Churn & Risk": {
                "low": "Escalations and competitor evaluations signal renewal conversations under pressure.",
                "high": "Recovery check-ins after issues can rebuild confidence when handled proactively.",
            },
            "Customer Success": {
                "low": "Onboarding delays or adoption gaps can turn success calls negative quickly.",
                "high": "Onboarding kickoffs and QBRs are clear relationship strengths with no escalations.",
            },
            "Integrations & API": {
                "low": "Connector timeouts, rate limits and sync failures drive integration frustration.",
                "high": "API and webhook rollouts that work as advertised reinforce platform value.",
            },
            "Product & Roadmap": {
                "low": "Roadmap misalignment or missing features can create disappointment in product conversations.",
                "high": "Roadmap alignment calls where features land well score positively with customers.",
            },
            "Internal Operations": {
                "low": "Internal planning calls rarely score low unless incidents are being escalated.",
                "high": "Sprint retros and postmortems run constructive, neutral-to-positive tone.",
            },
        }

        def zone_why(zone_rows: pd.DataFrame, direction: str, topic: str) -> str:
            if zone_rows.empty:
                return f"{direction.capitalize()} sentiment zone."
            rep = zone_rows.sort_values("sentiment_score", ascending=(direction == "low")).head(3)
            titles = [title_lookup.get(mid, "") for mid in rep["meeting_id"].tolist()]
            titles = [t for t in titles if t]
            insight = ZONE_INSIGHTS.get(topic, {}).get(direction, "")
            parts = []
            if insight:
                parts.append(insight)
            if titles:
                parts.append(f"Examples: {', '.join(t[:60] for t in titles[:2])}.")
            if not parts:
                return f"{direction.capitalize()} sentiment zone ({len(zone_rows)} calls)."
            return " ".join(parts)

        for ctype in ["support", "external", "internal"]:
            ctype_rows = cross[cross["call_type"] == ctype]
            ctype_rows = ctype_rows[ctype_rows["topic_name"] != "Other"]
            if len(ctype_rows) > 0:
                topic_cross[ctype] = {}
                for _, row in ctype_rows.iterrows():
                    topic_cross[ctype][row["topic_name"]] = {
                        "avg_sentiment": float(row["sentiment_score"]),
                        "negative_pct": float(row["negative_pct"]),
                        "call_count": int(row["call_count"]),
                    }
                # Problem zone: lowest sentiment
                worst_idx = ctype_rows["sentiment_score"].idxmin()
                worst = ctype_rows.loc[worst_idx]
                worst_calls = df_merged[
                    (df_merged["call_type"] == ctype) &
                    (df_merged["topic_name"] == worst["topic_name"])
                ][["meeting_id", "sentiment_score"]]
                problem_zones.append({
                    "call_type": ctype,
                    "topic": worst["topic_name"],
                    "sentiment": float(worst["sentiment_score"]),
                    "call_count": int(worst["call_count"]),
                    "negative_pct": float(worst["negative_pct"]),
                    "why": zone_why(worst_calls, "low", worst["topic_name"]),
                    "label": interpret_score(float(worst["sentiment_score"])),
                })
                # Strong zone: highest sentiment
                best_idx = ctype_rows["sentiment_score"].idxmax()
                best = ctype_rows.loc[best_idx]
                best_calls = df_merged[
                    (df_merged["call_type"] == ctype) &
                    (df_merged["topic_name"] == best["topic_name"])
                ][["meeting_id", "sentiment_score"]]
                strong_zones.append({
                    "call_type": ctype,
                    "topic": best["topic_name"],
                    "sentiment": float(best["sentiment_score"]),
                    "call_count": int(best["call_count"]),
                    "negative_pct": float(best["negative_pct"]),
                    "why": zone_why(best_calls, "high", best["topic_name"]),
                    "label": interpret_score(float(best["sentiment_score"])),
                })
                print(f"  {ctype}: worst={worst['topic_name']} ({worst['sentiment_score']}), best={best['topic_name']} ({best['sentiment_score']})")
    else:
        print("  topics.json not found. Skipping cross-analysis.")

    # ------------------------------------------------------------------
    # Category-level sentiment (enriches the topic slide)
    # ------------------------------------------------------------------
    category_sentiment = {}
    if topics_path.exists() and biz_assignments:
        df_assign = pd.DataFrame(biz_assignments)[["meeting_id", "primary"]]
        df_assign["meeting_id"] = df_assign["meeting_id"].astype(str)
        df_cat = df_typed.merge(df_assign, on="meeting_id", how="left")
        df_cat = df_cat[df_cat["primary"].notna() & (df_cat["primary"] != "Other")]
        cat_overall = df_cat.groupby("primary").agg({
            "sentiment_score": "mean",
            "negative_pct": "mean",
            "meeting_id": "count",
        }).reset_index()
        cat_overall.columns = ["category", "avg_sentiment", "negative_pct", "call_count"]
        for _, row in cat_overall.iterrows():
            cs = {
                "avg_sentiment": round(float(row["avg_sentiment"]), 2),
                "negative_pct": round(float(row["negative_pct"]), 1),
                "call_count": int(row["call_count"]),
                "by_call_type": {},
            }
            for ctype in ["support", "external", "internal"]:
                sub = df_cat[(df_cat["primary"] == row["category"]) & (df_cat["call_type"] == ctype)]
                if len(sub):
                    cs["by_call_type"][ctype] = {
                        "avg_sentiment": round(float(sub["sentiment_score"].mean()), 2),
                        "negative_pct": round(float(sub["negative_pct"].mean()), 1),
                        "call_count": int(len(sub)),
                    }
            category_sentiment[row["category"]] = cs

        # Write category sentiment back into topics.json so the topic slide can use it
        try:
            topics_data = json.loads(topics_path.read_text(encoding="utf-8"))
            biz = topics_data.get("business_taxonomy", {})
            for cat_info in biz.get("top_categories", []):
                cat = cat_info["category"]
                if cat in category_sentiment:
                    cat_info["avg_sentiment"] = category_sentiment[cat]["avg_sentiment"]
                    cat_info["negative_pct"] = category_sentiment[cat]["negative_pct"]
                    cat_info["by_call_type"] = category_sentiment[cat]["by_call_type"]
            topics_path.write_text(json.dumps(topics_data, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            print(f"  Could not update topics.json with category sentiment: {e}")

    # Overall problem/strong zones sorted by sentiment
    problem_zones_sorted = sorted(problem_zones, key=lambda x: x["sentiment"])[:3]
    strong_zones_sorted = sorted(strong_zones, key=lambda x: x["sentiment"], reverse=True)[:3]

    # Pick a caution/watch zone: Churn & Risk if present, else second-worst overall
    watch_zone = None
    churn_risk_zones = [z for z in problem_zones if z["topic"] == "Churn & Risk"]
    if churn_risk_zones:
        watch_zone = sorted(churn_risk_zones, key=lambda x: x["sentiment"])[0]
    elif len(problem_zones_sorted) > 1:
        watch_zone = problem_zones_sorted[1]

    # ------------------------------------------------------------------
    # Heatmap: sentiment score by call type × business taxonomy category
    # ------------------------------------------------------------------
    if topics_path.exists() and biz_assignments:
        # Order rows/columns for readability
        row_order = ["support", "external", "internal"]
        col_order = [
            "Billing & Contracts", "Identity & Access", "Compliance & Audit",
            "Platform Reliability", "Integrations & API", "Customer Success",
            "Threat Detection", "Product & Roadmap", "Churn & Risk", "Internal Operations",
        ]
        pivot = cross.pivot(index="call_type", columns="topic_name", values="sentiment_score")
        pivot = pivot.reindex(index=row_order, columns=col_order)

        fig, ax = create_chart_fig("04_sentiment_heatmap_by_taxonomy.png")
        # Prepare data array with NaN for empty cells
        values = pivot.values
        im = ax.imshow(values, cmap="RdYlGn", aspect="auto", vmin=1, vmax=5)

        # Annotate cells
        for i in range(len(row_order)):
            for j in range(len(col_order)):
                val = values[i, j]
                if not pd.isna(val):
                    text_color = "white" if val < 2.5 or val > 4.0 else "black"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

        ax.set_xticks(range(len(col_order)))
        ax.set_yticks(range(len(row_order)))
        ax.set_xticklabels(col_order, rotation=30, ha="right")
        ax.set_yticklabels(row_order)
        ax.set_title("Sentiment Score by Call Type × Topic Category")
        ax.set_xlabel("Topic Category")
        ax.set_ylabel("Call Type")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Avg Sentiment (1-5)")
        plt.tight_layout()
        save_chart(fig, "04_sentiment_heatmap_by_taxonomy.png")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_csv(df, "04_sentiment_details.csv")
    save_csv(weekly, "04_sentiment_weekly.csv")
    save_json({
        "by_type": by_type.to_dict(),
        "interpretation": sentiment_interpretation,
        "topic_cross": topic_cross,
        "category_sentiment": category_sentiment,
        "problem_zones": problem_zones_sorted,
        "strong_zones": strong_zones_sorted,
        "watch_zone": watch_zone,
        "total_calls_analyzed": len(df),
        "date_range": {"min": str(df["date"].min().date()), "max": str(df["date"].max().date())},
    }, "04_sentiment_stats.json")

    print("\n" + "=" * 60)
    print("04 SENTIMENT: Done")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
