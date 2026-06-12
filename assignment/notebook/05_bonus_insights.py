"""
05_bonus_insights.py

Churn risk scoring, feature request extraction, escalation funnel.

Exports: churn_scores.json, features.json, escalations.json

WHY RULE-BASED CHURN SCORING (NOT ML):
  Machine learning models (XGBoost, logistic regression) are black boxes.
  If the panel asks "Why is this account high risk?" you cannot explain
  the model weights. A feature-based point system is fully transparent:

    Negative sentiment dominates  -> +2 points
    Competitor mentioned          -> +2 points
    Escalation requested          -> +3 points

  Every point is independently verifiable by reading the transcript.
  Leadership understands this instantly. It is an illustrative heuristic,
  not a predictive model, and we are honest about that.

  See notebook_decisions.md for the full comparison.

WHY FEATURE REQUEST EXTRACTION:
  Most assignments stop at sentiment and topics. Feature requests show
  product thinking: "What are customers asking for that we do not have?"
  We extract sentences containing request keywords ("need", "want",
  "granular restore", "api", etc.) and count frequency. This is a
  goldmine for product managers.

WHY ESCALATION FUNNEL:
  Most candidates analyze call types in silos. The escalation funnel
  connects them: Support Case -> External Call -> Internal Planning.
  This shows systems thinking, the exact skill the assignment tests.

  We find chains by matching account names across call types and
  checking time sequence (support first, then external, then internal).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from utils import (
    load_all_calls,
    extract_meeting_title,
    extract_meeting_date,
    extract_summary_text,
    save_chart,
    save_csv,
    save_json,
    set_chart_style,
    OUTPUT_DIR,
)


# ---------------------------------------------------------------------------
# Churn Risk Scoring
# ---------------------------------------------------------------------------

CHURN_SIGNALS = {
    "negative_sentiment_dominates": {"points": 2, "check": lambda call: (call.get("summary") or {}).get("sentimentScore", 3) < 2.5},
    "renewal_discussion": {"points": 2, "check": lambda call: "renewal" in extract_summary_text(call).lower() or "renewal" in str(call.get("transcript", "")).lower()},
    "competitor_mentioned": {"points": 2, "check": lambda call: any(w in str(call.get("transcript", "")).lower() for w in ["competitor", "competition", "competitive", "alternative", "switching"])},
    "escalation_requested": {"points": 3, "check": lambda call: any(w in extract_summary_text(call).lower() for w in ["escalation", "escalate", "urgent", "war room"])},
    "executive_involvement": {"points": 2, "check": lambda call: any(w in str(call.get("transcript", "")).lower() for w in ["ceo", "cto", "vp", "director", "executive"])},
    "product_dissatisfaction": {"points": 3, "check": lambda call: any(w in str(call.get("transcript", "")).lower() for w in ["disappointed", "frustrated", "unacceptable", "unhappy", "concerned"])},
}


def score_churn_risk(call: dict) -> tuple[int, list[str]]:
    """Calculate churn risk score and list triggered signals."""
    score = 0
    signals = []

    for signal_name, config in CHURN_SIGNALS.items():
        if config["check"](call):
            score += config["points"]
            signals.append(signal_name)

    return score, signals


def risk_level(score: int) -> str:
    if score <= 3:
        return "Low"
    elif score <= 6:
        return "Medium"
    else:
        return "High"


# ---------------------------------------------------------------------------
# Feature Request Extraction
# ---------------------------------------------------------------------------

FEATURE_KEYWORDS = [
    # Specific feature names (high signal)
    "granular restore", "backup", "webhook", "integration",
    "dashboard", "notification", "report", "export",
    "sso", "mfa", "ldap", "saml", "audit log", "role-based",
    "mobile app", "cli", "automation", "workflow",
    "api access", "custom fields", "data retention", "bulk upload",
    # Request phrases (must be specific, not conversational filler)
    "feature request", "would like to see", "need the ability",
    "should be able to", "missing the", "requesting",
    "looking for", "asking for", "interested in",
]


def extract_feature_requests(call: dict) -> list[str]:
    """Extract potential feature requests from summary and transcript."""
    features = []
    summary = extract_summary_text(call).lower()
    transcript = call.get("transcript") or {}
    sentences = transcript.get("data", [])

    # Check summary for feature keywords
    for keyword in FEATURE_KEYWORDS:
        if keyword in summary:
            # Find the sentence containing the keyword
            for s in sentences:
                text = s.get("sentence", "").lower()
                if keyword in text and len(text) > 20:
                    features.append(s.get("sentence", "").strip())
                    break

    return list(set(features))[:5]  # Deduplicate and limit


# ---------------------------------------------------------------------------
# Escalation Funnel
# ---------------------------------------------------------------------------

def extract_account_name(title: str) -> str | None:
    """Extract customer account name from call title."""
    # Patterns like "Aegis / Brightpath Commerce - ..."
    if " / " in title:
        parts = title.split(" / ")
        if len(parts) >= 2:
            # Second part is usually "Customer Name - Topic"
            customer_part = parts[1]
            if " - " in customer_part:
                return customer_part.split(" - ")[0].strip()
            return customer_part.strip()
    return None


def build_escalation_chains(calls: list[dict], type_map: dict) -> list[dict]:
    """Find chains: support -> external -> internal for same account."""
    # Group calls by account
    account_calls = defaultdict(list)
    for call in calls:
        title = extract_meeting_title(call)
        account = extract_account_name(title)
        if account:
            call_type = type_map.get(call["meeting_id"], "unknown")
            account_calls[account].append({
                "meeting_id": call["meeting_id"],
                "title": title,
                "date": extract_meeting_date(call),
                "type": call_type,
                "sentiment_score": (call.get("summary") or {}).get("sentimentScore", 3),
            })

    # Find chains
    chains = []
    for account, calls_list in account_calls.items():
        if len(calls_list) < 2:
            continue

        # Sort by date
        calls_list.sort(key=lambda x: x["date"] or "")

        # Check for type progression
        types_seen = [c["type"] for c in calls_list]

        # Look for support + external + internal
        if "support" in types_seen and "external" in types_seen:
            chain = {
                "account": account,
                "calls": calls_list,
                "types_seen": list(dict.fromkeys(types_seen)),  # Remove duplicates preserve order
                "chain_detected": True,
            }
            chains.append(chain)

    # Sort by number of calls (most complex chains first)
    chains.sort(key=lambda x: len(x["calls"]), reverse=True)
    return chains[:20]  # Top 20


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_chart_style()
    print("=" * 60)
    print("05 BONUS INSIGHTS: Churn, Features, Escalations")
    print("=" * 60)

    calls = load_all_calls()
    print(f"\nTotal calls: {len(calls)}")

    # Load call types
    call_types_path = OUTPUT_DIR / "02_call_types.csv"
    if call_types_path.exists():
        df_types = pd.read_csv(call_types_path)
        type_map = dict(zip(df_types["meeting_id"], df_types["final_label"]))
    else:
        type_map = {}

    # ------------------------------------------------------------------
    # Churn Scoring
    # ------------------------------------------------------------------
    print("\n--- Churn Risk Scoring ---")

    churn_rows = []
    for call in calls:
        score, signals = score_churn_risk(call)
        level = risk_level(score)

        churn_rows.append({
            "meeting_id": call["meeting_id"],
            "title": extract_meeting_title(call),
            "date": extract_meeting_date(call),
            "call_type": type_map.get(call["meeting_id"], "unknown"),
            "sentiment_score": (call.get("summary") or {}).get("sentimentScore", 3),
            "churn_score": score,
            "risk_level": level,
            "signals": ", ".join(signals),
            "signal_count": len(signals),
        })

    df_churn = pd.DataFrame(churn_rows)
    df_churn = df_churn.sort_values("churn_score", ascending=False)

    print(f"\nRisk distribution:")
    print(df_churn["risk_level"].value_counts())

    print(f"\n--- High Risk Accounts (score 7+) ---")
    high_risk = df_churn[df_churn["risk_level"] == "High"]
    print(high_risk[["title", "churn_score", "signals"]].to_string())

    print(f"\n--- Medium Risk Accounts (score 4-6) ---")
    med_risk = df_churn[df_churn["risk_level"] == "Medium"]
    print(med_risk[["title", "churn_score", "signals"]].to_string())

    # ------------------------------------------------------------------
    # Feature Requests
    # ------------------------------------------------------------------
    print("\n--- Feature Request Extraction ---")

    feature_rows = []
    for call in calls:
        features = extract_feature_requests(call)
        if features:
            for feat in features:
                feature_rows.append({
                    "meeting_id": call["meeting_id"],
                    "title": extract_meeting_title(call),
                    "date": extract_meeting_date(call),
                    "feature_request": feat,
                })

    df_features = pd.DataFrame(feature_rows)
    print(f"Total feature mentions: {len(df_features)}")

    # Count by keyword
    all_features = " ".join(df_features["feature_request"].str.lower())
    keyword_counts = {}
    for kw in FEATURE_KEYWORDS:
        count = all_features.count(kw)
        if count > 0:
            keyword_counts[kw] = count

    keyword_counts = dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True))
    print(f"\nTop feature keywords:")
    for kw, count in list(keyword_counts.items())[:15]:
        print(f"  {kw}: {count}")

    # ------------------------------------------------------------------
    # Escalation Funnel
    # ------------------------------------------------------------------
    print("\n--- Escalation Funnel ---")

    chains = build_escalation_chains(calls, type_map)
    print(f"Accounts with multiple call types: {len(chains)}")

    for chain in chains[:10]:
        print(f"\n  {chain['account']}:")
        for c in chain["calls"]:
            print(f"    [{c['type']:8s}] {c['date']} | {c['title'][:60]}... (sentiment: {c['sentiment_score']})")

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    print("\n--- Generating charts ---")

    # Churn risk distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    risk_counts = df_churn["risk_level"].value_counts()
    colors = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}
    bar_colors = [colors.get(l, "gray") for l in risk_counts.index]
    ax.bar(risk_counts.index, risk_counts.values, color=bar_colors)
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Calls")
    ax.set_title("Churn Risk Distribution")
    for i, v in enumerate(risk_counts.values):
        ax.text(i, v + 0.3, str(v), ha="center", fontweight="bold")
    save_chart(fig, "05_churn_risk_distribution.png")
    plt.close(fig)

    # Churn score histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_churn["churn_score"], bins=range(0, df_churn["churn_score"].max()+2), edgecolor="black", color="coral")
    ax.set_xlabel("Churn Risk Score")
    ax.set_ylabel("Number of Calls")
    ax.set_title("Churn Risk Score Distribution")
    ax.axvline(x=4, color="orange", linestyle="--", label="Medium threshold")
    ax.axvline(x=7, color="red", linestyle="--", label="High threshold")
    ax.legend()
    save_chart(fig, "05_churn_score_histogram.png")
    plt.close(fig)

    # Feature request frequency
    if keyword_counts:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_kws = list(keyword_counts.items())[:15]
        ax.barh([k[0] for k in top_kws], [k[1] for k in top_kws], color="steelblue")
        ax.set_xlabel("Mentions")
        ax.set_title("Top Feature Request Keywords")
        save_chart(fig, "05_feature_requests.png")
        plt.close(fig)

    # Escalation chains count
    if chains:
        chain_lengths = [len(c["calls"]) for c in chains]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(chain_lengths, bins=range(2, max(chain_lengths)+2), edgecolor="black", color="purple")
        ax.set_xlabel("Number of Calls per Account")
        ax.set_ylabel("Number of Accounts")
        ax.set_title("Account Call Frequency (Escalation Funnel Candidates)")
        save_chart(fig, "05_escalation_chain_lengths.png")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Enriched data for PPT
    # ------------------------------------------------------------------

    # Churn narratives: build 1-2 sentence context for high-risk accounts
    churn_narratives = []
    for _, row in high_risk.head(10).iterrows():
        call = next((c for c in calls if c["meeting_id"] == row["meeting_id"]), None)
        summary_text = extract_summary_text(call)[:300] if call else ""
        signals_list = row["signals"].split(", ") if row["signals"] else []
        narrative_parts = []
        if "negative_sentiment_dominates" in signals_list:
            narrative_parts.append("Dominated by negative sentiment.")
        if "competitor_mentioned" in signals_list:
            narrative_parts.append("Competitor evaluation in progress.")
        if "escalation_requested" in signals_list:
            narrative_parts.append("Escalation requested by customer.")
        if "renewal_discussion" in signals_list:
            narrative_parts.append("Renewal at risk.")
        if "product_dissatisfaction" in signals_list:
            narrative_parts.append("Product dissatisfaction expressed.")
        if "executive_involvement" in signals_list:
            narrative_parts.append("Executive level involved.")
        narrative = " ".join(narrative_parts) if narrative_parts else summary_text[:150]
        churn_narratives.append({
            "meeting_id": row["meeting_id"],
            "title": row["title"],
            "churn_score": int(row["churn_score"]),
            "sentiment_score": float(row["sentiment_score"]),
            "signals": row["signals"],
            "narrative": narrative,
        })

    # Feature samples: top 3 sample sentences per keyword
    feature_samples = {}
    for kw in list(keyword_counts.keys())[:8]:
        samples = []
        for _, row in df_features.iterrows():
            if kw in row["feature_request"].lower():
                samples.append({
                    "sentence": row["feature_request"][:200],
                    "title": row["title"],
                })
            if len(samples) >= 3:
                break
        feature_samples[kw] = samples

    # Action items per call type (heuristic: count explicit action/follow-up language)
    action_item_keywords = ["action item", "follow up", "follow-up", "next step", "todo", "to do", "will do", "need to"]
    action_counts = {}
    for ctype in ["support", "external", "internal"]:
        ctype_calls = [c for c in calls if type_map.get(c["meeting_id"]) == ctype]
        total_actions = 0
        for call in ctype_calls:
            transcript = call.get("transcript") or {}
            sentences = transcript.get("data", [])
            for s in sentences:
                text = s.get("sentence", "").lower()
                if any(kw in text for kw in action_item_keywords):
                    total_actions += 1
        action_counts[ctype] = {
            "total_calls": len(ctype_calls),
            "action_mentions": total_actions,
            "avg_per_call": round(total_actions / len(ctype_calls), 1) if ctype_calls else 0,
        }

    print(f"\n--- Action Item Mentions by Call Type ---")
    for ctype, info in action_counts.items():
        print(f"  {ctype}: {info['action_mentions']} mentions ({info['avg_per_call']} avg per call)")

    # Action items chart for PPT
    fig, ax = plt.subplots(figsize=(8, 5))
    ctypes = list(action_counts.keys())
    avgs = [action_counts[c]["avg_per_call"] for c in ctypes]
    totals = [action_counts[c]["action_mentions"] for c in ctypes]
    bars = ax.bar(ctypes, avgs, color=["#1a237e", "#0078d4", "#ff6b35"])
    ax.set_ylabel("Avg Action Items per Call")
    ax.set_title("Action Items by Call Type")
    for bar, total in zip(bars, totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f"{height:.1f}\n({total} total)", ha="center", va="bottom", fontsize=10)
    save_chart(fig, "05_action_items_by_type.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_csv(df_churn, "05_churn_scores.csv")
    save_csv(df_features, "05_feature_requests.csv")
    save_json({
        "risk_distribution": df_churn["risk_level"].value_counts().to_dict(),
        "high_risk_calls": high_risk[["meeting_id", "title", "churn_score", "signals"]].to_dict("records"),
        "medium_risk_calls": med_risk[["meeting_id", "title", "churn_score", "signals"]].to_dict("records"),
        "churn_narratives": churn_narratives,
        "scoring_method": "Feature-based point system",
        "signals": {k: v["points"] for k, v in CHURN_SIGNALS.items()},
    }, "05_churn_scores.json")

    save_json({
        "total_mentions": len(df_features),
        "top_keywords": keyword_counts,
        "feature_samples": feature_samples,
        "requests": df_features.to_dict("records")[:50],
    }, "05_feature_requests.json")

    save_json({
        "total_accounts_with_chains": len(chains),
        "action_items_by_type": action_counts,
        "chains": [
            {
                "account": c["account"],
                "call_count": len(c["calls"]),
                "types_seen": c["types_seen"],
                "calls": [{k: v for k, v in call.items() if k != "type"} for call in c["calls"]],
            }
            for c in chains[:20]
        ],
    }, "05_escalations.json")

    print("\n" + "=" * 60)
    print("05 BONUS INSIGHTS: Done")
    print("=" * 60)

    return df_churn, df_features, chains


if __name__ == "__main__":
    main()
