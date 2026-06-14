"""
ppt_data.py

Load and validate all analysis outputs for the PowerPoint generator.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


def get_output_dir() -> Path:
    from utils import OUTPUT_DIR
    return OUTPUT_DIR


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def fmt_num(value, decimals: int = 1, fallback: str = "N/A") -> str:
    try:
        if value is None:
            return fallback
        return str(round(float(value), decimals))
    except (TypeError, ValueError):
        return fallback


@dataclass
class SentimentByType:
    support_score: float | None = None
    external_score: float | None = None
    internal_score: float | None = None
    support_neg: float | None = None
    external_neg: float | None = None
    internal_neg: float | None = None


@dataclass
class RiskDistribution:
    low: int = 0
    medium: int = 0
    high: int = 0


@dataclass
class PresentationData:
    output_dir: Path

    # Explore
    total_calls: int = 0
    date_min: str = "N/A"
    date_max: str = "N/A"
    duration_min: float | None = None
    duration_max: float | None = None
    duration_mean: float | None = None
    overall_sentiment_counts: dict[str, int] = field(default_factory=dict)
    avg_sentiment: float | None = None

    # Call types
    support_count: int = 0
    external_count: int = 0
    internal_count: int = 0

    # Sentiment
    sentiment: SentimentByType = field(default_factory=SentimentByType)
    sentiment_interpretation: dict = field(default_factory=dict)
    topic_cross: dict = field(default_factory=dict)
    category_sentiment: dict = field(default_factory=dict)
    problem_zones: list[dict] = field(default_factory=list)
    strong_zones: list[dict] = field(default_factory=list)
    watch_zone: dict = field(default_factory=dict)

    # Topics
    topic_method: str = "unknown"
    hdbscan_score: float | None = None
    kmeans_score: float | None = None
    hdbscan_noise: int = 0
    clusters: dict[str, dict] = field(default_factory=dict)
    topic_by_call_type: dict = field(default_factory=dict)
    business_taxonomy: dict = field(default_factory=dict)

    # Churn
    risk_distribution: RiskDistribution = field(default_factory=RiskDistribution)
    high_risk_calls: list[dict] = field(default_factory=list)
    medium_risk_calls: list[dict] = field(default_factory=list)
    churn_narratives: list[dict] = field(default_factory=list)
    churn_signals: dict[str, int] = field(default_factory=dict)

    # Features
    feature_keywords: dict[str, int] = field(default_factory=dict)
    feature_samples: dict[str, list[dict]] = field(default_factory=dict)
    feature_callouts: list[dict] = field(default_factory=list)

    # Escalations / Action items
    action_items: dict[str, dict] = field(default_factory=dict)
    carry_forward_actions: dict[str, dict] = field(default_factory=dict)
    carry_forward_total: int = 0
    escalation_chains: list[dict] = field(default_factory=list)

    # Recommendations
    recommendations: list[dict] = field(default_factory=list)

    # Warnings
    warnings: list[str] = field(default_factory=list)

    def add_warning(self, msg: str):
        self.warnings.append(msg)
        print(f"  WARNING (PPT data): {msg}")


def validate_ppt_data(data: PresentationData) -> list[str]:
    warnings = []
    risk_total = data.risk_distribution.low + data.risk_distribution.medium + data.risk_distribution.high
    if risk_total != data.total_calls and data.total_calls > 0:
        warnings.append(f"Risk distribution sums to {risk_total}, expected {data.total_calls}")
    if len(data.high_risk_calls) != data.risk_distribution.high:
        warnings.append(f"High risk list ({len(data.high_risk_calls)}) != count ({data.risk_distribution.high})")
    return warnings


def infer_call_type(title: str) -> str:
    """Infer call type from title when CSV lookup is unavailable."""
    t = title.lower()
    if any(k in t for k in ["support case", "detect outage", "incident:", "escalation:", "war room", "remediation"]):
        return "support"
    if any(k in t for k in ["all hands", "standup", "sprint", "retro", "planning", "q2 roadmap", "q1 business review", "post-incident review"]):
        return "internal"
    return "external"


def correct_topic_name(name: str, keywords) -> str:
    """
    Keyword-based sanity check for topic names.
    With a weak LLM/mock provider, names can be mismatched to keywords.
    Uses a simple scoring system; renames only when evidence is strong.
    """
    if not isinstance(keywords, (list, tuple)):
        keywords = []
    ktext = " ".join(str(k) for k in keywords).lower()

    signal_groups = {
        "Identity & Access Management": ["mfa", "sso", "saml", "identity", "session", "provisioning", "okta"],
        "Incident Response & Reliability": ["failure", "outage", "incident", "escalation", "hours", "event", "dashboard down", "data gaps"],
        "Compliance & Certification": ["hipaa", "comply", "soc", "iso", "audit", "pci", "gdpr", "reporting"],
        "Engineering & Sprint Planning": ["sprint", "qa", "standup", "retro", "engineering", "deployment", "release"],
        "Sales & Renewals": ["renewal", "renew", "account review", "multi-year", "qbr"],
    }

    scores = {topic: sum(1 for w in words if w in ktext) for topic, words in signal_groups.items()}
    best_topic, best_score = max(scores.items(), key=lambda x: x[1])

    # Only override if we have at least 2 matching signals and current name is not already close
    if best_score >= 2:
        current_lower = name.lower()
        best_lower = best_topic.lower()
        # If current name already contains the core concept, keep it
        # Require the *specific* core word(s) of the best topic to be present.
        # This prevents partial matches like "Incident Response & Outages" from
        # being kept when the evidence points to "Incident Response & Reliability".
        key_concepts = {
            "Identity & Access Management": ["identity", "access"],
            "Incident Response & Reliability": ["reliability"],
            "Compliance & Certification": ["compliance", "certification", "audit"],
            "Engineering & Sprint Planning": ["engineering", "sprint"],
            "Sales & Renewals": ["sales", "renewal"],
        }
        if not any(c in current_lower for c in key_concepts.get(best_topic, [])):
            return best_topic

    return name


def load_presentation_data(output_dir: Path | None = None) -> PresentationData:
    if output_dir is None:
        output_dir = get_output_dir()

    data = PresentationData(output_dir=output_dir)

    # Explore
    explore = load_json(output_dir / "01_explore_stats.json")
    data.total_calls = explore.get("total_calls", 0)
    data.date_min = explore.get("date_range", {}).get("min", "N/A")
    data.date_max = explore.get("date_range", {}).get("max", "N/A")
    duration = explore.get("duration", {})
    data.duration_min = duration.get("min")
    data.duration_max = duration.get("max")
    data.duration_mean = duration.get("mean")
    data.overall_sentiment_counts = explore.get("overall_sentiment_counts", {})
    data.avg_sentiment = explore.get("sentiment", {}).get("mean")

    # Call types
    df_types = load_csv(output_dir / "02_call_types.csv")
    if df_types is not None and "final_label" in df_types.columns:
        counts = df_types["final_label"].value_counts().to_dict()
        data.support_count = int(counts.get("support", 0))
        data.external_count = int(counts.get("external", 0))
        data.internal_count = int(counts.get("internal", 0))
    else:
        data.add_warning("02_call_types.csv not found or missing final_label column")

    # Sentiment
    sentiment = load_json(output_dir / "04_sentiment_stats.json")
    by_type = sentiment.get("by_type", {})
    data.sentiment.support_score = by_type.get("sentiment_score", {}).get("support")
    data.sentiment.external_score = by_type.get("sentiment_score", {}).get("external")
    data.sentiment.internal_score = by_type.get("sentiment_score", {}).get("internal")
    data.sentiment.support_neg = by_type.get("negative_pct", {}).get("support")
    data.sentiment.external_neg = by_type.get("negative_pct", {}).get("external")
    data.sentiment.internal_neg = by_type.get("negative_pct", {}).get("internal")
    data.sentiment_interpretation = sentiment.get("interpretation", {})
    data.topic_cross = sentiment.get("topic_cross", {})
    data.category_sentiment = sentiment.get("category_sentiment", {})
    data.problem_zones = sentiment.get("problem_zones", [])
    data.strong_zones = sentiment.get("strong_zones", [])
    data.watch_zone = sentiment.get("watch_zone", {})

    # Topics
    topics = load_json(output_dir / "topics.json")
    data.topic_method = topics.get("method", "unknown")
    data.hdbscan_score = topics.get("comparison", {}).get("hdbscan", {}).get("score")
    data.kmeans_score = topics.get("comparison", {}).get("kmeans", {}).get("score")
    data.hdbscan_noise = topics.get("comparison", {}).get("hdbscan", {}).get("n_noise", 0)
    raw_clusters = topics.get("clusters", {})
    data.clusters = {}
    for cid, info in raw_clusters.items():
        corrected = dict(info)
        corrected["name"] = correct_topic_name(info.get("name", f"Cluster {cid}"), info.get("keywords", []))
        data.clusters[cid] = corrected
    data.topic_by_call_type = topics.get("topic_by_call_type", {})
    data.business_taxonomy = topics.get("business_taxonomy", {})

    # Churn
    churn = load_json(output_dir / "05_churn_scores.json")
    risk_dist = churn.get("risk_distribution", {})
    data.risk_distribution.low = int(risk_dist.get("Low", 0))
    data.risk_distribution.medium = int(risk_dist.get("Medium", 0))
    data.risk_distribution.high = int(risk_dist.get("High", 0))
    data.high_risk_calls = churn.get("high_risk_calls", [])
    data.medium_risk_calls = churn.get("medium_risk_calls", [])
    # Enrich churn narratives with call type
    call_type_lookup = {}
    df_types = load_csv(output_dir / "02_call_types.csv")
    if df_types is not None and "meeting_id" in df_types.columns and "final_label" in df_types.columns:
        call_type_lookup = dict(zip(df_types["meeting_id"].astype(str), df_types["final_label"]))

    raw_narratives = churn.get("churn_narratives", [])
    data.churn_narratives = []
    for account in raw_narratives:
        enriched = dict(account)
        meeting_id = str(enriched.get("meeting_id", ""))
        enriched["call_type"] = call_type_lookup.get(meeting_id, infer_call_type(enriched.get("title", "")))
        data.churn_narratives.append(enriched)

    data.churn_signals = churn.get("signals", {})

    # Features
    features = load_json(output_dir / "05_feature_requests.json")
    data.feature_keywords = features.get("top_keywords", {})
    data.feature_samples = features.get("feature_samples", {})
    data.feature_callouts = features.get("feature_callouts", [])

    # Escalations
    esc = load_json(output_dir / "05_escalations.json")
    data.action_items = esc.get("action_items_by_type", {})
    data.carry_forward_actions = esc.get("carry_forward_actions", {})
    data.carry_forward_total = esc.get("carry_forward_total", 0)
    data.escalation_chains = esc.get("chains", [])

    # Recommendations
    rec = load_json(output_dir / "05_recommendations.json")
    data.recommendations = rec.get("recommendations", [])

    data.warnings.extend(validate_ppt_data(data))
    return data
