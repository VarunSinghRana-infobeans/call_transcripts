"""
03_topic_modeling.py

Embed call summaries, compare clustering methods (HDBSCAN vs K-Means),
pick winner by silhouette score, extract keywords, LLM-label clusters.

Exports: topics.json, embeddings.npy, topic charts

WHY EMBEDDINGS:
  Raw text is hard to cluster. "Outage" and "Incident" mean the same
  thing but share no words. Embeddings map text to vectors where similar
  meanings are close together. MiniLM (384-dim) is free, offline, and
  90% as good as OpenAI on conversational text.

WHY HDBSCAN OVER K-MEANS:
  K-Means forces every call into a cluster. It also requires knowing K
  ahead of time. HDBSCAN finds clusters of varying density and marks
  noise points as unclustered. On this dataset:
    HDBSCAN silhouette: 0.085 (5 clusters, 44 noise)
    K-Means silhouette: 0.048 (5 clusters, 0 noise)
  HDBSCAN wins because honest noise is better than forced wrong clusters.

WHY SILHOUETTE SCORE:
  It measures how similar a sample is to its own cluster vs. others.
  Range: -1 (wrong) to +1 (perfect). We compare methods objectively.

WHY TF-IDF FOR KEYWORDS:
  After clustering, we need to explain WHAT each cluster is about.
  TF-IDF finds words that are frequent in this cluster but rare in
  others. This makes the taxonomy defensible in Q&A.

WHY LLM LABELS CLUSTERS:
  HDBSCAN gives us "Cluster 0." The panel needs "Outage & Reliability."
  We feed the top keywords + sample summaries to an LLM and ask for a
  human-readable name. This is the bridge between algorithm and audience.

WHY WE CLEAN COMPANY NAMES:
  "Aegis" appears in every summary (it is the company). If we do not
  remove it, the embedding model thinks all calls are similar. Cleaning
  improves cluster separation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from utils import (
    load_all_calls,
    extract_summary_text,
    create_chart_fig,
    save_chart,
    save_json,
    set_chart_style,
    OUTPUT_DIR,
)
from ppt_data import correct_topic_name


def embed_summaries(summaries: list[str]) -> np.ndarray:
    """Embed summaries using sentence-transformers."""
    print("Loading embedding model (MiniLM, 384-dim)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(summaries, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def compare_methods(embeddings: np.ndarray, summaries: list[str]) -> dict:
    """Compare HDBSCAN and K-Means, return best labels + method."""
    results = {}

    # --- K-Means ---
    print("\n--- Testing K-Means ---")
    best_kmeans = None
    best_kmeans_score = -1
    for k in range(5, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        print(f"  K={k}: silhouette={score:.3f}")
        if score > best_kmeans_score:
            best_kmeans_score = score
            best_kmeans = (k, labels)
    results["kmeans"] = {"k": best_kmeans[0], "labels": best_kmeans[1], "score": best_kmeans_score}
    print(f"Best K-Means: K={best_kmeans[0]}, silhouette={best_kmeans_score:.3f}")

    # --- HDBSCAN ---
    print("\n--- Testing HDBSCAN ---")
    from hdbscan import HDBSCAN
    best_hdbscan = None
    best_hdbscan_score = -1
    for min_size in [3, 4, 5, 6]:
        for min_samples in [1, 2]:
            clusterer = HDBSCAN(min_cluster_size=min_size, min_samples=min_samples)
            labels = clusterer.fit_predict(embeddings)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            if n_clusters >= 3 and n_noise <= 60:
                # Silhouette only on non-noise points
                mask = labels != -1
                if mask.sum() > n_clusters:
                    score = silhouette_score(embeddings[mask], labels[mask])
                    print(f"  min_size={min_size}, min_samples={min_samples}: {n_clusters} clusters, {n_noise} noise, silhouette={score:.3f}")
                    if score > best_hdbscan_score:
                        best_hdbscan_score = score
                        best_hdbscan = (min_size, labels, n_clusters, n_noise)
    if best_hdbscan:
        results["hdbscan"] = {"min_size": best_hdbscan[0], "labels": best_hdbscan[1], "n_clusters": best_hdbscan[2], "n_noise": best_hdbscan[3], "score": best_hdbscan_score}
        print(f"Best HDBSCAN: min_size={best_hdbscan[0]}, {best_hdbscan[2]} clusters, {best_hdbscan[3]} noise, silhouette={best_hdbscan_score:.3f}")
    else:
        print("HDBSCAN failed to find valid clusters. Using K-Means.")

    # Pick winner
    if "hdbscan" in results and results["hdbscan"]["score"] > results["kmeans"]["score"]:
        winner = "hdbscan"
        best_labels = results["hdbscan"]["labels"]
        print(f"\nWinner: HDBSCAN (silhouette={results['hdbscan']['score']:.3f} vs {results['kmeans']['score']:.3f})")
    else:
        winner = "kmeans"
        best_labels = results["kmeans"]["labels"]
        print(f"\nWinner: K-Means K={results['kmeans']['k']} (silhouette={results['kmeans']['score']:.3f})")

    results["winner"] = winner
    results["best_labels"] = best_labels
    return results


# Generic words that appear across many B2B SaaS calls but do not distinguish topics
DOMAIN_STOPLIST = {
    "aegis", "detect", "comply", "protect", "customer", "client", "team",
    "call", "meeting", "discussion", "review", "update", "week", "month",
    "quarter", "q1", "q2", "q3", "q4", "year", "day", "time",
    "need", "want", "like", "think", "know", "look", "going", "make",
    "sure", "right", "really", "actually", "probably", "definitely",
    "okay", "ok", "yeah", "yes", "no", "well", "um", "uh",
    "also", "just", "still", "even", "back", "now", "way", "thing",
    "things", "people", "person", "guys", "folks", "everyone", "someone",
}


def extract_keywords(summaries: list[str], labels: list[int], n_keywords: int = 10) -> dict:
    """Extract top TF-IDF keywords per cluster, filtering domain stoplist."""
    cluster_docs = defaultdict(list)
    for summary, label in zip(summaries, labels):
        if label == -1:
            continue
        cluster_docs[label].append(summary)

    keywords = {}
    for cluster_id, docs in sorted(cluster_docs.items()):
        vectorizer = TfidfVectorizer(max_features=200, stop_words="english", ngram_range=(1, 2))
        try:
            tfidf = vectorizer.fit_transform(docs)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf.mean(axis=0).A1
            top_indices = scores.argsort()[::-1]

            # Filter out domain stoplist words and keep top N
            filtered = []
            for idx in top_indices:
                word = feature_names[idx]
                if word.lower() in DOMAIN_STOPLIST:
                    continue
                filtered.append(word)
                if len(filtered) >= n_keywords:
                    break
            keywords[int(cluster_id)] = filtered
        except ValueError:
            keywords[int(cluster_id)] = []
    return keywords


def llm_label_cluster(keywords: list[str], sample_summaries: list[str]) -> tuple[str, str]:
    """Ask LLM to name a cluster and summarize why the name fits.

    Returns (name, justification).
    """
    from utils import llm_generate

    prompt = f"""You are naming a topic category for a cluster of customer calls.

Top keywords in this cluster:
{', '.join(keywords[:10])}

Sample call summaries from this cluster:
1. {sample_summaries[0][:200]}
2. {sample_summaries[1][:200] if len(sample_summaries) > 1 else sample_summaries[0][:200]}
3. {sample_summaries[2][:200] if len(sample_summaries) > 2 else sample_summaries[0][:200]}

Give this topic a short, professional name (2-4 words) AND a one-sentence justification for why this name fits.

Format exactly like this:
Name: <2-4 word name>
Justification: <one sentence>"""

    response = llm_generate(prompt, max_tokens=100)
    name = None
    justification = "Cluster identified by TF-IDF keywords and sample summaries."

    for line in response.split("\n"):
        line = line.strip()
        if line.lower().startswith("name:"):
            name = line.split(":", 1)[1].strip().strip('"').strip("'")
        elif line.lower().startswith("justification:"):
            justification = line.split(":", 1)[1].strip()

    if not name or len(name) > 50 or "topic" in name.lower():
        return None, justification
    return name, justification


# ---------------------------------------------------------------------------
# Business taxonomy (10-category) — presented in the deck alongside clusters
# ---------------------------------------------------------------------------

BUSINESS_TAXONOMY = {
    "Billing & Contracts": [
        "billing", "invoice", "contract", "payment", "overage", "seat", "license",
        "renewal", "cost", "pricing", "quote", "purchase order", "po", "budget",
    ],
    "Identity & Access": [
        "sso", "single sign", "mfa", "multi-factor", "saml", "ldap", "identity",
        "access", "provisioning", "deprovision", "role", "rbac", "permission",
        "login", "authentication", "user account",
    ],
    "Compliance & Audit": [
        "compliance", "audit", "soc", "iso", "hipaa", "gdpr", "pci", "regulator",
        "framework", "certification", "evidence", "assessment", "auditor",
    ],
    "Platform Reliability": [
        "outage", "incident", "downtime", "reliability", "availability", "sla",
        "uptime", "failure", "degradation", "performance", "latency", "disruption",
    ],
    "Integrations & API": [
        "api", "integration", "webhook", "connector", "sdk", "cli", "endpoint",
        "rate limit", "token", "third-party", "sync", "api access",
    ],
    "Customer Success": [
        "onboarding", "training", "qbr", "business review", "adoption", "usage",
        "success", "check-in", "expansion", "health score", "relationship",
    ],
    "Threat Detection": [
        "detect", "alert", "threat", "malware", "ioc", "signature", "false positive",
        "detection", "monitoring", "siem", "anomaly", "investigation",
    ],
    "Product & Roadmap": [
        "roadmap", "feature", "product", "release", "launch", "enhancement",
        "request", "backlog", "sprint", "rollout", "deployment", "milestone",
    ],
    "Churn & Risk": [
        "churn", "cancel", "competitor", "competition", "alternative", "switching",
        "escalation", "escalate", "executive", "ceo", "cto", "dissatisfied",
        "frustrated", "unhappy", "at risk", "renewal concern",
    ],
    "Internal Operations": [
        "standup", "sprint", "retro", "planning", "engineering", "internal",
        "all hands", "team sync", "postmortem", "retrospective", "roadmap planning",
    ],
}


def score_taxonomy_categories(call: dict) -> dict[str, float]:
    """Score each business category for a call using title, summary, and topics."""
    title = (call.get("meeting_info") or {}).get("title", "").lower()
    summary = extract_summary_text(call).lower()
    topics = call.get("summary", {}).get("topics", [])
    if isinstance(topics, str):
        topics = [topics]
    topics_text = " ".join(str(t).lower() for t in topics)
    text = f"{title} {summary} {topics_text}"

    scores = {}
    for category, keywords in BUSINESS_TAXONOMY.items():
        score = 0.0
        for kw in keywords:
            # Title matches weighted higher because title is a strong signal
            count_title = title.count(kw)
            count_body = (summary.count(kw) + topics_text.count(kw))
            score += count_title * 2.0 + count_body * 1.0
        scores[category] = score
    return scores


def assign_business_categories(calls: list[dict]) -> list[dict]:
    """Return list of {meeting_id, primary, secondary, scores} for each call."""
    assignments = []
    for call in calls:
        scores = score_taxonomy_categories(call)
        sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_cats[0][0] if sorted_cats[0][1] > 0 else "Other"
        secondary = None
        if len(sorted_cats) > 1 and sorted_cats[1][1] > 0:
            secondary = sorted_cats[1][0]
        assignments.append({
            "meeting_id": call["meeting_id"],
            "title": (call.get("meeting_info") or {}).get("title", ""),
            "primary": primary,
            "secondary": secondary,
            "scores": {k: round(v, 2) for k, v in scores.items()},
        })
    return assignments


def build_business_taxonomy(calls: list[dict], type_map: dict) -> dict:
    """Build category counts, cross-tab by call type, and top-category narratives."""
    assignments = assign_business_categories(calls)
    df_assign = pd.DataFrame(assignments)

    # Overall counts
    primary_counts = df_assign["primary"].value_counts().to_dict()

    # Cross-tab by call type
    df_assign["call_type"] = df_assign["meeting_id"].map(type_map).fillna("unknown")
    by_type = {}
    for ctype in ["support", "external", "internal"]:
        subset = df_assign[df_assign["call_type"] == ctype]
        counts = subset["primary"].value_counts().to_dict()
        total = len(subset)
        by_type[ctype] = {
            "total": int(total),
            "categories": {
                cat: {"count": int(c), "pct": round(100 * c / total, 1) if total else 0}
                for cat, c in counts.items()
            },
        }

    # Top categories overall
    top_categories = []
    for cat, count in sorted(primary_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        # Find the call type where this category is most dominant
        best_type = None
        best_pct = 0
        for ctype, info in by_type.items():
            cat_info = info["categories"].get(cat)
            if cat_info and cat_info["pct"] > best_pct:
                best_pct = cat_info["pct"]
                best_type = ctype
        top_categories.append({
            "category": cat,
            "count": int(count),
            "pct_of_total": round(100 * count / len(calls), 1) if calls else 0,
            "dominant_call_type": best_type,
            "dominant_pct": best_pct,
        })

    # Narratives for the top 3 categories
    narratives = {}
    keyword_examples = {
        "Billing & Contracts": "seat overages, invoice adjustments, renewal terms",
        "Identity & Access": "SSO, MFA, RBAC, provisioning and deprovisioning",
        "Compliance & Audit": "SOC 2, ISO 27001, HIPAA evidence and auditor workflows",
        "Platform Reliability": "outages, SLA breaches, incident postmortems",
        "Integrations & API": "webhooks, API rate limits, CLI and connector gaps",
        "Customer Success": "onboarding, QBRs, adoption and expansion plays",
        "Threat Detection": "alert tuning, false positives, detection coverage",
        "Product & Roadmap": "feature requests, release timing, roadmap alignment",
        "Churn & Risk": "escalations, competitor evaluations, renewal concerns",
        "Internal Operations": "sprint planning, engineering standups, postmortems",
    }
    for cat_info in top_categories[:3]:
        cat = cat_info["category"]
        dominant_type = cat_info.get("dominant_call_type")
        dominant_pct = cat_info.get("dominant_pct", 0)
        support_str = by_type.get("support", {}).get("categories", {}).get(cat, {})

        if dominant_type == "external":
            narratives[cat] = f"Top topic for external calls ({dominant_pct}%). {keyword_examples.get(cat, '')} dominate."
        elif dominant_type == "support" and cat in ["Platform Reliability", "Threat Detection"]:
            narratives[cat] = f"Most prevalent in support calls ({support_str.get('pct', 0)}%). Drives negative sentiment when incidents hit customers."
        elif dominant_type == "support":
            narratives[cat] = f"Dominant in support calls ({support_str.get('pct', 0)}%). {keyword_examples.get(cat, '')}."
        elif dominant_type == "internal":
            narratives[cat] = f"Concentrated in internal calls ({dominant_pct}%). {keyword_examples.get(cat, '')}."
        else:
            narratives[cat] = f"{keyword_examples.get(cat, 'Recurring theme across calls')}."

    return {
        "assignments": assignments,
        "primary_counts": {k: int(v) for k, v in primary_counts.items()},
        "top_categories": top_categories,
        "by_call_type": by_type,
        "narratives": narratives,
        "taxonomy": {k: v for k, v in BUSINESS_TAXONOMY.items()},
    }


# Distinct, presentation-safe categorical colours (no near-white segments)
CATEGORY_PALETTE = [
    "#1f77b4",  # Billing & Contracts
    "#ff7f0e",  # Identity & Access
    "#2ca02c",  # Compliance & Audit
    "#d62728",  # Platform Reliability
    "#9467bd",  # Integrations & API
    "#8c564b",  # Customer Success
    "#e377c2",  # Threat Detection
    "#7f7f7f",  # Product & Roadmap
    "#bcbd22",  # Churn & Risk
    "#17becf",  # Internal Operations
    "#4c4c4c",  # Other
]


def plot_taxonomy_by_call_type(by_call_type: dict, total_calls: int):
    """Generate a stacked horizontal bar chart of category counts by call type."""
    categories = list(BUSINESS_TAXONOMY.keys()) + ["Other"]
    type_order = ["support", "external", "internal"]
    color_map = dict(zip(categories, CATEGORY_PALETTE))

    fig, ax = create_chart_fig("03_topic_distribution_by_type.png")
    bottoms = {cat: 0 for cat in categories}
    for ctype in type_order:
        cat_counts = by_call_type.get(ctype, {}).get("categories", {})
        counts = [cat_counts.get(cat, {}).get("count", 0) for cat in categories]
        ax.barh(ctype, counts[0], left=bottoms[categories[0]], color=color_map[categories[0]], label=categories[0] if ctype == "support" else "")
        for cat, count in zip(categories[1:], counts[1:]):
            ax.barh(ctype, count, left=bottoms[cat], color=color_map[cat], label=cat if ctype == "support" else "")
            bottoms[cat] += count

    ax.set_xlabel("Number of Calls")
    ax.set_title("Topic Category Distribution by Call Type")
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    save_chart(fig, "03_topic_distribution_by_type.png")
    plt.close(fig)


def main():
    set_chart_style()
    print("=" * 60)
    print("03 TOPIC MODELING: Clustering + Labeling")
    print("=" * 60)

    calls = load_all_calls()
    print(f"\nTotal calls: {len(calls)}")

    summaries = [extract_summary_text(c) for c in calls]
    meeting_ids = [c["meeting_id"] for c in calls]
    titles = [c.get("meeting_info", {}).get("title", "") for c in calls]

    # Remove empty summaries and filter out company name from keywords
    valid_indices = [i for i, s in enumerate(summaries) if s.strip()]
    print(f"Calls with summaries: {len(valid_indices)}")

    valid_summaries = [summaries[i] for i in valid_indices]
    valid_ids = [meeting_ids[i] for i in valid_indices]
    valid_titles = [titles[i] for i in valid_indices]

    # Clean summaries for better clustering (remove company name)
    cleaned_summaries = []
    for s in valid_summaries:
        # Remove common company/product names that appear in every summary
        cleaned = s.lower()
        for word in ["aegis", "detect", "comply", "protect"]:
            cleaned = cleaned.replace(word, "")
        cleaned_summaries.append(cleaned)

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------
    embeddings = embed_summaries(cleaned_summaries)
    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
    print(f"Embeddings saved: {OUTPUT_DIR / 'embeddings.npy'}")

    # ------------------------------------------------------------------
    # Cluster
    # ------------------------------------------------------------------
    results = compare_methods(embeddings, valid_summaries)
    best_labels = results["best_labels"]

    # ------------------------------------------------------------------
    # Extract keywords
    # ------------------------------------------------------------------
    print("\n--- Extracting keywords per cluster ---")
    keywords = extract_keywords(valid_summaries, best_labels)

    for cluster_id, words in sorted(keywords.items()):
        print(f"  Cluster {cluster_id}: {', '.join(words[:5])}")

    # ------------------------------------------------------------------
    # LLM label clusters
    # ------------------------------------------------------------------
    print("\n--- LLM labeling clusters ---")
    cluster_names = {}
    cluster_samples = defaultdict(list)

    for i, label in enumerate(best_labels):
        if label != -1:
            cluster_samples[label].append(valid_summaries[i])

    # Manual labels based on keywords (fallback if LLM fails)
    manual_labels = {
        0: "Identity & Access Management",
        1: "Incident Response & Outages",
        2: "Compliance & Audits",
        3: "Product Deployment & Setup",
        4: "Sales & Renewals",
        5: "Engineering & Planning",
    }

    # Track used names to prevent duplicates
    used_names = set()
    cluster_justifications = {}

    for cluster_id in sorted(cluster_samples.keys()):
        words = keywords.get(cluster_id, [])
        samples = cluster_samples[cluster_id]
        name, justification = llm_label_cluster(words, samples)

        # Deduplicate: if name already used, fallback to manual or generic
        if not name or name in used_names:
            name = manual_labels.get(cluster_id, f"Topic {cluster_id}")
            justification = "Named from manual fallback after LLM duplicate/failure."

        # Still duplicate? Append cluster ID
        if name in used_names:
            name = f"{name} ({cluster_id})"
            justification += " (deduplicated)"

        used_names.add(name)
        cluster_names[int(cluster_id)] = name
        cluster_justifications[int(cluster_id)] = justification
        print(f"  Cluster {cluster_id}: '{name}' ({len(samples)} calls)")

    # ------------------------------------------------------------------
    # Correct weak/mock LLM labels against TF-IDF keywords
    # ------------------------------------------------------------------
    corrected_cluster_names = {
        int(cid): correct_topic_name(name, keywords.get(int(cid), []))
        for cid, name in cluster_names.items()
    }

    # ------------------------------------------------------------------
    # Build topic assignments
    # ------------------------------------------------------------------
    topic_assignments = []
    for i, label in enumerate(best_labels):
        topic_assignments.append({
            "meeting_id": valid_ids[i],
            "title": valid_titles[i],
            "cluster_id": int(label),
            "topic_name": corrected_cluster_names.get(int(label), "Noise") if label != -1 else "Noise",
            "summary_snippet": valid_summaries[i][:200],
        })

    df_topics = pd.DataFrame(topic_assignments)
    print(f"\n--- Topic Distribution ---")
    print(df_topics["topic_name"].value_counts())

    # ------------------------------------------------------------------
    # Representative calls per topic
    # ------------------------------------------------------------------
    print("\n--- Representative calls per topic ---")
    topic_reps = {}
    for topic_name in df_topics["topic_name"].unique():
        if topic_name == "Noise":
            continue
        topic_calls = df_topics[df_topics["topic_name"] == topic_name]
        reps = topic_calls.head(3)[["title", "summary_snippet"]].to_dict("records")
        topic_reps[topic_name] = reps
        print(f"\n  {topic_name}:")
        for r in reps:
            print(f"    - {r['title'][:70]}...")

    # ------------------------------------------------------------------
    # Topic distribution by call type
    # ------------------------------------------------------------------
    print("\n--- Topic distribution by call type ---")
    call_types_path = OUTPUT_DIR / "02_call_types.csv"
    topic_by_type = {}
    if call_types_path.exists():
        df_types = pd.read_csv(call_types_path)
        df_merged = df_topics.merge(df_types[["meeting_id", "final_label"]], on="meeting_id", how="left")
        for ctype in ["support", "external", "internal"]:
            subset = df_merged[df_merged["final_label"] == ctype]
            counts = subset[subset["topic_name"] != "Noise"]["topic_name"].value_counts().to_dict()
            total = len(subset)
            topic_by_type[ctype] = {
                "total": total,
                "topics": {k: {"count": int(v), "pct": round(100*v/total, 1)} for k, v in counts.items()},
            }
            print(f"  {ctype}: {total} calls")
            for t, info in list(topic_by_type[ctype]["topics"].items())[:3]:
                print(f"    - {t}: {info['count']} ({info['pct']}%)")
    else:
        print("  02_call_types.csv not found. Skipping cross-tab.")

    # ------------------------------------------------------------------
    # Business taxonomy
    # ------------------------------------------------------------------
    print("\n--- Building business taxonomy ---")
    type_map = {}
    if call_types_path.exists():
        df_types = pd.read_csv(call_types_path)
        type_map = dict(zip(df_types["meeting_id"].astype(str), df_types["final_label"]))
    business_taxonomy = build_business_taxonomy(calls, type_map)
    print(f"  Top categories: {business_taxonomy['top_categories'][:3]}")

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    print("\n--- Generating charts ---")

    # Topic distribution bar chart
    topic_counts = df_topics[df_topics["topic_name"] != "Noise"]["topic_name"].value_counts()
    fig, ax = create_chart_fig("03_topic_distribution.png")
    ax.barh(topic_counts.index, topic_counts.values, color="steelblue")
    ax.set_xlabel("Number of calls")
    ax.set_title("Topic Distribution")
    for i, v in enumerate(topic_counts.values):
        ax.text(v + 0.3, i, str(v), va="center")
    plt.tight_layout()
    save_chart(fig, "03_topic_distribution.png")
    plt.close(fig)

    # Clustering comparison chart
    fig, ax = create_chart_fig("03_clustering_comparison.png")
    methods = []
    scores = []
    if "hdbscan" in results:
        methods.append(f"HDBSCAN\n({results['hdbscan']['n_clusters']} clusters)")
        scores.append(results["hdbscan"]["score"])
    methods.append(f"K-Means K={results['kmeans']['k']}")
    scores.append(results["kmeans"]["score"])
    colors = ["green" if s == max(scores) else "gray" for s in scores]
    ax.bar(methods, scores, color=colors)
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Clustering Method Comparison")
    ax.set_ylim(0, max(scores) * 1.2)
    for i, s in enumerate(scores):
        ax.text(i, s + 0.01, f"{s:.3f}", ha="center", fontweight="bold")
    save_chart(fig, "03_clustering_comparison.png")
    plt.close(fig)

    # Business taxonomy stacked bar
    plot_taxonomy_by_call_type(business_taxonomy["by_call_type"], len(calls))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    total_calls = len(best_labels)
    clustered_count = int((best_labels != -1).sum())
    noise_count = int((best_labels == -1).sum())
    hdbscan_info = results.get("hdbscan", {})

    topics_data = {
        "method": results["winner"],
        "comparison": {
            "hdbscan": {k: v for k, v in hdbscan_info.items() if k != "labels"},
            "kmeans": {k: v for k, v in results["kmeans"].items() if k != "labels"},
        },
        "clusters": {
            str(cid): {
                "name": cluster_names.get(cid, f"Cluster {cid}"),
                "justification": cluster_justifications.get(cid, ""),
                "keywords": keywords.get(cid, []),
                "count": int((best_labels == cid).sum()),
                "representative_calls": topic_reps.get(cluster_names.get(cid, f"Cluster {cid}"), []),
            }
            for cid in sorted(cluster_samples.keys())
        },
        "limitations": {
            "silhouette_weak": True,
            "hdbscan_silhouette": round(hdbscan_info.get("score", 0), 3),
            "kmeans_silhouette": round(results["kmeans"].get("score", 0), 3),
            "noise_count": noise_count,
            "noise_pct": round(100 * noise_count / total_calls, 1) if total_calls else 0,
            "clustered_count": clustered_count,
            "clustered_pct": round(100 * clustered_count / total_calls, 1) if total_calls else 0,
            "notes": [
                "Silhouette scores are weak in absolute terms (ideal > 0.5).",
                "HDBSCAN wins relatively by honestly flagging noise instead of forcing assignments.",
                "Cluster labels are interpretive and should be validated with keywords + sample calls.",
            ],
        },
        "topic_by_call_type": topic_by_type,
        "assignments": topic_assignments,
        "business_taxonomy": business_taxonomy,
    }
    save_json(topics_data, "topics.json")

    print("\n" + "=" * 60)
    print("03 TOPIC MODELING: Done")
    print("=" * 60)

    return topics_data


if __name__ == "__main__":
    main()
