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
    # Charts
    # ------------------------------------------------------------------
    print("\n--- Generating charts ---")

    # Topic distribution bar chart
    topic_counts = df_topics[df_topics["topic_name"] != "Noise"]["topic_name"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(topic_counts.index, topic_counts.values, color="steelblue")
    ax.set_xlabel("Number of calls")
    ax.set_title("Topic Distribution")
    ax.tick_params(axis="y", labelsize=9)
    for i, v in enumerate(topic_counts.values):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=9)
    plt.tight_layout()
    save_chart(fig, "03_topic_distribution.png")
    plt.close(fig)

    # Clustering comparison chart
    fig, ax = plt.subplots(figsize=(8, 5))
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
    }
    save_json(topics_data, "topics.json")

    print("\n" + "=" * 60)
    print("03 TOPIC MODELING: Done")
    print("=" * 60)

    return topics_data


if __name__ == "__main__":
    main()
