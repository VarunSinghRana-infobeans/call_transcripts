# How AI Works in Track A (Notebook Scripts)

> No Jupyter needed. Just Python scripts + LLM calls.

---

## What AI Does Here

AI is NOT the main worker. It is the **reviewer, labeler, and narrator**.

The heavy lifting (clustering, scoring, statistics) is done by algorithms.
AI makes the output human-readable and defensible.

---

## 1. Call Type Classification (script 02)

### Step 1: Heuristic Rules (No AI)

```python
def heuristic_classify(title):
    title_lower = title.lower()
    if any(word in title_lower for word in ['support', 'ticket', 'issue', 'bug', 'incident']):
        return 'support'
    if any(word in title_lower for word in ['renewal', 'contract', 'pricing', 'evaluation', 'demo']):
        return 'external'
    if any(word in title_lower for word in ['sprint', 'planning', 'sync', 'standup', 'retro']):
        return 'internal'
    return 'ambiguous'
```

### Step 2: AI Reviews Ambiguous Cases (LLM)

```
Input to LLM:
  "Classify this call title: 'Brightpath - Technical Deep Dive'
   Options: support, external, internal
   Also consider the summary: 'Discussion of API integration...'"

Output from LLM:
  "external"

Why AI here:
  Heuristics catch 70% of cases cleanly.
  AI handles the 30% that are ambiguous.
  We log both guesses for transparency.
```

### Transparency Table

```
meeting_id    title                           heuristic    llm        final      confidence
------------  ------------------------------  ---------  ---------  ---------  ----------
m-001         "Support Ticket #6977 Review"   support    support    support    0.95
m-002         "Brightpath - Competitive Eval"  external   external   external   0.92
m-003         "Sprint 23 Planning"            internal   internal   internal   0.98
m-004         "Product Sync - Identity"       internal   external   internal   0.71   <- flagged
```

**AI cost:** ~30 API calls (only for ambiguous cases).

---

## 2. Topic Cluster Labeling (script 03)

### Step 1: HDBSCAN Clustering (No AI)

```python
from hdbscan import HDBSCAN
import numpy as np

# Embed 100 call summaries -> 100 vectors of 384 dims
embeddings = model.encode(summaries)

# HDBSCAN finds natural groups
clusterer = HDBSCAN(min_cluster_size=5)
labels = clusterer.fit_predict(embeddings)

# Result: cluster 0, cluster 1, cluster 2, ... (-1 = noise)
```

HDBSCAN gives us **numbers**. AI gives us **names**.

### Step 2: AI Labels Each Cluster (LLM)

```python
# For cluster 0, extract top TF-IDF keywords
keywords = ["outage", "incident", "recovery", "latency", "deployment"]

# Ask LLM to name it
prompt = f"""
These words appear frequently together in call summaries:
{', '.join(keywords)}

Give this topic a short, professional name (2-4 words).
"""

# LLM responds: "Outage & Reliability"
```

### Result

```
Cluster 0: "Outage & Reliability" (23 calls)
  Keywords: outage, incident, recovery, latency, deployment

Cluster 1: "Renewal & Expansion" (17 calls)
  Keywords: renewal, contract, expansion, pricing, multi-year

Cluster 2: "Compliance & Reporting" (19 calls)
  Keywords: compliance, reporting, audit, HIPAA, ISO
```

**Why AI here:**
  - HDBSCAN says "Cluster 0." LLM says "Outage & Reliability."
  - Panel asks "Why these categories?" We show keywords + LLM reasoning.
  - Defensible and human-readable.

**AI cost:** 6-8 API calls (one per cluster).

---

## 3. Churn Risk Scoring (script 05)

### Rule-Based Scoring (No AI for the score)

```python
def score_churn_risk(call):
    score = 0
    if call.negative_sentiment_pct > 0.5:
        score += 2
    if 'renewal' in call.summary.lower():
        score += 2
    if 'competitor' in call.transcript.lower():
        score += 2
    if 'escalate' in call.transcript.lower():
        score += 3
    if has_executive_involvement(call):
        score += 2
    if 'disappointed' in call.transcript.lower() or 'frustrated' in call.transcript.lower():
        score += 3
    return score
```

### AI Extracts Evidence (Optional, for slides)

```python
# Ask LLM: "Find the exact sentence where the customer expresses dissatisfaction"
prompt = f"""
In this call transcript, find the most negative sentence.
Return the exact quote and who said it.

Transcript:
{call.transcript}
"""

# LLM responds:
# "Raj (Customer): 'My team has been getting hammered with tickets since Saturday.'"
```

**Why AI here:**
  - Scoring is rule-based (transparent, no black box).
  - AI extracts the "smoking gun" quotes for slides.
  - Panel sees the evidence, not just the number.

**AI cost:** ~20 API calls (one per at-risk account).

---

## 4. Slide Narrative Generation (script 06)

### AI Writes the Story

```python
prompt = f"""
Here are the findings from analyzing 100 customer calls:

- 23 calls about outages and reliability issues
- Support sentiment dropped 18% the week of March 9
- 4 accounts scored high for churn risk
- "Granular restore" was requested 6 times

Write 3 bullet points that a CEO would care about.
Make them specific, impactful, and actionable.
"""

# LLM responds:
# "1. Reliability issues are the #1 driver of negative customer sentiment,
#     affecting 23 of 100 recent calls.
#  2. Four enterprise accounts show high churn risk signals,
#     including competitor mentions and escalation requests.
#  3. Customers have requested 'granular restore' 6 times in 2 months,
#     indicating a gap in our backup product line."
```

**Why AI here:**
  - We have the numbers. AI makes them compelling.
  - Different audiences need different framing.
  - CEO wants business impact. Engineer wants technical details.

**AI cost:** 5-10 API calls (one per slide section).

---

## 5. AI Chat Simulation (for platform pitch)

### If Panel Asks "What If I Want to Ask Questions?"

```python
# This is what the platform would do. For the notebook pitch,
# we simulate one Q&A pair to show the concept.

question = "Which accounts mentioned competitors?"

# In platform: vector search + LLM answers from transcripts
# In notebook: we pre-compute the answer and put it in slides

prompt = f"""
Based on these churn scores and call summaries,
which accounts mentioned competitors and what did they say?

{churn_data}
"""

# LLM responds with a formatted table for the slide
```

---

## AI Usage Summary

| Script | What AI Does | How Many Calls | Cost |
|--------|-------------|----------------|------|
| 02_call_types.py | Reviews ambiguous titles | ~30 | Low |
| 03_topic_modeling.py | Names clusters | 6-8 | Very low |
| 05_bonus_insights.py | Extracts evidence quotes | ~20 | Low |
| 06_generate_slides.py | Writes narrative | 5-10 | Very low |

**Total: ~70 API calls.** With GPT-4o-mini or local Ollama: essentially free.

---

## AI Provider Options

```
Option 1: OpenAI API (cloud)
  - GPT-4o-mini: fast, cheap, high quality
  - GPT-4o: best quality, slightly more expensive
  - Cost for 70 calls: ~$0.50 - $2.00

Option 2: Ollama (local)
  - llama3.2 or mistral: free, offline
  - Requires 8GB+ RAM
  - Slower but zero cost

Option 3: Skip AI entirely
  - Hardcode cluster names
  - Write slide narrative manually
  - Risk: less compelling, more work

Recommendation: OpenAI API with GPT-4o-mini.
  Fast, cheap, and the panel expects you to use modern tools.
```

---

## One-Line Summary

> **Algorithms find the patterns. AI explains them. Together they make insights that stick.**
