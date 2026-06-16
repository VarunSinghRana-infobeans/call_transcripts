# Transcript Intelligence — Notebook Track

> Python scripts that analyze 100 B2B SaaS call transcripts and generate a PowerPoint report.
> No Jupyter notebooks. No web app. Just pure Python + charts + slides.

---

## What This Is

This folder contains the **analysis engine** for the Transcript Intelligence assignment.

It reads 100 JSON call transcripts, runs statistical analysis, discovers topics, scores churn risk, and outputs a 21-slide PowerPoint presentation.

The panel sees:
- **6 Python scripts** with full methodology
- **20+ charts** (matplotlib)
- **1 PowerPoint** with findings and narrative
- **Optional watermark-free PDF** via LibreOffice
- **Q&A ready** with bookmarked analysis steps

---

## Folder Structure

```
assignment/notebook/
|
|-- scripts/
|   |-- utils.py              # Shared: JSON loading, chart saving, data export
|   |-- ai_config.py          # AI provider switcher: OpenAI / Ollama / Mock
|   |-- ppt_data.py           # Loads/validates all outputs for the PPT
|
|-- .env.example              # Configuration template for AI providers
|-- .env                      # Your actual config (copy from .env.example)
|
|-- 01_explore.py             # Load data, validate, print distributions
|-- 02_call_types.py          # Heuristic + LLM classification
|-- 03_topic_modeling.py      # HDBSCAN clustering + LLM labeling
|-- 04_sentiment.py           # Sentiment trends by type and week
|-- 05_bonus_insights.py      # Churn scoring + feature requests + escalation funnel
|-- 06_generate_ppt.py        # Build PowerPoint from all outputs
|-- export_pdf.py             # Convert PPTX to watermark-free PDF
|
|-- output/                   # Generated files
|   |-- charts/               # 20+ PNG charts
|   |-- *.csv                 # Data tables
|   |-- *.json                # Structured results
|   |-- Transcript_Intelligence_Report.pptx   # Final presentation
|   |-- Transcript_Intelligence_Report.pdf    # Watermark-free PDF (optional)
|
|-- HOW_AI_WORKS.md           # How AI is used in each script
|-- HOW_TO_USE.md             # Step-by-step run guide
|-- README.md                 # This file
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn hdbscan sentence-transformers python-pptx openai python-dotenv
```

Or use the virtual environment:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure AI Provider

```bash
cp .env.example .env
```

Edit `.env` and pick one:

```bash
# Option A: OpenAI (cloud, recommended)
AI_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here

# Option B: Ollama (local, free)
AI_PROVIDER=ollama
# Run: ollama pull llama3.2
# Run: ollama serve

# Option C: Mock (no API, for testing)
AI_PROVIDER=mock
```

See `HOW_AI_WORKS.md` for detailed provider comparison.

### 3. Run Analysis

```bash
# Option A: run the whole pipeline in one command (creates a timestamped run folder)
python run_all.py

# Option B: run scripts individually
python 01_explore.py
python 02_call_types.py
python 03_topic_modeling.py
python 04_sentiment.py
python 05_bonus_insights.py
python 06_generate_ppt.py
```

### 4. Export / Open the Report

```bash
# Optional: create a watermark-free PDF (requires LibreOffice)
python export_pdf.py

# Open the PowerPoint
output/Transcript_Intelligence_Report.pptx
```

---

## What Each Script Does

| Script | Purpose | AI Used? | Why This Approach |
|--------|---------|----------|-------------------|
| `01_explore.py` | Load 100 calls, validate schema, print stats | No | Need to know the data before analyzing it |
| `02_call_types.py` | Classify as support/external/internal | Yes (ambiguous cases only) | Heuristics catch 70%, LLM handles edge cases |
| `03_topic_modeling.py` | Discover topic clusters from summaries | Yes (cluster naming) | HDBSCAN finds structure; LLM makes it readable |
| `04_sentiment.py` | Aggregate sentiment trends over time | No | Dataset already has per-sentence labels |
| `05_bonus_insights.py` | Churn risk + feature requests + escalation | No | Rule-based scoring is transparent and verifiable |
| `06_generate_ppt.py` | Build 21-slide PowerPoint | No | python-pptx is the standard for programmatic slides |
| `export_pdf.py` | Convert PPTX to PDF | No | Uses LibreOffice headless conversion |

---

## Key Design Decisions

### Why Python Scripts Instead of Jupyter?

Jupyter is great for exploration but has problems for deliverables:
- **Reproducibility:** Scripts run top-to-bottom every time. Notebooks can execute cells out of order.
- **Version control:** Scripts are plain text, easy to diff. Notebook JSON is unreadable in git.
- **Production path:** If the panel picks the platform track, these scripts become the backend data pipeline.
- **CI/CD ready:** Scripts can run in GitHub Actions, Docker, or cron jobs. Notebooks cannot.

**Verdict:** Jupyter for exploration, Python scripts for deliverables.

### Why PowerPoint Instead of Google Slides / PDF?

- **PowerPoint:** Industry standard, editable by the panel, supports animations and transitions.
- **Google Slides:** Requires internet, sharing permissions, Google account.
- **PDF:** Static, not editable, hard to present from.

**Verdict:** PowerPoint is the safest format for a take-home assignment.

### Why HDBSCAN Instead of K-Means?

We tested both and compared silhouette scores:

```
HDBSCAN:  5 clusters, silhouette=0.085 (44% noise)
K-Means:  5 clusters, silhouette=0.048
```

**Honest assessment:** Both scores are low in absolute terms (ideal > 0.5). With N=100, clear topic separation is difficult. HDBSCAN wins *relatively* because:
- **Noise detection:** Flags ambiguous calls instead of forcing them into wrong clusters.
- **Variable density:** Some topics have 20+ calls, others have 5. HDBSCAN handles both.
- **No K needed:** We don't know the "right" number of topics ahead of time.

**Trade-offs:**
- HDBSCAN leaves ~44% of calls as "noise" (unclustered). We accept this for better cluster purity.
- Low silhouette means clusters overlap. We mitigate this with TF-IDF keywords and manual review.
- For production (N>1000), consider BERTopic or supervised topic models.

### Why Rule-Based Churn Scoring Instead of ML Model?

| Approach | Transparency | Panel Q&A | Leadership Trust |
|----------|-------------|-----------|------------------|
| ML model (XGBoost, etc.) | Black box | Hard to explain | Low |
| Weighted formula | Semi-transparent | "Why 0.4 weight?" | Medium |
| **Feature-based points (OURS)** | Fully transparent | "Count negative sentences" | **High** |

Every point is independently verifiable. The panel can read the transcript and confirm the score.

**Verdict:** Transparency beats complexity for interview settings.

### Why 384-dim Embeddings Instead of OpenAI 1536-dim?

| Model | Dimensions | Size | Cost | Speed |
|-------|-----------|------|------|-------|
| MiniLM (ours) | 384 | 80 MB | $0 | ~10ms |
| OpenAI text-embedding-3-small | 1536 | Cloud only | $0.02/1M tokens | ~300ms |

MiniLM gives 90% of OpenAI's quality on conversational text. For 100 calls, the difference is negligible. For the assignment, free and offline wins.

---

## Outputs

### Charts (24 total)

| Chart | Script | What It Shows |
|-------|--------|---------------|
| Duration distribution | 01 | How long calls are (9-54 min) |
| Sentiment distribution | 01 | Score spread across 100 calls |
| Call volume over time | 01 | Calls per week (Feb-Apr 2026) |
| Call type distribution | 02 | Support vs external vs internal |
| Topic distribution | 03 | 5 discovered categories |
| Clustering comparison | 03 | HDBSCAN vs K-Means scores |
| Sentiment trend | 04 | Week-over-week by call type |
| Negative sentiment trend | 04 | Which weeks were worst |
| Churn risk distribution | 05 | High/medium/low risk counts |
| Feature requests | 05 | Most requested capabilities |

### Data Files

| File | Content |
|------|---------|
| `01_calls_summary.csv` | Metadata for all 100 calls |
| `02_call_types.csv` | Classification with confidence |
| `04_sentiment_weekly.csv` | Aggregated weekly sentiment |
| `05_churn_scores.csv` | Risk scores for every call |
| `topics.json` | Cluster keywords, names, assignments |
| `embeddings.npy` | 100 vectors of 384 dimensions |

### Presentation

`Transcript_Intelligence_Report.pptx` — 21 slides:
1. Title + KPIs
2. Executive Summary
3. Pipeline Overview
4. Dataset Overview
5. Topic Categorization
6. Sentiment Analysis by Call Type
7. Where Sentiment Goes Negative
8. Where Sentiment Is Strongest
9. Churn Risk Detection
10. Renewal Risk
11. Feature Request Intelligence
12. Action Items & Call Efficiency
13. Carry-Forward Actions
14. Recommendations at a Glance
15-19. Recommendation Details (one slide per top recommendation)
20. AI & Data Reasonableness Check
21. Appendix: Pipeline Methodology

---

## Extending This

### Add More Calls

Drop new JSON folders into `dataset/`. Re-run scripts. Embeddings and clusters update automatically.

### Try Different Clustering

Edit `03_topic_modeling.py`, change HDBSCAN parameters:

```python
clusterer = HDBSCAN(min_cluster_size=3)  # More clusters, less noise
```

### Switch Embedding Model

```python
# In 03_topic_modeling.py
model = SentenceTransformer("BAAI/bge-small-en")  # 512-dim, better quality
```

### Use Different LLM

Edit `.env`:

```bash
# OpenAI
OPENAI_MODEL=gpt-4o  # Better quality, more expensive

# Ollama
OLLAMA_MODEL=mistral  # Different local model
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'sklearn'` | `pip install scikit-learn` |
| ` sentence_transformers` not found | `pip install sentence-transformers` |
| `pptx` import error | `pip install python-pptx` |
| HDBSCAN crashes | `pip install hdbscan` |
| Ollama not responding | Run `ollama serve` in another terminal |
| OpenAI rate limit | Switch to `AI_PROVIDER=mock` for testing |
| Charts missing from PPT | Run scripts 01-05 first, then 06 |
| PDF export fails | Install LibreOffice or use PowerPoint's export manually |

---

## One-Line Summary

> **Six Python scripts. One data-validation layer. Twenty-one slides. Zero notebooks.**
