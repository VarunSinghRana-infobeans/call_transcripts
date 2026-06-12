# How to Use This Analysis Pipeline

> Step-by-step guide from zero to PowerPoint.

---

## Prerequisites

- Python 3.10 or higher
- ~2 GB free disk space (for embedding model)
- Internet connection (first run only, to download models)

---

## Step 1: Install Python Packages

### Option A: Global Install (Simplest)

```bash
cd assignment/notebook
pip install pandas numpy matplotlib seaborn scikit-learn hdbscan sentence-transformers python-pptx openai python-dotenv
```

### Option B: Virtual Environment (Recommended)

```bash
cd assignment/notebook
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import pandas, numpy, matplotlib, sklearn, hdbscan, sentence_transformers, pptx; print('All packages OK')"
```

---

## Step 2: Configure AI Provider

The scripts use AI for call type classification and topic cluster naming.
You have three options. Pick one.

### Option A: OpenAI (Cloud) — Easiest, Costs ~$0.50

```bash
# 1. Get API key from https://platform.openai.com/api-keys
# 2. Copy config template
cp .env.example .env

# 3. Edit .env and add your key
OPENAI_API_KEY=sk-your-actual-key-here
AI_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
```

**Cost estimate:** 70 API calls × $0.00015 per call = **~$0.50 total**

**Why OpenAI:** Fast, high quality, no local setup. The panel expects you to use modern tools.

### Option B: Ollama (Local) — Free, Works Offline

```bash
# 1. Download Ollama from https://ollama.com
# 2. Open a terminal, keep it running:
ollama pull llama3.2
ollama serve

# 3. In another terminal, copy config:
cp .env.example .env

# 4. Edit .env:
AI_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
```

**Why Ollama:** Zero cost, works without internet, no API key needed. Slightly slower than OpenAI.

**Important:** `ollama serve` must stay running in a separate terminal window while you run the scripts.

### Option C: Mock (No AI) — For Testing

```bash
cp .env.example .env
# Edit .env:
AI_PROVIDER=mock
```

**Why Mock:** Free, instant, no setup. Uses keyword matching instead of LLM. Good for testing script logic before spending money.

**Trade-off:** Less accurate classification and naming. Use for development, switch to OpenAI for final run.

---

## Step 3: Run the Pipeline

Scripts must run in order. Each script reads output from previous scripts.

```bash
# Step 1: Load and explore data
python 01_explore.py

# Step 2: Classify call types (uses AI for ambiguous cases)
python 02_call_types.py

# Step 3: Discover topics with clustering (uses AI for naming)
python 03_topic_modeling.py

# Step 4: Analyze sentiment trends
python 04_sentiment.py

# Step 5: Churn scoring, feature requests, escalation funnel
python 05_bonus_insights.py

# Step 6: Generate PowerPoint
python 06_generate_ppt.py
```

### Expected Output

```
output/
|-- charts/
|   |-- 01_duration_distribution.png
|   |-- 01_sentiment_distribution.png
|   |-- 01_call_volume_over_time.png
|   |-- 02_call_types_distribution.png
|   |-- 03_topic_distribution.png
|   |-- 03_clustering_comparison.png
|   |-- 04_sentiment_trend_by_type.png
|   |-- 04_negative_sentiment_trend.png
|   |-- 04_sentiment_boxplot.png
|   |-- 04_sentiment_stacked_by_type.png
|   |-- 05_churn_risk_distribution.png
|   |-- 05_churn_score_histogram.png
|   |-- 05_feature_requests.png
|   |-- 05_escalation_chain_lengths.png
|   |-- 01_overall_sentiment.png
|   |-- ... (14+ total)
|
|-- 01_calls_summary.csv
|-- 01_explore_stats.json
|-- 02_call_types.csv
|-- 02_call_types_stats.json
|-- 04_sentiment_details.csv
|-- 04_sentiment_weekly.csv
|-- 04_sentiment_stats.json
|-- 05_churn_scores.csv
|-- 05_churn_scores.json
|-- 05_feature_requests.csv
|-- 05_feature_requests.json
|-- 05_escalations.json
|-- embeddings.npy
|-- topics.json
|-- Transcript_Intelligence_Report.pptx    <-- FINAL DELIVERABLE
```

---

## Step 4: Review the Report

Open `output/Transcript_Intelligence_Report.pptx` in PowerPoint or Google Slides.

The report has 14 slides:
- **Slide 1:** Title + KPIs
- **Slide 2:** Executive Summary
- **Slide 3:** Pipeline Overview
- **Slide 4:** Dataset Overview
- **Slide 5:** Topic Categorization
- **Slide 6:** Sentiment Analysis by Call Type
- **Slide 7:** Where Sentiment Goes Negative
- **Slide 8:** Where Sentiment Is Strongest
- **Slide 9:** Churn Risk Detection
- **Slide 10:** Feature Request Intelligence
- **Slide 11:** Action Items & Call Efficiency
- **Slide 12:** Recommendations
- **Slide 13:** Appendix: Pipeline Methodology
- **Slide 14:** Appendix: Additional Charts

---

## Step 5: Prepare for Q&A

Bookmark these cells/files for panel questions:

| If they ask about... | Open this... | Key evidence |
|---------------------|--------------|--------------|
| "How did you classify calls?" | `02_call_types.csv` | Heuristic + LLM table with confidence scores |
| "Why these topic categories?" | `output/topics.json` | Cluster keywords + HDBSCAN vs K-Means comparison |
| "How do you know sentiment dropped?" | `04_sentiment_weekly.csv` | Week-over-week numbers by call type |
| "Why is this account high risk?" | `05_churn_scores.json` | Feature-based point breakdown |
| "What feature requests came up?" | `05_feature_requests.json` | Keyword frequency counts |

---

## Re-running Individual Scripts

If you change one script, you only need to re-run it and downstream scripts:

```bash
# Example: You tweaked churn scoring in 05_bonus_insights.py
python 05_bonus_insights.py
python 06_generate_ppt.py          # Rebuild PPT with new data

# No need to re-run 01-04 if you didn't change them
```

---

## Switching AI Providers Mid-Stream

```bash
# Test with mock (fast, free)
AI_PROVIDER=mock python 02_call_types.py
AI_PROVIDER=mock python 03_topic_modeling.py

# Final run with OpenAI (better quality)
AI_PROVIDER=openai python 02_call_types.py
AI_PROVIDER=openai python 03_topic_modeling.py

# Rebuild PPT (no AI needed)
python 06_generate_ppt.py
```

---

## Common Issues

### "No module named 'sklearn'"

```bash
pip install scikit-learn
```

Note: The package name is `scikit-learn` but you import it as `sklearn`.

### "No module named 'sentence_transformers'"

```bash
pip install sentence-transformers
```

Note: Use hyphen in pip, underscore in import.

### "OSError: cannot load library 'hdbscan'"

On Windows, HDBSCAN sometimes fails to compile. Try:

```bash
pip uninstall hdbscan
pip install hdbscan --no-cache-dir
```

Or use the pre-built wheel:

```bash
pip install hdbscan==0.8.33
```

### "OpenAI API key not set"

```bash
# Check if key is loaded
python -c "import os; print(os.environ.get('OPENAI_API_KEY', 'NOT SET'))"

# If NOT SET, make sure .env file exists and has the key
# Or set it manually:
set OPENAI_API_KEY=sk-your-key    # Windows
export OPENAI_API_KEY=sk-your-key # Mac/Linux
```

### "Ollama server not responding"

```bash
# Check if server is running
curl http://localhost:11434/api/tags

# If no response, start it:
ollama serve
```

### Charts appear in output folder but not in PPT

Run `06_generate_ppt.py` AFTER all other scripts. It reads chart files from `output/charts/`.

```bash
python 01_explore.py
python 02_call_types.py
python 03_topic_modeling.py
python 04_sentiment.py
python 05_bonus_insights.py
python 06_generate_ppt.py   # <-- Must be last
```

---

## Time Estimate

| Step | Time | Notes |
|------|------|-------|
| Install packages | 5-10 min | One-time |
| Configure AI | 2-5 min | Depends on provider |
| Run 01_explore.py | 10 sec | Fast |
| Run 02_call_types.py | 10-60 sec | Mock=instant, OpenAI=~30 calls |
| Run 03_topic_modeling.py | 30-120 sec | Model download on first run |
| Run 04_sentiment.py | 10 sec | Fast |
| Run 05_bonus_insights.py | 10 sec | Fast |
| Run 06_generate_ppt.py | 5 sec | Fast |
| **Total first run** | **5-10 min** | Mostly waiting for model download |
| **Total subsequent** | **1-2 min** | All scripts |

---

## Next Steps After Notebook

If the panel picks the **platform track** (Track B), the analysis outputs feed directly into it:

```
Notebook outputs              Platform input
----------------              --------------
output/topics.json      ->    PostgreSQL topics table
output/embeddings.npy   ->    pgvector call_chunks table
output/05_churn_scores.json -> Dashboard risk alerts
```

The platform consumes the same data. No duplicated analysis work.

See `../platform/` for the full-stack implementation.
