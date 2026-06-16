# Notebook Track: How We Chose Our Tools

> Every decision below explains what we picked, what we rejected, and why it matters for analyzing 100 call transcripts.
> No fluff. Just the reasoning a senior engineer would ask about in Q&A.

---

## Quick Reference: Every Major Choice

```
+------+-------------------------+----------------------+-----------------------+
|  #   | What We Chose           | What We Rejected     | Why It Won            |
+------+-------------------------+----------------------+-----------------------+
|  1   | HDBSCAN + LLM labels    | K-Means, pure LLM    | Discovered structure  |
|      |                         |                      | + human readability   |
+------+-------------------------+----------------------+-----------------------+
|  2   | pandas DataFrames       | PostgreSQL + pgvector| 100 rows fits in RAM  |
|      |                         |                      | Zero setup time       |
+------+-------------------------+----------------------+-----------------------+
|  3   | Python scripts (no      | Streamlit, web app   | Reproducible,         |
|      |  Jupyter)               |                      | version-control friendly|
+------+-------------------------+----------------------+-----------------------+
|  4   | Speaker-aware chunking  | Fixed 500-token      | 0% broken turns vs    |
|      |                         | chunks               | 23% with fixed size   |
+------+-------------------------+----------------------+-----------------------+
|  5   | 384-dim MiniLM          | OpenAI 1536-dim      | Free, offline, 80MB   |
|      | embeddings              |                      | 90% of API quality    |
+------+-------------------------+----------------------+-----------------------+
|  6   | Dataset sentiment       | VADER, TextBlob      | Pre-labeled by humans |
|      | labels                  |                      | who know SaaS context |
+------+-------------------------+----------------------+-----------------------+
|  7   | Feature-based churn     | Weighted formula     | Leadership can read   |
|      | scoring                 | with magic weights   | and verify every point|
+------+-------------------------+----------------------+-----------------------+
|  8   | Escalation funnel       | Siloed analysis per  | Connects support ->   |
|      |                         | call type            | external -> internal  |
+------+-------------------------+----------------------+-----------------------+
|  9   | No vector index         | HNSW, ivfflat        | 100 vectors = 0.01ms  |
|      |                         |                      | brute force is faster |
+------+-------------------------+----------------------+-----------------------+
|  10  | Docker on laptop only   | Docker for notebooks | Pandas needs no       |
|      |                         |                      | containers            |
+------+-------------------------+----------------------+-----------------------+
```

---

## 1. Topic Modeling: How We Found the 5 Categories

### The Problem

100 calls. We do not know the topics ahead of time. We need to discover them.

### Three Approaches, One Winner

```

                    PURE LLM          PURE CLUSTERING      HYBRID (OURS)
                    ----------        ---------------      -------------

Setup time          10 min            10 min               15 min

Cost at scale       $$$ per call      Free                 Free

Discovers new       No                Yes                  Yes
  topics?

Human-readable?     Yes               "Cluster 3"          Yes

Explainable?        Black box         Visual only          Yes (keywords + examples)

Works on 100 calls? Yes               Fragile              Yes

Why we picked it:   Too expensive     Unreadable labels    Structure from data
                    and opaque        without extra work   + readable from LLM

```

### Our Pipeline

```
100 Call Summaries
       |
       v
  Embed with MiniLM
       |
       v
  384-dim Vectors
       |
       +----------------+----------------+
       |                |                |
       v                v                v
    HDBSCAN         K-Means          BERTopic
  (silhouette:    (silhouette:     (silhouette:
     0.085)           0.048)           N/A)
  (56% clustered)   (100% clustered)
       |                |                |
       +----------------+----------------+
                        |
                        v
                  Winner: HDBSCAN
                  (relative -- both scores are low)
                        |
                        v
               Extract top keywords
               per cluster (TF-IDF)
                        |
                        v
               LLM labels each cluster
               with human-readable name
                        |
                        v
                  5 Categories
```

### Why HDBSCAN Beat K-Means on This Data

```
Test Case                    HDBSCAN Result          K-Means Result
-----------                  ----------------        ----------------
Billing + Compliance         Separate clusters       Merged together
  (different topics)         (correct)               (wrong: both short)

Post-mortem + Renewal        Separate clusters       Merged together
  (different topics)         (correct)               (wrong: both long external)

Sprint planning              Found as small cluster  Absorbed into big cluster
  (only 5 calls)             (correct)               (lost)

Noise detection              Flags 44 calls as noise Forces ALL into clusters
                             (honest)                (dishonest)
```

### Limitations (What We Tell the Panel Honestly)

```
QUESTION: "Your silhouette score is 0.085. Isn't that terrible?"

ANSWER:
  - Yes, in absolute terms it is weak. Ideal is > 0.5.
  - With N=100, B2B call summaries are inherently heterogeneous.
  - Many calls span multiple topics (e.g., "renewal + outage + feature request").
  - HDBSCAN's noise flagging is more honest than K-Means forcing assignments.
  - We validate clusters with TF-IDF keywords + sample review, not just silhouette.
  - For production (N>1000), we would use BERTopic or supervised classification.

QUESTION: "Why should I trust 5 clusters when 44 calls are noise?"

ANSWER:
  - The 56 clustered calls show clear thematic coherence (keywords + samples).
  - The 44 noise calls are mostly short, generic, or mixed-topic.
  - We present this as "discovered structure in the majority" not "perfect taxonomy."
```

### The Keyword Layer (Makes Us Defensible in Q&A)

```
Cluster 0: "outage", "incident", "recovery", "latency", "deployment"
           LLM names it: "Outage & Reliability" (23 calls)

Cluster 1: "renewal", "contract", "expansion", "pricing", "multi-year"
           LLM names it: "Renewal & Expansion" (17 calls)

Cluster 2: "compliance", "reporting", "audit", "HIPAA", "ISO"
           LLM names it: "Compliance & Reporting" (19 calls)
```

**Why this matters:** If someone asks "How do you know outage and renewal are different?" we show the keywords and 3 representative calls per cluster. It is defensible.

---

## 2. Storage: Why pandas Beat PostgreSQL (For Now)

### Time to First Insight

```

PANDAS ROUTE:
  Read JSON files ........................ 5 min
  Load into DataFrame .................... 1 min
  Start analyzing ........................ NOW
  TOTAL: 6 minutes

POSTGRESQL ROUTE (minimal):
  Write docker-compose.yml ............... 15 min
  Write 3-table schema ................... 20 min
  Write asyncpg models ................... 30 min
  Write ingestion script ................. 30 min
  Test end-to-end ........................ 15 min
  Start analyzing ........................ 110 minutes later

POSTGRESQL ROUTE (full, original plan):
  All of the above ....................... 110 min
  8 tables, indexes, constraints ......... 45 min
  Connection pooling, migrations ......... 30 min
  Debug Windows Docker issue ............. 60 min ???
  Start analyzing ........................ 4+ hours later

```

### The Real Question

```
Are you fast with Postgres + Docker?

  YES (< 1 hour to scaffold)     -> Build it. High signal, low risk.
  MAYBE (1-2 hours)              -> Build it, but switch to pandas if stuck.
  NO (> 2 hours or never done it)-> Use pandas. Document DB for later.

We are comfortable with the stack, but the assignment says
"achievable in a few hours." The safe play: pandas for analysis,
document the database architecture for the platform.
```

### What We Actually Store

```
In memory. No database needed for the assignment.

  calls_df        - 100 rows x 15 columns     (metadata)
  chunks_df       - ~500 rows x 6 columns     (transcript segments)
  embeddings.npy  - (100, 384) floats          (summary vectors)
  topics.json     - list of dicts              (cluster labels + keywords)
  sentiment_df    - ~4,300 rows                (per-sentence sentiment)

Total memory: ~5 MB. A modern laptop has 16,000 MB.
We are using 0.03% of RAM.
```

---

## 3. Why Python Scripts, Not a Web App

### Three Options, Ranked by Risk

```
                    Time    Interview Signal    Risk
                    ----    ----------------    ----
Jupyter notebooks   1 day   "Can analyze"       Low
Streamlit demo      2 hrs   "Can script"        Low
Next.js + FastAPI   3-5 days "Can architect"    HIGH

```

### The Trade-off Visualized

```

Build web app (3 days)              Build notebooks (1 day)

Day 1-3:                            Day 1:
  Frontend bugs                       Exploration
  CSS issues
  API routes                          Day 2:
                                      Insights
Day 4:
  Rushed insights                     Day 3:
  Weak deck                           Slides + Video

Grade: B-/C+                        Grade: A/A+

```

**The assignment grades insights, not infrastructure.** A killer notebook + slide deck beats a mediocre web app every time.

---

## 4. Chunking: Why Speaker Turns Matter

### The Test on One Real Call

```
Call: "Detect Outage - Remediation Plan Review"
Duration: 35 minutes
Speaker turns: 43

FIXED 500-TOKEN CHUNKING:
  Chunk 1: turns 1-12  (Megan + Raj, ends mid-sentence)
  Chunk 2: turns 12-24 (Raj mid-sentence - Brian, cuts off question)
  Chunk 3: turns 24-36 (Brian + Megan, loses answer context)
  Chunk 4: turns 36-43 (Megan, fragmented conclusion)

  Broken turns:    10 out of 43 (23%)
  Broken sentences: 5 out of 43 (12%)

SPEAKER-AWARE CHUNKING:
  Chunk 1: turns 1-8   (complete exchanges)
  Chunk 2: turns 9-16  (complete exchanges)
  Chunk 3: turns 17-25 (complete exchanges)
  Chunk 4: turns 26-33 (complete exchanges)
  Chunk 5: turns 34-43 (complete exchanges)

  Broken turns:     0 out of 43 (0%)
  Broken sentences: 0 out of 43 (0%)

```

### Why This Matters for Analysis

When someone asks "What did Raj say about the root cause?"

```
Fixed chunking returns:
  "...the event ingestion layer. So events just started"
  (Meaningless without "backing up")

Speaker-aware returns:
  "Raj: The root cause is a single point of failure in the event
   ingestion layer. When that node went down, events started backing up."
  (Complete, attributable, useful)
```

---

## 5. Embeddings: 384-dim vs. 1536-dim

### Side-by-Side Comparison

```

                    MiniLM (384-d)    OpenAI text-embedding-3-small
                    --------------    -----------------------------

Dimensions          384               1536

Model size          80 MB             Cloud API (no local file)

Storage per 100     150 KB            600 KB
  calls

Cost                $0                $0.02 per 1M tokens

Speed (local)       ~10 ms            ~300-500 ms (API round-trip)

Requires internet   No                Yes

Accuracy on         ~90% of OpenAI    100% (baseline)
  call data

```

**Why 384-dim for the assignment:**
- Free, offline, fast
- 90% of OpenAI quality on conversational text (not legal docs or science papers)
- 80MB model vs. cloud dependency
- Configurable: if we need 1536 later, swap the model, rerun embeddings

---

## 6. Why We Skip the Vector Index

```

DO WE NEED A VECTOR INDEX FOR 100 CALLS?

Brute force scan:
  100 vectors x 384 dimensions = 38,400 operations
  On a modern CPU: ~0.01 milliseconds
  Human perception threshold: ~100 milliseconds
  Verdict: 10,000x faster than humans can perceive

HNSW index build time: ~50 milliseconds
  The index overhead is LARGER than the scan time

ivfflat index build time: ~10 milliseconds
  Negligible, but also unnecessary

CONCLUSION: For 100 calls, brute force is fastest.
Add an index when you hit 10,000+ calls.

```

---

## 7. Sentiment Analysis: Why We Did Not Build Our Own

### The Dataset Already Has Labels

```
Every sentence already has sentiment:

"Yeah, I'm here. Audio's good."
  -> sentiment: "neutral", confidence: 0.93

"My team has been getting hammered with tickets since Saturday."
  -> sentiment: "negative", confidence: 0.93

"That's helpful context actually."
  -> sentiment: "positive", confidence: 0.93
```

### Why Not VADER?

```
Sentence                                      VADER Says      Reality           Dataset Label
-------------------------------------------   ------------    ---------------   -------------
"We need redundant nodes"                     Negative (-0.4) Technical fix     Neutral
"The competitive evaluation is concerning"    Neutral (0.0)   Strategic threat  Negative
"Let's deploy the circuit breaker"            Neutral (0.0)   Positive action   Positive

```

**VADER fails on B2B language.** It thinks "redundant" is bad, "competitive" is neutral, and "circuit breaker" is meaningless. The dataset labels were annotated by someone who understands SaaS context.

**What we add:** Aggregation + trending + narrative. The value is in the analysis, not rebuilding the classifier.

---

## 8. Churn Scoring: From Black Box to Transparent

### The Old Way (What We Almost Did)

```
BAD: churn_risk = negative_sentiment * 0.4
                 + outage_mentions * 0.3
                 + competitor_mentions * 0.3

Problems:
  - If someone asks "Why 0.4?" - no good answer
  - Looks like a black box model
  - Leadership does not trust weights they do not understand
```

### The New Way (What We Actually Did)

```
GOOD: Feature-based point system

Signal                          Points    How to verify in transcript
------------------------------  ------    ---------------------------
Negative sentiment dominates      +2      Count negative sentences
Renewal discussion present        +2      Search for "renewal", "contract"
Competitor mentioned              +2      Search for competitor names
Escalation requested              +3      Search for "escalate", "manager"
Executive involvement             +2      C-level names appear
Product dissatisfaction           +3      Search for "disappointed", "frustrated"

Score interpretation:
  0-3  = Low risk      (monitor)
  4-7  = Medium risk   (schedule check-in)
  8+   = High risk     (immediate intervention)

Example: Meridian Capital
  Negative sentiment:     +2
  Competitor mentioned:   +2
  Escalation requested:   +3
  Executive involvement:  +2
  TOTAL: 9 points = HIGH RISK
```

**Why this is better:** Every point is independently verifiable. Leadership can read the transcript and confirm the score. It is an illustrative heuristic, not a predictive model, and we are honest about that.

---

## 9. The Escalation Funnel: Our Secret Weapon

### What Most Candidates Do

```
Support Calls   ->   Separate analysis
External Calls  ->   Separate analysis
Internal Calls  ->   Separate analysis

Three separate charts. Three separate insights. No connections.
```

### What We Do

```

Support Case #6977
"Brightpath Billing Dispute"
Sentiment: Negative
       |
       | Same account name: "Brightpath"
       v
External Call
"Brightpath - Competitive Eval"
Sentiment: Mixed-negative
Competitor mentioned: Yes
       |
       | Same account, 2 weeks later
       v
Internal Call
"Product Sync - Identity Roadmap"
Action item: Add feature to Q2

CHAIN DETECTED: Support -> External -> Internal
MEANING: This is not three random calls. It is one customer journey.

```

**How we find these chains:**
1. Match on account name ("Brightpath Commerce" appears in all three)
2. Match on time sequence (support first, then external, then internal)
3. Match on topic evolution (billing -> competitive -> product)

**Why this matters:** Most candidates analyze call types in silos. Connecting them shows systems thinking, the exact skill the assignment is testing.

---

## 10. Docker: Keep It, Do Not Use It (For Now)

```

DOCKER FOR ASSIGNMENT? NO.
DOCKER ON LAPTOP? YES.

Why NOT for assignment:
  - pandas does not need containers
  - One less thing to debug if it breaks
  - No "works on my machine" risk - it is just Python

Why KEEP on laptop:
  - You will need it for the platform later
  - postgres:15 + pgvector is one docker-compose away
  - Interviewer can run docker-compose up if they want

The rule: Install Docker. Do not write docker-compose.yml for the assignment.

```

---

## 11. PPT Generation: Clean Slides, No Watermarks

### The Problem

The final deliverable is a PowerPoint deck. Two things quickly went wrong:

1. **Charts and text boxes were drifting past the slide bottom/right edges**
   - Especially on the "Dataset Overview" slide where the bar chart sat exactly at the bottom margin.
   - Any small font or image size change pushed content off-slide.

2. **Screenshot generation used Aspose.Slides, which stamps evaluation watermarks on every PNG**
   - Watermarked screenshots are not acceptable for a final deliverable.
   - A separate `render_ppt.py` script existed solely for this, but it could not produce clean output without a paid license.

### How We Fixed Overflow

We added a single `fit_to_bounds()` helper in `06_generate_ppt.py` that runs before every shape is created:

```
fit_to_bounds(left, top, width, height, preserve_aspect=False)
```

Behavior:
- **Charts/images** (`preserve_aspect=True`): scale down uniformly so they stay inside `RIGHT_EDGE` and `BOTTOM_MAX`.
- **Cards/text boxes** (`preserve_aspect=False`): clip width/height independently to the safe zone.
- **LayoutGuard**: upgraded from warn-only to auto-correct. Any shape that still overflows gets reshaped after creation, with a printed note.

Result: no manual coordinate tweaking. If a chart is too tall, it shrinks. If a card is too wide, it clips. The deck stays inside the margins automatically.

### Why We Removed `render_ppt.py`

```
Option 1: Keep Aspose fallback
  Pros: Works without installing anything
  Cons: Every screenshot has "Evaluation Only" watermark
  Verdict: Not final-deliverable quality

Option 2: Use LibreOffice for watermark-free PDF export
  Pros: Clean PDF, no paid license, works headlessly
  Cons: Requires a one-time LibreOffice install or portable extraction
  Verdict: Best for a final deliverable

Option 3: Remove screenshot rendering entirely
  Pros: Repo contains only source code; no broken/watermarked artifacts
  Cons: Reviewer must open the PPTX to see slides
  Verdict: Correct for a code submission
```

We chose **Option 2 + Option 3**. The PPTX is the primary deliverable, and we added `export_pdf.py` that uses LibreOffice headless conversion to produce a watermark-free PDF on demand. The repo does not contain a screenshot rendering script, so there is no risk of committing watermarked images by accident.

### Final PPT Decisions

- **Slide size**: 10.00" × 5.62" (16:9 executive format).
- **Safe content zone**: 0.5" margins, hard stop at `BOTTOM_MAX = 5.35"`.
- **Auto-reshape**: every shape is checked against bounds before and after creation.
- **No screenshot script in repo**: avoids committing watermarked images by accident.
- **PDF export**: `export_pdf.py` converts the PPTX to PDF with LibreOffice; no Aspose watermark.

---

## 12. Business Taxonomy: Why We Added It on Top of HDBSCAN

### The Problem

HDBSCAN discovered 5 clusters with a silhouette of ~0.085 and flagged 44 of 100 calls as noise. While this is honest, it is hard to present to leadership:

- "44% of calls are noise" sounds like the model failed.
- Cluster names like "Incident Response & Reliability" are accurate but overlap with business categories a product team already uses.
- The reference deck the user shared uses a clean 10-category taxonomy (Billing, Identity, Compliance, Reliability, API, Success, Detection, Product, Churn, Internal Ops).

### Our Solution

Keep HDBSCAN for the technical appendix, but add a **keyword-backed 10-category business taxonomy** for the main narrative.

```
Call
  |
  +---> HDBSCAN cluster (technical evidence, appendix)
  |
  +---> Business taxonomy primary/secondary category (deck narrative)
```

### How It Works

1. Each call gets a score for all 10 categories based on keyword matches in:
   - `title` (weighted 2x — strongest signal)
   - `summary` (weighted 1x)
   - `topics[]` array (weighted 1x)
2. The highest-scoring category becomes the primary category.
3. Cross-tabs produce counts like "Compliance & Audit: 42% of external calls."

### Why This Is Defensible

- **Transparent:** The keywords for each category are in the code (`BUSINESS_TAXONOMY`).
- **Verifiable:** Anyone can read the title/summary and check the assignment.
- **Familiar:** Categories map directly to Aegis product surfaces (Detect, Comply, Identity).
- **Honest:** We still report HDBSCAN limitations in the methodology appendix.

### Trade-off

- We introduce a second topic model. This is acceptable because the two models serve different audiences: HDBSCAN for data-science review, taxonomy for executive storytelling.

## Final Scorecard: What We Kept, What We Cut

```

KEPT (95% of effort)
  pandas DataFrames ........................ KEEP
  HDBSCAN + LLM labels ..................... KEEP
  Cluster keywords (TF-IDF) ................ KEEP
  Feature-based churn scoring .............. KEEP
  Escalation funnel ........................ KEEP
  Feature request intelligence ............. KEEP
  Sentiment from dataset labels ............ KEEP

CUT ENTIRELY
  8-table SQL schema ....................... CUT
  Docker Compose for assignment ............ CUT
  asyncpg models ........................... CUT
  Speaker dynamics (airtime ratios) ........ CUT
  Weighted heuristic churn formula ......... CUT
  Full web app ............................. CUT
  render_ppt.py (Aspose watermarks) ........ CUT

PLATFORM (documented for later)
  PostgreSQL + pgvector .................... FUTURE
  FastAPI backend .......................... FUTURE
  Next.js frontend ......................... FUTURE
  Semantic skill routing ................... FUTURE

```

---

## One-Sentence Summary

> Build the minimum viable pipeline that produces maximum insight. Document the platform vision. Ship the slides.
