"""
06_generate_ppt.py

Generate a production-quality PowerPoint presentation.
Visual style: clean executive briefing deck with card-based layouts,
consistent grid, generous margins, and no text overflow.

Exports: Transcript_Intelligence_Report.pptx
"""

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from datetime import datetime

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt

from ppt_data import PresentationData, fmt_num, load_presentation_data
from utils import CHARTS_DIR, OUTPUT_DIR


def fmt_pct(value, decimals=1, fallback="N/A"):
    return fmt_num(value, decimals, fallback)


# ---------------------------------------------------------------------------
# Slide dimensions — 16:9 executive format (matches reference deck)
# ---------------------------------------------------------------------------
SLIDE_WIDTH = Inches(10.0)
SLIDE_HEIGHT = Inches(5.62)

MARGIN = Inches(0.5)
CONTENT_W = SLIDE_WIDTH - MARGIN * 2
RIGHT_EDGE = SLIDE_WIDTH - MARGIN
BOTTOM_MAX = Inches(5.35)  # Hard stop for content

# Title area
TITLE_Y = Inches(0.25)
TITLE_H = Inches(0.50)
SUBTITLE_Y = Inches(0.82)
SUBTITLE_H = Inches(0.38)
UNDERLINE_Y = Inches(0.78)
UNDERLINE_W = Inches(1.80)
UNDERLINE_H = Inches(0.04)

# Content area
CONTENT_Y = Inches(1.30)

# Layout zones
CHART_X = MARGIN
CHART_W = Inches(5.4)
RIGHT_X = Inches(6.2)
RIGHT_W = Inches(3.25)

# Colors
C_PRIMARY = RGBColor(0x1a, 0x23, 0x7e)      # Dark blue
C_SECONDARY = RGBColor(0x00, 0x78, 0xd4)    # Bright blue
C_ACCENT = RGBColor(0xff, 0x6b, 0x35)       # Orange
C_TEXT = RGBColor(0x33, 0x33, 0x33)         # Dark gray
C_LIGHT = RGBColor(0x88, 0x88, 0x88)        # Light gray
C_WHITE = RGBColor(0xff, 0xff, 0xff)
C_BG = RGBColor(0xf5, 0xf6, 0xf8)           # Card background

C_RED = RGBColor(0xc0, 0x39, 0x2b)
C_AMBER = RGBColor(0xe6, 0x8a, 0x00)
C_GREEN = RGBColor(0x2e, 0x7d, 0x32)

# Fonts
FONT_TITLE = Pt(34)
FONT_SUBTITLE = Pt(15)
FONT_HEADING = Pt(23)
FONT_BODY = Pt(12)
FONT_SMALL = Pt(11)
FONT_TINY = Pt(10)
FONT_MICRO = Pt(9)
FONT_KPI = Pt(30)
FONT_KPI_LABEL = Pt(11)
FONT_NUMBER = Pt(14)

# Emoji mapping
TOPIC_ICONS = {
    "Compliance & Certification": "🔒",
    "Compliance & Audit": "🔒",
    "Compliance & Audits": "🔒",
    "Incident Response & Reliability": "⚠️",
    "Incident Response & Outages": "⚠️",
    "Platform Reliability": "⚠️",
    "Identity & Access Management": "🪪",
    "Identity & Access": "🪪",
    "Engineering & Sprint Planning": "🔧",
    "Internal Ops": "🔧",
    "Sales & Renewals": "💰",
    "Customer Success": "🤝",
    "Product Deployment & Setup": "🚀",
    "Product": "🚀",
    "Billing": "💳",
    "API": "🔌",
    "Detection": "🔍",
    "Churn & Risk": "🚨",
}


def _to_pt(value) -> float:
    """Convert emu or pt value to points."""
    if isinstance(value, (int, float)) and value > 1000:  # emu
        return value / 12700
    return float(value)


def _to_in(value) -> float:
    """Convert emu or inches value to inches."""
    if isinstance(value, (int, float)) and value > 1000:  # emu
        return value / 914400
    return float(value)


def chars_per_inch(font_size_pt: float) -> float:
    """Approximate Calibri characters per inch at a given font size."""
    return font_size_pt * 1.15


def estimate_lines(text: str, width, font_size) -> int:
    """Rough estimate of wrapped lines."""
    width_in = _to_in(width)
    font_size_pt = _to_pt(font_size)
    if width_in <= 0 or not text:
        return 0
    cpi = chars_per_inch(font_size_pt)
    chars_per_line = max(1, int(width_in * cpi))
    return max(1, (len(text) + chars_per_line - 1) // chars_per_line)


def fit_text(text: str, width, font_size, max_lines: int) -> str:
    """Truncate text with ellipsis so it fits in the given line budget."""
    if not text:
        return text
    width_in = _to_in(width)
    font_size_pt = _to_pt(font_size)
    cpi = chars_per_inch(font_size_pt)
    chars_per_line = max(1, int(width_in * cpi))
    max_chars = chars_per_line * max_lines
    if len(text) <= max_chars:
        return text
    # Leave room for ellipsis
    return textwrap.shorten(text, width=max(max_chars - 3, 10), placeholder="...")


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def style_text(p, size, bold=False, color=C_TEXT, align=PP_ALIGN.LEFT):
    p.font.size = size
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = "Calibri"
    p.space_after = Pt(0)
    p.line_spacing = 1.0
    p.alignment = align


def fit_to_bounds(left, top, width, height, preserve_aspect=False,
                  max_right=RIGHT_EDGE, max_bottom=BOTTOM_MAX):
    """
    Reshape a rectangle so it stays inside the slide content bounds.
    Returns (left, top, width, height) as Inches objects.

    - preserve_aspect=True scales width and height together (best for charts/images).
    - preserve_aspect=False clips width/height independently (best for cards/text).
    """
    left_in = _to_in(left)
    top_in = _to_in(top)
    width_in = _to_in(width)
    height_in = _to_in(height)
    max_right_in = _to_in(max_right)
    max_bottom_in = _to_in(max_bottom)

    # Move inside from left/top first
    if left_in < MARGIN.inches:
        left_in = MARGIN.inches

    right_in = left_in + width_in
    bottom_in = top_in + height_in

    overshoot_x = max(0.0, right_in - max_right_in)
    overshoot_y = max(0.0, bottom_in - max_bottom_in)

    if preserve_aspect and (overshoot_x > 0 or overshoot_y > 0):
        # Scale down uniformly based on the larger overshoot ratio
        scale_x = 1.0 if overshoot_x <= 0 else (max_right_in - left_in) / width_in
        scale_y = 1.0 if overshoot_y <= 0 else (max_bottom_in - top_in) / height_in
        scale = min(scale_x, scale_y)
        width_in *= scale
        height_in *= scale
    else:
        if overshoot_x > 0:
            width_in = max(0.05, max_right_in - left_in)
        if overshoot_y > 0:
            height_in = max(0.05, max_bottom_in - top_in)

    return Inches(left_in), Inches(top_in), Inches(width_in), Inches(height_in)


def add_text(slide, left, top, width, height, text, size=FONT_BODY, bold=False, color=C_TEXT, align=PP_ALIGN.LEFT):
    """Add a textbox with zero internal margins to maximize usable space."""
    left, top, width, height = fit_to_bounds(left, top, width, height)
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(0)
    tf.margin_right = Pt(0)
    tf.margin_top = Pt(0)
    tf.margin_bottom = Pt(0)
    tf.auto_size = MSO_AUTO_SIZE.NONE
    p = tf.paragraphs[0]
    p.text = text
    style_text(p, size, bold=bold, color=color, align=align)
    return box


def add_slide_title(slide, title, subtitle=""):
    """Plain title + optional subtitle with a colored underline."""
    add_text(slide, MARGIN, TITLE_Y, CONTENT_W, TITLE_H,
             title, FONT_HEADING, bold=True, color=C_PRIMARY)
    # Accent underline
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, UNDERLINE_Y, UNDERLINE_W, UNDERLINE_H)
    line.fill.solid()
    line.fill.fore_color.rgb = C_SECONDARY
    line.line.fill.background()
    if subtitle:
        add_text(slide, MARGIN, SUBTITLE_Y, CONTENT_W, SUBTITLE_H,
                 subtitle, FONT_SMALL, color=C_LIGHT)


def add_card(slide, left, top, width, height, fill=C_BG, line=None):
    """Add a rounded rectangle card background. Add BEFORE text so text sits on top."""
    left, top, width, height = fit_to_bounds(left, top, width, height)
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = fill
    if line:
        card.line.color.rgb = line
        card.line.width = Pt(1)
    else:
        card.line.fill.background()
    return card


def add_chart_if_exists(slide, filename, left, top, width, height):
    path = CHARTS_DIR / filename
    if path.exists():
        left, top, width, height = fit_to_bounds(left, top, width, height, preserve_aspect=True)
        return slide.shapes.add_picture(str(path), left, top, width, height)
    return None


def add_circle_badge(slide, left, top, size, text, color=C_SECONDARY, font_size=FONT_NUMBER):
    left, top, width, height = fit_to_bounds(left, top, size, size)
    size = width  # width == height after fitting
    circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, left, top, size, size)
    circle.fill.solid()
    circle.fill.fore_color.rgb = color
    circle.line.fill.background()
    add_text(slide, left, top + Inches(0.02), size, size - Inches(0.04),
             str(text), font_size, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    return circle


# ---------------------------------------------------------------------------
# Layout guard — checks, reshapes, and warns when shapes approach/exceed bounds
# ---------------------------------------------------------------------------

class LayoutGuard:
    def __init__(self, slide, name):
        self.slide = slide
        self.name = name
        self.warnings = []
        self.fixed = []

    def _is_background_or_decorative(self, shape):
        left_in = shape.left.inches
        top_in = shape.top.inches
        width_in = shape.width.inches
        height_in = shape.height.inches
        right_in = left_in + width_in
        bottom_in = top_in + height_in
        full_slide = (left_in < 0.01 and top_in < 0.01 and
                      abs(width_in - SLIDE_WIDTH.inches) < 0.01 and
                      abs(height_in - SLIDE_HEIGHT.inches) < 0.01)
        # Decorative shapes intentionally bleed off-canvas
        decorative = (
            top_in < -0.5 or
            left_in > SLIDE_WIDTH.inches - 1.0 or
            right_in > SLIDE_WIDTH.inches + 1.0 or
            bottom_in > SLIDE_HEIGHT.inches + 0.5
        )
        return full_slide or decorative

    def _reshape_shape(self, shape):
        left_in = shape.left.inches
        top_in = shape.top.inches
        width_in = shape.width.inches
        height_in = shape.height.inches

        preserve_aspect = shape.shape_type == 13  # MSO_SHAPE_TYPE.PICTURE
        new_left, new_top, new_width, new_height = fit_to_bounds(
            left_in, top_in, width_in, height_in,
            preserve_aspect=preserve_aspect
        )
        shape.left = new_left
        shape.top = new_top
        shape.width = new_width
        shape.height = new_height
        return (new_width.inches != width_in or new_height.inches != height_in or
                new_left.inches != left_in or new_top.inches != top_in)

    def check(self, fix=True):
        for shape in self.slide.shapes:
            if self._is_background_or_decorative(shape):
                continue

            left_in = shape.left.inches
            top_in = shape.top.inches
            width_in = shape.width.inches
            height_in = shape.height.inches
            right_in = left_in + width_in
            bottom_in = top_in + height_in

            overflows = (
                right_in > SLIDE_WIDTH.inches + 0.20 or
                bottom_in > SLIDE_HEIGHT.inches + 0.20 or
                bottom_in > BOTTOM_MAX.inches + 0.10
            )

            if overflows:
                if fix:
                    did_fix = self._reshape_shape(shape)
                    if did_fix:
                        self.fixed.append(
                            f"reshaped {shape.shape_type.name} "
                            f"({width_in:.2f}\"x{height_in:.2f}\" -> "
                            f"{shape.width.inches:.2f}\"x{shape.height.inches:.2f}\")"
                        )
                        # Re-check after fix
                        right_in = shape.left.inches + shape.width.inches
                        bottom_in = shape.top.inches + shape.height.inches

                if right_in > SLIDE_WIDTH.inches + 0.20:
                    self.warnings.append(f"exceeds right edge ({right_in:.2f}\")")
                if bottom_in > SLIDE_HEIGHT.inches + 0.20:
                    self.warnings.append(f"exceeds slide bottom ({bottom_in:.2f}\")")
                if bottom_in > BOTTOM_MAX.inches + 0.10:
                    self.warnings.append(f"near bottom edge ({bottom_in:.2f}\", max {BOTTOM_MAX.inches:.2f}\")")

        if self.fixed:
            print(f"  [LAYOUT FIXED] {self.name}")
            for f in self.fixed:
                print(f"      + {f}")
        if self.warnings:
            print(f"  [LAYOUT WARNING] {self.name}")
            for w in self.warnings:
                print(f"      - {w}")
        return self.warnings


def check(slide, name):
    LayoutGuard(slide, name).check(fix=True)


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------

def add_title_slide(prs, data: PresentationData):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
    bg.fill.solid()
    bg.fill.fore_color.rgb = C_PRIMARY
    bg.line.fill.background()

    # Decorative translucent circles
    c1 = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(7.8), Inches(-1.2), Inches(4.5), Inches(4.5))
    c1.fill.solid()
    c1.fill.fore_color.rgb = RGBColor(0x22, 0x2d, 0x8f)
    c1.line.fill.background()

    c2 = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(8.6), Inches(0.2), Inches(2.8), Inches(2.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = RGBColor(0x25, 0x32, 0xa0)
    c2.line.fill.background()

    add_text(slide, MARGIN, Inches(1.40), CONTENT_W, Inches(0.60),
             "TRANSCRIPT INTELLIGENCE", FONT_TITLE, bold=True, color=C_WHITE)
    add_text(slide, MARGIN, Inches(2.05), CONTENT_W, Inches(0.45),
             "Aegis Cloud Security", FONT_SUBTITLE, color=RGBColor(0xcc, 0xcc, 0xcc))
    add_text(slide, MARGIN, Inches(2.55), CONTENT_W, Inches(0.40),
             "Call Analytics Pipeline - Product & Engineering Briefing",
             FONT_BODY, color=RGBColor(0xcc, 0xcc, 0xcc))

    kpis = [
        (data.total_calls, "Calls Analysed", C_WHITE),
        (3, "Call Types", C_WHITE),
        (sum(data.feature_keywords.values()), "Feature Signals", C_WHITE),
        (data.risk_distribution.high, "Churn Flags", C_ACCENT),
    ]
    box_w = Inches(2.1)
    gap = Inches(0.23)
    start_x = MARGIN
    for i, (val, label, color) in enumerate(kpis):
        left = start_x + i * (box_w + gap)
        add_card(slide, left, Inches(3.75), box_w, Inches(0.90), fill=RGBColor(0x22, 0x2d, 0x8f))
        add_text(slide, left, Inches(3.80), box_w, Inches(0.42),
                 str(val), Pt(28), bold=True, color=color, align=PP_ALIGN.CENTER)
        add_text(slide, left, Inches(4.20), box_w, Inches(0.35),
                 label, FONT_KPI_LABEL, color=RGBColor(0xcc, 0xcc, 0xcc), align=PP_ALIGN.CENTER)

    add_text(slide, MARGIN, Inches(5.05), CONTENT_W, Inches(0.30),
             datetime.now().strftime("%B %Y"), FONT_SMALL, color=C_WHITE)
    check(slide, "Title")
    return slide


def add_executive_summary_slide(prs, data: PresentationData):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_title(slide, "Executive Summary", "Three findings that drive decisions today")

    top_feature = next(iter(data.feature_keywords.items()), ("feature", 0))
    worst_zone = data.problem_zones[0] if data.problem_zones else {"topic": "N/A", "call_type": "N/A", "sentiment": 0}

    findings = [
        ("🚨", "Churn risk is concentrated",
         f"{data.risk_distribution.high} accounts flagged high-risk. Support cases and external renewals both show competitor mentions and escalations.",
         C_RED),
        ("📉", f"Sentiment bottoms at {worst_zone['topic']}",
         f"{worst_zone['topic']} × {worst_zone['call_type'].title()} scores {fmt_num(worst_zone['sentiment'])}/5 — the lowest zone in the dataset.",
         C_AMBER),
        ("📣", f"'{top_feature[0].title()}' dominates feature asks",
         f"{top_feature[1]} mentions make it the clearest PM priority. Export and reporting gaps drive repeated friction.",
         C_SECONDARY),
    ]

    y = CONTENT_Y
    card_h = Inches(1.20)
    for icon, title, body, color in findings:
        body = fit_text(body, CONTENT_W - Inches(1.1), FONT_TINY, 3)
        add_card(slide, MARGIN, y, CONTENT_W, card_h)
        add_text(slide, MARGIN + Inches(0.12), y + Inches(0.12), Inches(0.55), Inches(0.55),
                 icon, FONT_HEADING, bold=True, color=color)
        add_text(slide, MARGIN + Inches(0.75), y + Inches(0.15), CONTENT_W - Inches(1.0), Inches(0.30),
                 title, FONT_BODY, bold=True, color=color)
        add_text(slide, MARGIN + Inches(0.75), y + Inches(0.48), CONTENT_W - Inches(1.0), Inches(0.60),
                 body, FONT_TINY, color=C_TEXT)
        y += card_h + Inches(0.08)
    check(slide, "Executive Summary")
    return slide


def add_pipeline_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_title(slide, "What We Built")

    steps = [
        ("01", "Data Loading", "Parse call folders: meeting-info, transcript, summary, speaker-meta."),
        ("02", "Call Type Inference", "Heuristic + LLM hybrid labels support, external, internal."),
        ("03", "Topic Categorization", "HDBSCAN clustering on embeddings, LLM naming, TF-IDF keywords."),
        ("04", "Sentiment Analysis", "Pre-scored sentimentScore (1-5) plus sentence-level labels."),
        ("05", "Bonus Insights", "Churn scoring, feature extraction, escalation chains."),
    ]

    # 3 columns top row, 2 columns bottom row (centered)
    card_w = Inches(2.95)
    card_h = Inches(1.60)
    top_y = Inches(1.15)
    bot_y = Inches(2.90)
    positions = [
        (MARGIN, top_y), (MARGIN + Inches(3.10), top_y), (MARGIN + Inches(6.20), top_y),
        (MARGIN + Inches(1.55), bot_y), (MARGIN + Inches(4.65), bot_y),
    ]

    for i, (num, title, desc) in enumerate(steps):
        x, y = positions[i]
        add_card(slide, x, y, card_w, card_h)
        add_circle_badge(slide, x + Inches(0.18), y + Inches(0.18), Inches(0.48), num, C_SECONDARY)
        add_text(slide, x + Inches(0.75), y + Inches(0.22), Inches(2.05), Inches(0.32),
                 title, FONT_BODY, bold=True, color=C_PRIMARY)
        desc = fit_text(desc, Inches(2.65), FONT_TINY, 4)
        add_text(slide, x + Inches(0.18), y + Inches(0.72), Inches(2.60), Inches(0.78),
                 desc, FONT_TINY, color=C_TEXT)
    check(slide, "Pipeline")
    return slide


def add_dataset_slide(prs, data: PresentationData):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_title(slide, "Dataset Overview",
                    f"{data.total_calls} calls from Aegis dataset · {data.date_min} to {data.date_max}")

    kpis = [
        (data.total_calls, "Total Calls", C_PRIMARY),
        (fmt_num(data.duration_mean), "Avg Duration (min)", C_SECONDARY),
        (fmt_num(data.avg_sentiment), "Avg Sentiment", C_GREEN),
        (f"{data.date_min[-5:]} → {data.date_max[-5:]}", "Date Range", C_ACCENT),
    ]
    box_w = Inches(2.35)
    box_h = Inches(1.0)
    gap = Inches(0.10)
    start_x = MARGIN
    for i, (val, label, color) in enumerate(kpis):
        left = start_x + i * (box_w + gap)
        add_card(slide, left, CONTENT_Y, box_w, box_h)
        val_font = Pt(24) if label == "Date Range" else FONT_KPI
        add_text(slide, left, CONTENT_Y + Inches(0.08), box_w, Inches(0.55),
                 str(val), val_font, bold=True, color=color, align=PP_ALIGN.CENTER)
        add_text(slide, left, CONTENT_Y + Inches(0.63), box_w, Inches(0.28),
                 label, FONT_KPI_LABEL, color=C_LIGHT, align=PP_ALIGN.CENTER)

    if data.total_calls > 0:
        breakdown = (
            f"Support: {data.support_count} calls ({round(100*data.support_count/data.total_calls)}%)   ·   "
            f"External: {data.external_count} calls ({round(100*data.external_count/data.total_calls)}%)   ·   "
            f"Internal: {data.internal_count} calls ({round(100*data.internal_count/data.total_calls)}%)"
        )
    else:
        breakdown = "Support: 0 · External: 0 · Internal: 0"

    add_text(slide, MARGIN, Inches(2.45), CONTENT_W, Inches(0.30),
             breakdown, FONT_BODY, bold=True, color=C_PRIMARY, align=PP_ALIGN.CENTER)

    add_chart_if_exists(slide, "02_call_types_distribution.png", MARGIN + Inches(1.0), Inches(2.85), Inches(7.0), Inches(2.5))
    check(slide, "Dataset")
    return slide


def _topic_insight(name, count, keywords, type_info):
    """Generate a concise insight for a topic cluster (up to 3 lines)."""
    if "Compliance" in name:
        return fit_text(f"Top topic with {count} calls. Dominant in external conversations around ISO 27001, SOC 2, HIPAA.", Inches(3.1), FONT_TINY, 3)
    if "Incident" in name or "Reliability" in name:
        return fit_text(f"Appears in support and external calls — infrastructure incidents drive negative sentiment.", Inches(3.1), FONT_TINY, 3)
    if "Identity" in name or "Access" in name:
        return fit_text(f"Concentrated in internal and support calls: SSO, MFA, RBAC, provisioning issues.", Inches(3.1), FONT_TINY, 3)
    if "Engineering" in name or "Sprint" in name:
        return fit_text(f"Internal engineering conversations: sprint planning, pipeline, QA, roadmap.", Inches(3.1), FONT_TINY, 3)
    if "Sales" in name or "Renewal" in name:
        return fit_text(f"External revenue conversations: renewals, account reviews, compliance wins.", Inches(3.1), FONT_TINY, 3)
    if "Product" in name or "Deployment" in name:
        return fit_text(f"Product rollout and deployment discussions across support and external calls.", Inches(3.1), FONT_TINY, 3)
    kw = ", ".join(keywords[:3])
    return fit_text(f"{count} calls. Keywords: {kw}.", Inches(3.1), FONT_TINY, 3)


def add_topic_slide(prs, data: PresentationData):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_title(slide, "Topic Categorization",
                    "HDBSCAN clustering on 384-dim embeddings. Top categories show where Aegis comes up most.")

    add_chart_if_exists(slide, "03_topic_distribution.png", CHART_X, CONTENT_Y, CHART_W, Inches(3.6))

    add_text(slide, CHART_X, CONTENT_Y + Inches(3.7), CHART_W, Inches(0.30),
             f"{data.hdbscan_noise} calls flagged as noise — HDBSCAN leaves ambiguous transcripts unclustered.",
             FONT_MICRO, color=C_LIGHT)

    clusters = sorted(data.clusters.items(), key=lambda x: x[1].get("count", 0), reverse=True)[:3]
    card_h = Inches(1.25)
    y = CONTENT_Y
    for cid, info in clusters:
        name = info.get("name", f"Cluster {cid}")
        count = info.get("count", 0)
        keywords = info.get("keywords", [])
        icon = TOPIC_ICONS.get(name, "")
        title_text = f"{icon}  {name}" if icon else name
        insight = _topic_insight(name, count, keywords, data.topic_by_call_type)

        add_card(slide, RIGHT_X, y, RIGHT_W, card_h)
        add_text(slide, RIGHT_X + Inches(0.12), y + Inches(0.10), RIGHT_W - Inches(0.24), Inches(0.28),
                 title_text, FONT_BODY, bold=True, color=C_PRIMARY)
        add_text(slide, RIGHT_X + Inches(0.12), y + Inches(0.42), RIGHT_W - Inches(0.24), Inches(0.70),
                 insight, FONT_TINY, color=C_TEXT)
        y += card_h + Inches(0.08)
    check(slide, "Topics")
    return slide


def add_sentiment_slide(prs, data: PresentationData):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_title(slide, "Sentiment Analysis by Call Type")

    add_chart_if_exists(slide, "04_sentiment_trend_by_type.png", CHART_X, CONTENT_Y, CHART_W, Inches(3.5))

    explanations = {
        "support": "Mostly mixed-positive — agents resolve issues, but customers call when things break.",
        "external": "Highest score — compliance wins and renewals drive positive tone with prospects.",
        "internal": "Below neutral — incident postmortems and risk reviews weigh it down.",
    }

    # Widen explanations to 3 lines so nothing is truncated

    types = [
        ("Support", data.sentiment.support_score, data.sentiment.support_neg, explanations["support"], C_RED),
        ("External", data.sentiment.external_score, data.sentiment.external_neg, explanations["external"], C_GREEN),
        ("Internal", data.sentiment.internal_score, data.sentiment.internal_neg, explanations["internal"], C_SECONDARY),
    ]

    card_h = Inches(1.15)
    y = CONTENT_Y
    for name, score, neg_pct, explanation, color in types:
        explanation = fit_text(explanation, RIGHT_W - Inches(0.24), FONT_TINY, 3)
        add_card(slide, RIGHT_X, y, RIGHT_W, card_h)
        add_text(slide, RIGHT_X + Inches(0.12), y + Inches(0.10), RIGHT_W - Inches(0.24), Inches(0.26),
                 name, FONT_BODY, bold=True, color=color)
        add_text(slide, RIGHT_X + Inches(0.12), y + Inches(0.36), RIGHT_W - Inches(0.24), Inches(0.22),
                 f"Score: {fmt_num(score)} / 5  ·  {fmt_pct(neg_pct)}% negative", FONT_TINY, color=C_LIGHT)
        add_text(slide, RIGHT_X + Inches(0.12), y + Inches(0.58), RIGHT_W - Inches(0.24), Inches(0.50),
                 explanation, FONT_TINY, color=C_TEXT)
        y += card_h + Inches(0.08)
    check(slide, "Sentiment")
    return slide


def add_problem_zones_slide(prs, data: PresentationData):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_title(slide, "Where Sentiment Goes Negative",
                    "Average sentiment per call type × topic. Red zones need product or process intervention.")

    add_chart_if_exists(slide, "04_sentiment_boxplot.png", MARGIN, CONTENT_Y, CONTENT_W, Inches(2.5))

    zones = data.problem_zones[:3]
    if not zones:
        zones = [{"topic": "N/A", "call_type": "N/A", "sentiment": 0, "why": "No data"}]

    card_w = Inches(2.95)
    card_h = Inches(1.25)
    base_y = Inches(4.00)
    xs = [MARGIN, MARGIN + Inches(3.05), MARGIN + Inches(6.10)]

    for x, zone in zip(xs, zones):
        why = fit_text(zone.get("why", ""), card_w - Inches(0.24), FONT_TINY, 3)
        add_card(slide, x, base_y, card_w, card_h)
        add_text(slide, x + Inches(0.12), base_y + Inches(0.12), card_w - Inches(0.24), Inches(0.28),
                 f"{zone['topic']} × {zone['call_type'].title()}", FONT_SMALL, bold=True, color=C_RED)
        add_text(slide, x + Inches(0.12), base_y + Inches(0.44), card_w - Inches(0.24), Inches(0.45),
                 f"Sentiment: {fmt_num(zone['sentiment'])} — {why}", FONT_TINY, color=C_TEXT)
    check(slide, "Problem Zones")
    return slide


def add_strong_zones_slide(prs, data: PresentationData):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_title(slide, "Where Sentiment Is Strongest",
                    "Green zones are relationship and product strengths to reinforce.")

    add_chart_if_exists(slide, "04_sentiment_stacked_by_type.png", MARGIN, CONTENT_Y, CONTENT_W, Inches(2.5))

    zones = data.strong_zones[:3]
    if not zones:
        zones = [{"topic": "N/A", "call_type": "N/A", "sentiment": 0, "why": "No data"}]

    card_w = Inches(2.95)
    card_h = Inches(1.25)
    base_y = Inches(4.00)
    xs = [MARGIN, MARGIN + Inches(3.05), MARGIN + Inches(6.10)]

    for x, zone in zip(xs, zones):
        why = fit_text(zone.get("why", ""), card_w - Inches(0.24), FONT_TINY, 3)
        add_card(slide, x, base_y, card_w, card_h)
        add_text(slide, x + Inches(0.12), base_y + Inches(0.12), card_w - Inches(0.24), Inches(0.28),
                 f"{zone['topic']} × {zone['call_type'].title()}", FONT_SMALL, bold=True, color=C_GREEN)
        add_text(slide, x + Inches(0.12), base_y + Inches(0.44), card_w - Inches(0.24), Inches(0.45),
                 f"Sentiment: {fmt_num(zone['sentiment'])} — {why}", FONT_TINY, color=C_TEXT)
    check(slide, "Strong Zones")
    return slide


def _format_signal(signal: str) -> str:
    return signal.replace("_", " ").title()


def add_churn_slide(prs, data: PresentationData):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_title(slide, "Churn Risk Detection",
                    "Keyword-based churn scoring (0-10). High score + low sentiment = immediate attention.")

    add_chart_if_exists(slide, "05_churn_risk_distribution.png", CHART_X, CONTENT_Y, CHART_W, Inches(4.0))

    add_text(slide, RIGHT_X, CONTENT_Y, RIGHT_W, Inches(0.30),
             f"⚠  Flagged Accounts ({data.risk_distribution.high} high risk)", FONT_BODY, bold=True, color=C_RED)

    card_w = Inches(3.4)
    card_h = Inches(1.70)
    y = CONTENT_Y + Inches(0.38)

    for account in data.churn_narratives[:2]:
        add_card(slide, RIGHT_X, y, card_w, card_h)
        title = fit_text(account.get("title", "Unknown"), card_w - Inches(0.24), FONT_SMALL, 2)
        add_text(slide, RIGHT_X + Inches(0.12), y + Inches(0.12), card_w - Inches(0.24), Inches(0.45),
                 title, FONT_SMALL, bold=True, color=C_TEXT)
        add_text(slide, RIGHT_X + Inches(0.12), y + Inches(0.60), card_w - Inches(0.24), Inches(0.22),
                 f"Churn signal: {min(account.get('churn_score', 0), 10)}/10  |  Score: {account.get('sentiment_score', 0)}/5",
                 FONT_TINY, color=C_LIGHT)

        raw_signals = account.get("signals", [])
        if isinstance(raw_signals, str):
            signals = [s.strip() for s in raw_signals.split(",") if s.strip()]
        elif isinstance(raw_signals, dict):
            signals = list(raw_signals.keys())
        else:
            signals = list(raw_signals)
        signal_text = "  |  ".join(_format_signal(s) for s in signals[:3])
        signal_text = fit_text(signal_text, card_w - Inches(0.24), FONT_MICRO, 3)
        add_text(slide, RIGHT_X + Inches(0.12), y + Inches(0.86), card_w - Inches(0.24), Inches(0.75),
                 signal_text, FONT_MICRO, color=C_TEXT)
        y += card_h + Inches(0.10)
    check(slide, "Churn")
    return slide


def add_feature_slide(prs, data: PresentationData):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    total_signals = sum(data.feature_keywords.values())
    add_slide_title(slide, "Feature Request Intelligence",
                    f"{total_signals} feature-request signals extracted. Direct input for PM prioritization.")

    add_chart_if_exists(slide, "05_feature_requests.png", CHART_X, CONTENT_Y, CHART_W, Inches(3.0))

    add_text(slide, RIGHT_X, CONTENT_Y, RIGHT_W, Inches(0.28),
             "PM-Ready Backlog Items", FONT_BODY, bold=True, color=C_PRIMARY)

    top_features = list(data.feature_keywords.items())[:5]
    # Priority rule derived from rank: #1 = P1, #2-3 = P2, rest = P3
    def prio_for_rank(i):
        return "P1" if i == 0 else ("P2" if i <= 2 else "P3")

    display_names = {"sso": "SSO", "mfa": "MFA", "ldap": "LDAP", "saml": "SAML"}

    card_h = Inches(0.65)
    y = CONTENT_Y + Inches(0.38)
    for i, (kw, count) in enumerate(top_features):
        prio = prio_for_rank(i)
        prio_color = C_RED if prio == "P1" else (C_AMBER if prio == "P2" else C_GREEN)

        add_card(slide, RIGHT_X, y, RIGHT_W, card_h)
        add_circle_badge(slide, RIGHT_X + Inches(0.10), y + Inches(0.14), Inches(0.36), prio, prio_color, FONT_TINY)

        kw_display = display_names.get(kw.lower(), kw.title())
        add_text(slide, RIGHT_X + Inches(0.54), y + Inches(0.10), Inches(1.45), Inches(0.24),
                 kw_display, FONT_SMALL, bold=True, color=C_TEXT)
        add_text(slide, RIGHT_X + Inches(2.0), y + Inches(0.10), Inches(1.25), Inches(0.24),
                 f"{count} mentions", FONT_TINY, color=C_LIGHT, align=PP_ALIGN.RIGHT)

        desc = fit_text(f"Mentioned {count} times across calls.", Inches(2.75), FONT_TINY, 2)
        add_text(slide, RIGHT_X + Inches(0.54), y + Inches(0.35), Inches(2.75), Inches(0.24),
                 desc, FONT_TINY, color=C_TEXT)

        y += card_h + Inches(0.06)
    check(slide, "Features")
    return slide


def add_action_items_slide(prs, data: PresentationData):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_title(slide, "Action Items & Call Efficiency",
                    "Follow-through commitments by call type — a proxy for engagement and urgency.")

    add_chart_if_exists(slide, "05_action_items_by_type.png", MARGIN, CONTENT_Y, Inches(4.8), Inches(2.9))

    items = []
    for ctype in ["external", "support", "internal"]:
        info = data.action_items.get(ctype, {})
        avg = info.get("avg_per_call", 0)
        total = info.get("action_mentions", 0)
        items.append((ctype.title(), avg, total))

    # Sort by avg action items so the headline reflects the actual leader
    items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
    leader_type, leader_avg, _ = items_sorted[0]
    second_type, second_avg, _ = items_sorted[1]
    third_type, third_avg, _ = items_sorted[2]
    insights = [
        (f"{leader_type} calls drive follow-through", f"{leader_avg:.1f} avg action items per {leader_type.lower()} call — highest of all call types."),
        (f"{second_type} calls: {second_avg:.1f} avg", f"Middle performer — follow-ups vary by {second_type.lower()} context."),
        (f"{third_type} calls: {third_avg:.1f} avg", "Fewest items but may still be high urgency: engineering fixes and postmortems."),
        ("Opportunity: action item tracking", "Cross-reference action items across call sequences for closed-loop accountability."),
    ]

    card_w = Inches(4.2)
    card_h = Inches(0.90)
    x = Inches(5.4)
    y = CONTENT_Y
    for title, body in insights:
        body = fit_text(body, card_w - Inches(0.24), FONT_TINY, 3)
        add_card(slide, x, y, card_w, card_h)
        add_text(slide, x + Inches(0.12), y + Inches(0.10), card_w - Inches(0.24), Inches(0.24),
                 title, FONT_SMALL, bold=True, color=C_SECONDARY)
        add_text(slide, x + Inches(0.12), y + Inches(0.36), card_w - Inches(0.24), Inches(0.45),
                 body, FONT_TINY, color=C_TEXT)
        y += card_h + Inches(0.08)
    check(slide, "Action Items")
    return slide


def add_recommendations_slide(prs, data: PresentationData):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_title(slide, "Recommendations")

    external_avg = data.action_items.get("external", {}).get("avg_per_call", 0)
    top_feature = next(iter(data.feature_keywords.items()), ("feature requests", 0))
    worst_zone = data.problem_zones[0] if data.problem_zones else {"topic": "reliability"}

    recs = [
        ("1", "[Product]", f"Close the '{top_feature[0]}' gap", f"'{top_feature[0].title()}' is the #1 signal ({top_feature[1]} mentions). Fast win."),
        ("2", "[Product]", "Add Excel/CSV audit report export", "Auditor tooling mismatch drives audit friction."),
        ("3", "[Engineering]", f"Prioritize fixes in '{worst_zone.get('topic', 'reliability')}'", f"{worst_zone.get('topic', 'Problem zone')} × {worst_zone.get('call_type', 'support').title()} is the lowest sentiment zone."),
        ("4", "[Sales/CS]", f"Monitor {data.risk_distribution.high} high-risk accounts", "Automated churn scoring catches signals weeks earlier."),
        ("5", "[Support]", "Coach reps on problem-zone calls", "Focus on incident handling and expectation-setting."),
        ("6", "[Analytics]", "Build action item tracking", f"External calls avg {external_avg} actions. Closed-loop tracking is highest-value next feature."),
    ]

    card_w = Inches(4.55)
    card_h = Inches(1.25)
    col_x = [MARGIN, MARGIN + Inches(4.75)]
    start_y = CONTENT_Y

    for i, (num, tag, title, desc) in enumerate(recs):
        col = i % 2
        row = i // 2
        x = col_x[col]
        y = start_y + row * (card_h + Inches(0.10))

        add_card(slide, x, y, card_w, card_h)
        add_circle_badge(slide, x + Inches(0.12), y + Inches(0.12), Inches(0.42), num, C_SECONDARY, FONT_SMALL)
        add_text(slide, x + Inches(0.62), y + Inches(0.10), Inches(1.2), Inches(0.22),
                 tag, FONT_TINY, bold=True, color=C_SECONDARY)
        title = fit_text(title, card_w - Inches(0.70), FONT_SMALL, 2)
        add_text(slide, x + Inches(0.62), y + Inches(0.32), card_w - Inches(0.70), Inches(0.45),
                 title, FONT_SMALL, bold=True, color=C_TEXT)
        desc = fit_text(desc, card_w - Inches(0.70), FONT_TINY, 3)
        add_text(slide, x + Inches(0.62), y + Inches(0.79), card_w - Inches(0.70), Inches(0.38),
                 desc, FONT_TINY, color=C_TEXT)
    check(slide, "Recommendations")
    return slide


def add_methodology_slide(prs, data: PresentationData):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_title(slide, "Appendix: Pipeline Methodology")

    cards = [
        ("Data Sources", f"{data.total_calls} calls from Aegis dataset: meeting-info, summary, transcript, speaker-meta."),
        ("Sentiment Approach", "sentimentScore (1-5) from summary.json — pre-labeled. Sentence-level sentimentType labels. No LLM re-scoring."),
        ("Topic & Churn", f"HDBSCAN clustering (silhouette: {fmt_num(data.hdbscan_score, 3)}). {data.hdbscan_noise} calls noise. LLM naming + TF-IDF. Churn is rule-based and illustrative."),
        ("Limitations", f"Silhouette {fmt_num(data.hdbscan_score, 3)} is weak (ideal > 0.5); {round(100*data.hdbscan_noise/data.total_calls) if data.total_calls else 0}% noise rate; N={data.total_calls} means trends are suggestive."),
    ]

    card_w = Inches(4.55)
    card_h = Inches(1.80)
    positions = [
        (MARGIN, CONTENT_Y),
        (MARGIN + Inches(4.75), CONTENT_Y),
        (MARGIN, CONTENT_Y + Inches(1.95)),
        (MARGIN + Inches(4.75), CONTENT_Y + Inches(1.95)),
    ]

    for (x, y), (title, body) in zip(positions, cards):
        add_card(slide, x, y, card_w, card_h)
        add_text(slide, x + Inches(0.12), y + Inches(0.12), card_w - Inches(0.24), Inches(0.28),
                 title, FONT_BODY, bold=True, color=C_PRIMARY)
        body = fit_text(body, card_w - Inches(0.24), FONT_TINY, 5)
        add_text(slide, x + Inches(0.12), y + Inches(0.45), card_w - Inches(0.24), Inches(1.25),
                 body, FONT_TINY, color=C_TEXT)
    check(slide, "Methodology")
    return slide


def add_charts_appendix_slide(prs, data: PresentationData):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_slide_title(slide, "Appendix: Additional Charts", "Evidence behind the findings")

    chart_w = Inches(4.4)
    chart_h = Inches(1.9)
    gap_x = Inches(0.2)
    gap_y = Inches(0.15)
    positions = [
        (MARGIN, CONTENT_Y),
        (MARGIN + chart_w + gap_x, CONTENT_Y),
        (MARGIN, CONTENT_Y + chart_h + gap_y),
        (MARGIN + chart_w + gap_x, CONTENT_Y + chart_h + gap_y),
    ]
    charts = [
        "05_churn_score_histogram.png",
        "04_negative_sentiment_trend.png",
        "03_clustering_comparison.png",
        "01_duration_distribution.png",
    ]
    for (x, y), filename in zip(positions, charts):
        add_chart_if_exists(slide, filename, x, y, chart_w, chart_h)
    check(slide, "Charts Appendix")
    return slide


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("06 GENERATE PPT: Building presentation")
    print("=" * 60)

    try:
        prs = Presentation()
        prs.slide_width = SLIDE_WIDTH
        prs.slide_height = SLIDE_HEIGHT

        data = load_presentation_data(OUTPUT_DIR)
        if data.warnings:
            print("\nData validation warnings:")
            for w in data.warnings:
                print(f"  - {w}")

        slides = [
            ("Title + KPIs", add_title_slide, True),
            ("Executive Summary", add_executive_summary_slide, True),
            ("Pipeline", add_pipeline_slide, False),
            ("Dataset", add_dataset_slide, True),
            ("Topics", add_topic_slide, True),
            ("Sentiment", add_sentiment_slide, True),
            ("Problem Zones", add_problem_zones_slide, True),
            ("Strong Zones", add_strong_zones_slide, True),
            ("Churn", add_churn_slide, True),
            ("Features", add_feature_slide, True),
            ("Action Items", add_action_items_slide, True),
            ("Recommendations", add_recommendations_slide, True),
            ("Methodology", add_methodology_slide, True),
            ("Charts Appendix", add_charts_appendix_slide, True),
        ]

        for i, (name, func, needs_data) in enumerate(slides, 1):
            func(prs, data) if needs_data else func(prs)
            print(f"  Slide {i}: {name}")

        output_path = OUTPUT_DIR / "Transcript_Intelligence_Report.pptx"
        prs.save(output_path)
        print("\n" + "=" * 60)
        print("06 GENERATE PPT: Done")
        print(f"Presentation saved: {output_path}")
        print(f"Total slides: {len(prs.slides)}")
        print("=" * 60)
    except FileNotFoundError as e:
        print(f"\nERROR: Missing required analysis output. Run scripts 01-05 first.\n  {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: PPT generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
