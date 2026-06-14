"""
test_ppt_data.py

Tests that PPT-bound statistics match the source JSON/CSV outputs.
This catches data-narrative mismatches before the panel sees them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def get_latest_run_dir() -> Path:
    """Find the most recent pipeline output folder."""
    output_base = Path(__file__).parent.parent / "output"
    run_dirs = sorted(
        [d for d in output_base.iterdir() if d.is_dir() and d.name.startswith("Transcript_Intelligence_")],
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError("No Transcript_Intelligence output folder found")
    return run_dirs[0]


def test_load_presentation_data_runs():
    """ppt_data.load_presentation_data should load without crashing."""
    from ppt_data import load_presentation_data

    data = load_presentation_data(get_latest_run_dir())
    assert data.total_calls == 100
    assert data.support_count + data.external_count + data.internal_count == 100


def test_sentiment_numbers_match_json():
    """Sentiment numbers in ppt_data must match 04_sentiment_stats.json."""
    import json
    from ppt_data import load_presentation_data

    run_dir = get_latest_run_dir()
    data = load_presentation_data(run_dir)
    path = run_dir / "04_sentiment_stats.json"
    raw = json.loads(path.read_text())

    by_type = raw["by_type"]
    assert data.sentiment.support_score == by_type["sentiment_score"]["support"]
    assert data.sentiment.external_score == by_type["sentiment_score"]["external"]
    assert data.sentiment.internal_score == by_type["sentiment_score"]["internal"]


def test_risk_distribution_matches_json():
    """Churn counts in ppt_data must match 05_churn_scores.json."""
    import json
    from ppt_data import load_presentation_data

    run_dir = get_latest_run_dir()
    data = load_presentation_data(run_dir)
    path = run_dir / "05_churn_scores.json"
    raw = json.loads(path.read_text())

    assert data.risk_distribution.high == raw["risk_distribution"]["High"]
    assert data.risk_distribution.medium == raw["risk_distribution"]["Medium"]
    assert data.risk_distribution.low == raw["risk_distribution"]["Low"]


def test_feature_keywords_match_json():
    """Feature keyword counts in ppt_data must match 05_feature_requests.json."""
    import json
    from ppt_data import load_presentation_data

    run_dir = get_latest_run_dir()
    data = load_presentation_data(run_dir)
    path = run_dir / "05_feature_requests.json"
    raw = json.loads(path.read_text())

    assert data.feature_keywords == raw["top_keywords"]


def test_business_taxonomy_loaded():
    """Business taxonomy should be loaded from topics.json."""
    import json
    from ppt_data import load_presentation_data

    run_dir = get_latest_run_dir()
    data = load_presentation_data(run_dir)
    path = run_dir / "topics.json"
    raw = json.loads(path.read_text())

    biz = raw.get("business_taxonomy", {})
    if biz:
        assert data.business_taxonomy.get("top_categories") is not None
        total_categorized = sum(biz.get("primary_counts", {}).values())
        assert total_categorized > 0


def test_carry_forward_actions_loaded():
    """Carry-forward action totals should match 05_escalations.json."""
    import json
    from ppt_data import load_presentation_data

    run_dir = get_latest_run_dir()
    data = load_presentation_data(run_dir)
    path = run_dir / "05_escalations.json"
    raw = json.loads(path.read_text())

    assert data.carry_forward_total == raw.get("carry_forward_total", 0)
    if data.carry_forward_total:
        assert sum(data.carry_forward_actions[c]["count"] for c in ["support", "external", "internal"]) == data.carry_forward_total


def test_no_hardcoded_sentiment_in_ppt_script():
    """06_generate_ppt.py must not contain old hardcoded sentiment values."""
    ppt_script = Path(__file__).parent.parent / "06_generate_ppt.py"
    content = ppt_script.read_text(encoding="utf-8")
    assert "2.72" not in content, "Old hardcoded support sentiment 2.72 found"
    assert "29%" not in content, "Old hardcoded negative percentage found"


def test_ppt_data_warns_on_missing_files():
    """ppt_data should report warnings when source files are missing."""
    from ppt_data import load_presentation_data
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        data = load_presentation_data(Path(tmp))
        assert len(data.warnings) > 0


if __name__ == "__main__":
    print("Running PPT data tests...")
    test_load_presentation_data_runs()
    print("  OK: load_presentation_data")
    test_sentiment_numbers_match_json()
    print("  OK: sentiment numbers match")
    test_risk_distribution_matches_json()
    print("  OK: risk distribution matches")
    test_feature_keywords_match_json()
    print("  OK: feature keywords match")
    test_business_taxonomy_loaded()
    print("  OK: business taxonomy loaded")
    test_carry_forward_actions_loaded()
    print("  OK: carry-forward actions loaded")
    test_no_hardcoded_sentiment_in_ppt_script()
    print("  OK: no hardcoded sentiment in PPT script")
    test_ppt_data_warns_on_missing_files()
    print("  OK: warnings on missing files")
    print("\nAll PPT data tests passed!")
