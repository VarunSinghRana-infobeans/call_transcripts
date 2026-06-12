"""
test_smoke.py

Quick smoke tests to catch the most critical bugs before panel review.
Run with: python -m pytest tests/test_smoke.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_utils_imports_without_errors():
    """utils.py should import without hardcoded path errors."""
    import utils
    assert utils.DATASET_DIR is not None
    assert utils.OUTPUT_DIR is not None
    # OUTPUT_DIR should be relative, not a hardcoded Windows path
    assert "d:/call_trannscript" not in str(utils.OUTPUT_DIR).lower()
    # Should point to the output directory under the notebook folder
    assert "output" in str(utils.OUTPUT_DIR).lower()


def test_ai_config_providers_exist():
    """All three AI providers should be importable and instantiable."""
    import ai_config

    mock = ai_config.MockProvider()
    assert mock.is_available()
    assert mock.classify("test support ticket") == "support"
    assert mock.classify("test sales renewal") == "external"

    openai = ai_config.OpenAIProvider()
    # Should exist even without API key
    assert openai.is_available() == bool(os.environ.get("OPENAI_API_KEY"))

    ollama = ai_config.OllamaProvider()
    assert hasattr(ollama, "base_url")


def test_ai_config_mock_generate_returns_sensible():
    """Mock provider should return sensible topic names."""
    import ai_config

    mock = ai_config.MockProvider()
    name = mock.generate("Name this cluster. Keywords: sso, mfa, identity")
    assert "Identity" in name or "Access" in name


def test_start_py_scripts_list_is_complete():
    """start.py should reference all 6 scripts."""
    start_path = Path(__file__).parent.parent / "start.py"
    content = start_path.read_text()
    for num in ["01", "02", "03", "04", "05", "06"]:
        assert f"{num}_" in content, f"Script {num} missing from start.py"


def test_no_hardcoded_windows_paths_in_scripts():
    """Scripts should use relative paths, not absolute Windows paths."""
    script_dir = Path(__file__).parent.parent / "scripts"
    for py_file in script_dir.glob("*.py"):
        content = py_file.read_text()
        # Allow Path("d:/...") in comments but not in actual code
        lines = content.split("\n")
        in_docstring = False
        docstring_char = None
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Track multi-line docstrings
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if stripped.count('"""') < 2 and stripped.count("'''") < 2:
                    in_docstring = not in_docstring
                    docstring_char = stripped[:3]
                continue
            if in_docstring:
                continue
            # Skip single-line comments
            if stripped.startswith("#"):
                continue
            if "d:/call_trannscript" in line.lower():
                raise AssertionError(
                    f"Hardcoded Windows path in {py_file.name}:{i}: {line.strip()}"
                )


def test_all_scripts_exist():
    """All 6 analysis scripts should exist."""
    base = Path(__file__).parent.parent
    for num in ["01", "02", "03", "04", "05", "06"]:
        script = base / f"{num}_explore.py" if num == "01" else \
                 base / f"{num}_call_types.py" if num == "02" else \
                 base / f"{num}_topic_modeling.py" if num == "03" else \
                 base / f"{num}_sentiment.py" if num == "04" else \
                 base / f"{num}_bonus_insights.py" if num == "05" else \
                 base / f"{num}_generate_ppt.py"
        assert script.exists(), f"Missing script: {script.name}"


def test_ppt_script_reads_json_outputs():
    """PPT script should reference JSON files, not hardcode all stats."""
    ppt_path = Path(__file__).parent.parent / "06_generate_ppt.py"
    content = ppt_path.read_text()
    # Either reads JSON directly or uses ppt_data module
    assert ("json.load" in content or "json.loads" in content or
            "ppt_data" in content), \
        "PPT script should read JSON outputs dynamically"


def test_dataset_dir_falls_back_gracefully():
    """Dataset directory should fallback if default path doesn't exist."""
    import utils
    # Should not crash even if dataset is missing
    # (it might warn, but shouldn't raise)
    assert isinstance(utils.DATASET_DIR, Path)


def test_feature_keywords_not_filler_words():
    """Feature keywords should exclude conversational filler like 'want', 'need'."""
    insights_path = Path(__file__).parent.parent / "05_bonus_insights.py"
    lines = insights_path.read_text().split("\n")
    in_docstring = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if stripped.count('"""') < 2 and stripped.count("'''") < 2:
                in_docstring = not in_docstring
            continue
        if in_docstring or stripped.startswith("#"):
            continue
        # Check that 'want' and 'need' are not standalone keywords in code
        assert '"want"' not in line, "'want' is a conversational filler, not a feature keyword"
        assert '"need"' not in line, "'need' is a conversational filler, not a feature keyword"


if __name__ == "__main__":
    print("Running smoke tests...")
    test_utils_imports_without_errors()
    print("  OK: utils imports")
    test_ai_config_providers_exist()
    print("  OK: AI providers")
    test_ai_config_mock_generate_returns_sensible()
    print("  OK: Mock provider")
    test_start_py_scripts_list_is_complete()
    print("  OK: start.py scripts list")
    test_no_hardcoded_windows_paths_in_scripts()
    print("  OK: No hardcoded paths")
    test_all_scripts_exist()
    print("  OK: All scripts exist")
    test_ppt_script_reads_json_outputs()
    print("  OK: PPT reads JSON")
    test_dataset_dir_falls_back_gracefully()
    print("  OK: Dataset dir fallback")
    test_feature_keywords_not_filler_words()
    print("  OK: Feature keywords cleaned")
    print("\nAll smoke tests passed!")
