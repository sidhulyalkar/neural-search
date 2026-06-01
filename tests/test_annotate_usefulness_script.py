# tests/test_annotate_usefulness_script.py
import json
import subprocess
import sys
from pathlib import Path


def test_script_exists_and_is_importable():
    """Script must exist and have no syntax errors."""
    result = subprocess.run(
        [sys.executable, "-c", "import ast; ast.parse(open('scripts/annotate_usefulness.py').read())"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_script_dry_run_exits_cleanly(tmp_path):
    """With --dry-run, script should print pairs without writing anything."""
    pairs_file = tmp_path / "pairs.jsonl"
    original_content = json.dumps({
        "query_id": "q001",
        "query": "mouse decision-making",
        "intent": "replication",
        "candidate_id": "dandi:000001",
        "usefulness_label": "useful",
        "label_type": "seed",
        "notes": "",
    }) + "\n"
    pairs_file.write_text(original_content)
    result = subprocess.run(
        [sys.executable, "scripts/annotate_usefulness.py",
         "--file", str(pairs_file), "--dry-run"],
        capture_output=True, text=True, input="",
        cwd="/mnt/c/Users/sidso/Documents/neural-search",
    )
    assert result.returncode == 0
    content_after = pairs_file.read_text()
    assert content_after == original_content, "dry-run must not modify the file"
