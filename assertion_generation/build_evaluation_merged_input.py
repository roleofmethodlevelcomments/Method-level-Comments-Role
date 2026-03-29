#!/usr/bin/env python3
"""
Build a single merged JSON for assertion effectiveness evaluation.

Merges:
  - Dataset (e.g. strengthened_comments_full.json): provides test_case_without_assertions,
    bug_report.bug_id, bug_report.project_name, etc.
  - One oracle JSON per mode: each has bug_id and generated_oracle.generated_assertions.

Output: evaluation_merged.json with:
  - assertion_free_test_code (from dataset)
  - generated_assertions_by_mode: { "code_only": "...", "original_comment": "...", ... }

Mode name mapping (you can change this):
  - code_only           <- step4_without_comments
  - original_comment     <- step5_with_comments
  - updated_comment      <- step5_with_postfix_comments
  - strengthened_comment <- step5_with_strengthened_comments

Usage:
  python build_evaluation_merged_input.py \\
    --dataset strengthened_comments_full.json \\
    --oracle-code-only llm_generated_oracles_step4_without_comments_deepseek_hybrid_balanced.json \\
    --oracle-original llm_generated_oracles_step5_with_comments_deepseek_hybrid_balanced.json \\
    --oracle-updated llm_generated_oracles_step5_with_postfix_comments_deepseek_hybrid_balanced.json \\
    --oracle-strengthened llm_generated_oracles_step5_with_strengthened_comments_deepseek_hybrid_balanced.json \\
    --output evaluation_merged.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def get_bug_id(entry: Dict[str, Any]) -> str:
    br = entry.get("bug_report") or {}
    return str(br.get("bug_id", "")).strip()


def get_assertion_free_test(entry: Dict[str, Any]) -> str:
    s = entry.get("test_case_without_assertions") or ""
    return s.strip() if s else ""


def load_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def load_oracle_by_bug_id(path: Path) -> Dict[str, str]:
    """Load oracle file; return bug_id -> generated_assertions text."""
    entries = load_json(path)
    out = {}
    for e in entries:
        bid = get_bug_id(e)
        if not bid:
            continue
        assertions = (e.get("generated_oracle") or {}).get("generated_assertions", "")
        out[bid] = (assertions or "").strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build merged evaluation input from dataset + oracle files.")
    ap.add_argument("--dataset", type=Path, required=True, help="Dataset JSON (e.g. strengthened_comments_full.json)")
    ap.add_argument("--oracle-code-only", type=Path, help="Oracle JSON for step4 / code only")
    ap.add_argument("--oracle-original", type=Path, help="Oracle JSON for step5 with original comment")
    ap.add_argument("--oracle-updated", type=Path, help="Oracle JSON for step5 with postfix/updated comment")
    ap.add_argument("--oracle-strengthened", type=Path, help="Oracle JSON for step5 with strengthened comment")
    ap.add_argument("--output", type=Path, default=Path("evaluation_merged.json"), help="Output merged JSON")
    args = ap.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    dataset = load_json(args.dataset)
    mode_files = [
        ("code_only", args.oracle_code_only),
        ("original_comment", args.oracle_original),
        ("updated_comment", args.oracle_updated),
        ("strengthened_comment", args.oracle_strengthened),
    ]
    mode_files = [(name, p) for name, p in mode_files if p and p.exists()]

    oracles_by_mode: Dict[str, Dict[str, str]] = {}
    for mode_name, path in mode_files:
        oracles_by_mode[mode_name] = load_oracle_by_bug_id(path)

    merged: List[Dict[str, Any]] = []
    for entry in dataset:
        bug_id = get_bug_id(entry)
        assertion_free = get_assertion_free_test(entry)
        if not bug_id or not assertion_free:
            continue
        by_mode = {}
        for mode_name, by_bid in oracles_by_mode.items():
            if bug_id in by_bid:
                by_mode[mode_name] = by_bid[bug_id]
        if not by_mode:
            continue
        new_entry = dict(entry)
        new_entry["assertion_free_test_code"] = assertion_free
        new_entry["generated_assertions_by_mode"] = by_mode
        merged.append(new_entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(merged)} entries to {args.output}")
    print("Modes included:", list(set().union(*(e["generated_assertions_by_mode"].keys() for e in merged))))


if __name__ == "__main__":
    main()
