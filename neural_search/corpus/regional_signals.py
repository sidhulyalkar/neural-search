"""Mine reviewable regional enrichment signals from existing corpus text."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from neural_search.corpus.regional_map import (
    DEFAULT_INPUTS,
    DEFAULT_TARGETS,
    RecordLike,
    _as_mapping,
    _compile_alias_pattern,
    _label_values,
    _record_id,
    _record_text,
    _slug,
    load_dataset_records,
    load_region_targets,
)

DEFAULT_RULES = Path("data/config/regional_signal_rules.yaml")
DEFAULT_OUT_DIR = Path("data/corpus/enrichment/regional_signals")
BROAD_REGIONS = {"cortex", "whole_brain"}


def load_signal_rules(path: str | Path = DEFAULT_RULES) -> list[dict[str, Any]]:
    """Load configured regional signal rules."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    rules = payload.get("regional_signal_rules", [])
    if not isinstance(rules, list):
        raise ValueError("regional signal rules must define regional_signal_rules list")
    normalized: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_id = _slug(str(rule.get("id", "")))
        regions = [_slug(str(region)) for region in rule.get("regions", []) if str(region).strip()]
        if not rule_id or not regions:
            continue
        normalized.append(
            {
                "id": rule_id,
                "regions": regions,
                "confidence": float(rule.get("confidence", 0.6)),
                "include_all": [str(item).lower() for item in rule.get("include_all", [])],
                "include_any": [str(item).lower() for item in rule.get("include_any", [])],
                "patterns": [str(item) for item in rule.get("patterns", [])],
                "exclude": [str(item).lower() for item in rule.get("exclude", [])],
                "evidence_tier": str(rule.get("evidence_tier") or "title_description_candidate"),
                "rationale": str(rule.get("rationale") or ""),
            },
        )
    return normalized


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term and term in text]


def _matched_patterns(text: str, patterns: list[str]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(pattern)
    return matches


def _rule_matches(text: str, rule: dict[str, Any]) -> tuple[bool, list[str]]:
    if _matched_terms(text, rule["exclude"]):
        return False, []
    all_terms = _matched_terms(text, rule["include_all"])
    if len(all_terms) != len(rule["include_all"]):
        return False, []
    any_terms = _matched_terms(text, rule["include_any"])
    pattern_terms = _matched_patterns(text, rule["patterns"])
    if rule["include_any"] and not any_terms:
        return False, []
    if rule["patterns"] and not pattern_terms:
        return False, []
    if not rule["include_all"] and not rule["include_any"] and not rule["patterns"]:
        return False, []
    return True, [*all_terms, *any_terms, *pattern_terms]


def _atlas_suggestions(record: RecordLike, targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text = _record_text(record).lower()
    suggestions: list[dict[str, Any]] = []
    for target in targets:
        matched_aliases = [
            alias
            for alias in target.get("aliases", [])
            if _compile_alias_pattern(alias).search(text)
        ]
        if not matched_aliases:
            continue
        region = str(target["id"])
        confidence = 0.55 if region in BROAD_REGIONS else 0.74
        suggestions.append(
            {
                "region": region,
                "confidence": confidence,
                "evidence_tier": "atlas_alias_candidate",
                "evidence": matched_aliases[:5],
                "rule_id": "atlas_alias",
                "rationale": f"Matched atlas aliases for {region}.",
            },
        )
    return suggestions


def _rule_suggestions(record: RecordLike, rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text = _record_text(record).lower()
    suggestions: list[dict[str, Any]] = []
    for rule in rules:
        matches, evidence = _rule_matches(text, rule)
        if not matches:
            continue
        for region in rule["regions"]:
            suggestions.append(
                {
                    "region": region,
                    "confidence": rule["confidence"],
                    "evidence_tier": rule["evidence_tier"],
                    "evidence": evidence[:5],
                    "rule_id": rule["id"],
                    "rationale": rule["rationale"],
                },
            )
    return suggestions


def suggest_record_regions(
    record: RecordLike,
    targets: list[dict[str, Any]],
    rules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return deduplicated regional suggestions absent from verified labels."""

    payload = _as_mapping(record)
    verified = set(_label_values(payload.get("brain_regions")))
    suggestions = [*_atlas_suggestions(record, targets), *_rule_suggestions(record, rules)]
    by_region: dict[str, dict[str, Any]] = {}
    for suggestion in suggestions:
        region = suggestion["region"]
        if region in verified:
            continue
        current = by_region.get(region)
        if current is None or suggestion["confidence"] > current["confidence"]:
            by_region[region] = suggestion
    return sorted(by_region.values(), key=lambda item: (-item["confidence"], item["region"]))


def build_regional_signal_overlay(
    records: list[RecordLike],
    targets: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    *,
    min_confidence: float = 0.0,
    include_verified_records: bool = False,
) -> list[dict[str, Any]]:
    """Build a reviewable regional enrichment overlay."""

    overlay: list[dict[str, Any]] = []
    for record in records:
        payload = _as_mapping(record)
        verified_regions = _label_values(payload.get("brain_regions"))
        if verified_regions and not include_verified_records:
            continue
        suggestions = [
            suggestion
            for suggestion in suggest_record_regions(record, targets, rules)
            if suggestion["confidence"] >= min_confidence
        ]
        if not suggestions:
            continue
        overlay.append(
            {
                "record_id": _record_id(record),
                "source": str(payload.get("source") or ""),
                "source_id": str(payload.get("source_id") or ""),
                "title": str(payload.get("title") or ""),
                "has_verified_regions": bool(verified_regions),
                "verified_regions": verified_regions,
                "suggested_regions": suggestions,
            },
        )
    return overlay


def build_acquisition_backlog(
    overlay: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank suggested regions by potential corpus-coverage gain."""

    target_system = {str(target["id"]): str(target["system"]) for target in targets}
    counts: Counter[str] = Counter()
    high_confidence: Counter[str] = Counter()
    sources: dict[str, set[str]] = {}
    examples: dict[str, list[str]] = {}
    for item in overlay:
        for suggestion in item["suggested_regions"]:
            region = suggestion["region"]
            counts[region] += 1
            if suggestion["confidence"] >= 0.75:
                high_confidence[region] += 1
            sources.setdefault(region, set()).add(item["source"])
            examples.setdefault(region, [])
            if len(examples[region]) < 5:
                examples[region].append(item["record_id"])

    backlog = [
        {
            "region": region,
            "system": target_system.get(region, "unassigned"),
            "suggested_record_count": count,
            "high_confidence_count": high_confidence[region],
            "sources": sorted(sources.get(region, set())),
            "example_records": examples.get(region, []),
            "priority": _priority(count, high_confidence[region], region),
        }
        for region, count in counts.items()
    ]
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return sorted(
        backlog,
        key=lambda item: (
            priority_order[item["priority"]],
            -int(item["high_confidence_count"]),
            -int(item["suggested_record_count"]),
            item["region"],
        ),
    )


def _priority(total: int, high_confidence: int, region: str) -> str:
    if high_confidence >= 5 or (region not in BROAD_REGIONS and total >= 8):
        return "critical"
    if high_confidence >= 2 or (region not in BROAD_REGIONS and total >= 4):
        return "high"
    if total >= 2:
        return "medium"
    return "low"


def render_signal_report(overlay: list[dict[str, Any]], backlog: list[dict[str, Any]]) -> str:
    """Render a compact Markdown report for regional suggestions."""

    region_counter: Counter[str] = Counter()
    high_confidence_records = set()
    record_ids = {item["record_id"] for item in overlay}
    for item in overlay:
        if any(suggestion["confidence"] >= 0.75 for suggestion in item["suggested_regions"]):
            high_confidence_records.add(item["record_id"])
        for suggestion in item["suggested_regions"]:
            region_counter[suggestion["region"]] += 1

    lines = [
        "# Regional Signal Overlay",
        "",
        f"- Records with regional suggestions: {len(record_ids)}",
        f"- Records with high-confidence suggestions: {len(high_confidence_records)}",
        f"- Suggested region labels: {sum(region_counter.values())}",
        f"- Suggested regions: {len(region_counter)}",
        "",
        "## Highest Priority Acquisition Backlog",
        "",
        "| Region | Priority | Suggestions | High Confidence | Sources | Examples |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for item in backlog[:25]:
        lines.append(
            "| {region} | {priority} | {total} | {high} | {sources} | {examples} |".format(
                region=item["region"],
                priority=item["priority"],
                total=item["suggested_record_count"],
                high=item["high_confidence_count"],
                sources=", ".join(item["sources"]) or "-",
                examples=", ".join(item["example_records"]) or "-",
            ),
        )

    lines.extend(["", "## Top Suggested Regions", ""])
    for region, count in region_counter.most_common(25):
        lines.append(f"- {region}: {count}")
    return "\n".join(lines).rstrip() + "\n"


def write_regional_signal_artifacts(
    input_paths: list[str | Path] | None = None,
    target_path: str | Path = DEFAULT_TARGETS,
    rules_path: str | Path = DEFAULT_RULES,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    *,
    min_confidence: float = 0.0,
    include_verified_records: bool = False,
) -> dict[str, Path]:
    """Write regional suggestion JSONL, acquisition backlog, and report."""

    records = load_dataset_records(list(input_paths) if input_paths is not None else list(DEFAULT_INPUTS))
    targets = load_region_targets(target_path)
    rules = load_signal_rules(rules_path)
    overlay = build_regional_signal_overlay(
        records,
        targets,
        rules,
        min_confidence=min_confidence,
        include_verified_records=include_verified_records,
    )
    backlog = build_acquisition_backlog(overlay, targets)

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    overlay_path = output / "regional_signal_overlay.jsonl"
    backlog_path = output / "regional_acquisition_backlog.json"
    report_path = output / "regional_signal_report.md"
    with overlay_path.open("w", encoding="utf-8") as handle:
        for item in overlay:
            handle.write(json.dumps(item, sort_keys=True))
            handle.write("\n")
    backlog_path.write_text(json.dumps(backlog, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(render_signal_report(overlay, backlog), encoding="utf-8")
    return {"overlay": overlay_path, "backlog": backlog_path, "report": report_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build regional signal overlay.")
    parser.add_argument("--input", action="append", dest="inputs")
    parser.add_argument("--targets", default=str(DEFAULT_TARGETS))
    parser.add_argument("--rules", default=str(DEFAULT_RULES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--include-verified-records", action="store_true")
    args = parser.parse_args(argv)

    inputs = [Path(item) for item in args.inputs] if args.inputs else list(DEFAULT_INPUTS)
    outputs = write_regional_signal_artifacts(
        inputs,
        args.targets,
        args.rules,
        args.out_dir,
        min_confidence=args.min_confidence,
        include_verified_records=args.include_verified_records,
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
