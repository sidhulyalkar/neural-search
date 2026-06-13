"""LLM-based brain region inference for corpus records without regions.

Infers brain regions from dataset titles and descriptions using an LLM.
All inferred labels are silver-tier with provenance tracking — they are NOT
human-adjudicated gold labels.

Provider priority (first available key wins):
  1. ANTHROPIC_API_KEY → Claude Haiku 4.5 (fastest, cheapest)
  2. GEMINI_API_KEY    → Gemini 2.0 Flash

Usage:
    python scripts/corpus/infer_regions_llm.py [--dry-run] [--limit N]
    ANTHROPIC_API_KEY=sk-ant-... python scripts/corpus/infer_regions_llm.py
    GEMINI_API_KEY=...           python scripts/corpus/infer_regions_llm.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("llm_region_infer")

CORPUS_PATH = Path("data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl")
LLM_CACHE_PATH = Path("data/raw/dandi/llm_region_cache.jsonl")

# Provenance tags — MUST NOT be "human_gold" or any variant
_PROVENANCE_ANTHROPIC = "claude_haiku_inferred_silver_not_human_gold"
_PROVENANCE_GEMINI = "gemini_flash_inferred_silver_not_human_gold"

# Region IDs the model is allowed to return (subset of our ontology)
VALID_REGION_IDS = {
    "visual_cortex", "v1", "v2", "v4", "area_mt", "mst",
    "prefrontal_cortex", "mPFC", "OFC", "ACC", "dlPFC", "dmFC",
    "premotor_cortex", "PMd", "motor_cortex",
    "striatum", "dorsal_striatum", "ventral_striatum", "nucleus_accumbens",
    "caudate", "putamen",
    "somatosensory_cortex",
    "parietal_cortex", "posterior_parietal_cortex", "lip", "aip",
    "temporal_cortex", "auditory_cortex",
    "retrosplenial_cortex", "posterior_cingulate_cortex", "insula",
    "neocortex",
    "hippocampus", "ca1", "ca2", "ca3", "dentate_gyrus", "subiculum",
    "entorhinal_cortex", "medial_entorhinal_cortex", "lateral_entorhinal_cortex",
    "amygdala", "basolateral_amygdala", "central_amygdala",
    "thalamus", "lateral_geniculate_nucleus", "medial_geniculate_nucleus",
    "mediodorsal_thalamus", "pulvinar",
    "globus_pallidus", "subthalamic_nucleus",
    "substantia_nigra", "vta",
    "superior_colliculus", "inferior_colliculus", "periaqueductal_gray",
    "brainstem", "locus_coeruleus",
    "cerebellum",
    "hypothalamus",
    "septum",
    "olfactory_bulb", "piriform_cortex",
    "broca_area", "wernicke_area",
    "spinal_cord",
    "retina",
}

_SYSTEM_PROMPT = """You are a neuroscience expert helping to label datasets by brain region.

Given a dataset title and description, identify which specific brain regions are the PRIMARY focus of the recording or study. Return ONLY a JSON array of brain region IDs from the allowed list below. Return an empty array [] if no specific region can be confidently determined.

Rules:
- Only return regions that are clearly the focus (not just mentioned in passing)
- For whole-brain fMRI studies with no specific target, return []
- "Behavioral data" with no recordings → []
- Test/example/demo datasets → []
- Use the most specific region ID that applies (e.g., "ca1" not just "hippocampus" if CA1 is explicit)
- A dataset can have multiple regions

Allowed region IDs:
visual_cortex, v1, v2, v4, area_mt, mst,
prefrontal_cortex, mPFC, OFC, ACC, dlPFC, dmFC,
premotor_cortex, PMd, motor_cortex,
striatum, dorsal_striatum, ventral_striatum, nucleus_accumbens, caudate, putamen,
somatosensory_cortex,
parietal_cortex, posterior_parietal_cortex, lip, aip,
temporal_cortex, auditory_cortex,
retrosplenial_cortex, posterior_cingulate_cortex, insula,
neocortex,
hippocampus, ca1, ca2, ca3, dentate_gyrus, subiculum,
entorhinal_cortex, medial_entorhinal_cortex, lateral_entorhinal_cortex,
amygdala, basolateral_amygdala, central_amygdala,
thalamus, lateral_geniculate_nucleus, medial_geniculate_nucleus, mediodorsal_thalamus, pulvinar,
globus_pallidus, subthalamic_nucleus,
substantia_nigra, vta,
superior_colliculus, inferior_colliculus, periaqueductal_gray,
brainstem, locus_coeruleus,
cerebellum,
hypothalamus,
septum,
olfactory_bulb, piriform_cortex,
broca_area, wernicke_area,
spinal_cord,
retina

Return ONLY valid JSON array. No explanation, no markdown fences."""


def _load_cache() -> dict[str, list[str]]:
    cache: dict[str, list[str]] = {}
    if LLM_CACHE_PATH.exists():
        for line in LLM_CACHE_PATH.read_text().strip().splitlines():
            if line.strip():
                entry = json.loads(line)
                cache[entry["source_id"]] = entry["regions"]
    return cache


def _save_cache(source_id: str, regions: list[str]) -> None:
    with LLM_CACHE_PATH.open("a") as f:
        f.write(json.dumps({"source_id": source_id, "regions": regions}) + "\n")


def _flatten_ids(br: list) -> list[str]:
    return [(v.get("id") if isinstance(v, dict) else v) for v in (br or []) if v]


def _has_region(record: dict) -> bool:
    return any(_flatten_ids(record.get("brain_regions") or []))


def _is_test_dataset(record: dict) -> bool:
    title = (record.get("title") or "").lower()
    test_keywords = {"test", "example", "asdf", "demo", "development", "testing",
                     "template", "abc", "dummy", "placeholder"}
    return any(kw in title for kw in test_keywords) or len(title.strip()) < 5


def _parse_llm_response(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [r for r in parsed if r in VALID_REGION_IDS]


def _call_anthropic(client, title: str, description: str) -> list[str]:
    prompt = f"Title: {title}\nDescription: {description[:400] if description else '(none)'}"
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": _SYSTEM_PROMPT + "\n\n" + prompt}],
        )
        return _parse_llm_response(msg.content[0].text)
    except Exception as e:
        logger.warning("Anthropic call failed: %s", e)
        return []


_GEMINI_MODELS = [
    "models/gemini-2.5-flash-lite",
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
    "models/gemini-flash-lite-latest",
]


def _call_gemini(client, title: str, description: str, _model_state: list = [0]) -> list[str]:
    prompt = f"Title: {title}\nDescription: {description[:400] if description else '(none)'}"
    for attempt in range(len(_GEMINI_MODELS)):
        model = _GEMINI_MODELS[_model_state[0] % len(_GEMINI_MODELS)]
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    {"role": "user", "parts": [{"text": _SYSTEM_PROMPT + "\n\n" + prompt}]}
                ],
            )
            return _parse_llm_response(response.text)
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                logger.warning("Quota exhausted on %s, trying next model", model)
                _model_state[0] += 1
                continue
            logger.warning("Gemini call failed: %s", e)
            return []
    logger.error("All Gemini models quota-exhausted")
    return []


def _load_api_key(env_var: str) -> str | None:
    key = os.environ.get(env_var)
    if key:
        return key
    env_path = Path(".env.local")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith(f"{env_var}="):
                return line.split("=", 1)[1].strip()
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", choices=["dandi", "openneuro", "spark", "all"],
                        default="all")
    args = parser.parse_args(argv)

    # Determine provider (Anthropic preferred for quality + no quota issues)
    call_llm = None
    provenance = None

    anthropic_key = _load_api_key("ANTHROPIC_API_KEY")
    gemini_key = _load_api_key("GEMINI_API_KEY")

    if anthropic_key:
        try:
            import anthropic as _anthropic
            client = _anthropic.Anthropic(api_key=anthropic_key)
            call_llm = lambda t, d: _call_anthropic(client, t, d)
            provenance = _PROVENANCE_ANTHROPIC
            logger.info("Provider: Anthropic Claude Haiku")
        except ImportError:
            logger.warning("anthropic SDK not installed; pip install anthropic")

    if call_llm is None and gemini_key:
        try:
            from google import genai as _genai
            client = _genai.Client(api_key=gemini_key)
            call_llm = lambda t, d: _call_gemini(client, t, d)
            provenance = _PROVENANCE_GEMINI
            logger.info("Provider: Gemini Flash")
        except ImportError:
            logger.warning("google-genai not installed; pip install google-genai")

    if call_llm is None:
        logger.error(
            "No LLM API key found. Set ANTHROPIC_API_KEY or GEMINI_API_KEY.\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "  then re-run: python scripts/corpus/infer_regions_llm.py"
        )
        return 1

    corpus = [
        json.loads(l)
        for l in CORPUS_PATH.read_text().strip().splitlines()
        if l.strip()
    ]
    llm_cache = _load_cache()

    targets = [
        r for r in corpus
        if not _has_region(r)
        and not _is_test_dataset(r)
        and (args.source == "all" or r.get("source") == args.source)
    ]
    logger.info("Records without regions (excl. test datasets): %d", len(targets))

    if args.limit:
        targets = targets[: args.limit]

    enriched_map: dict[str, list[str]] = {}
    skipped = 0
    inferred = 0

    for i, rec in enumerate(targets, 1):
        sid = rec["source_id"]
        if sid in llm_cache:
            regions = llm_cache[sid]
            if regions:
                logger.debug("[%d/%d] Cache hit %s → %s", i, len(targets), sid, regions)
                enriched_map[sid] = regions
            skipped += 1
            continue

        title = rec.get("title") or ""
        description = rec.get("description") or ""
        if not title:
            _save_cache(sid, [])
            continue

        logger.info("[%d/%d] Inferring regions for %s: %s", i, len(targets), sid,
                    title[:60])
        regions = call_llm(title, description)

        if not args.dry_run:
            _save_cache(sid, regions)

        if regions:
            logger.info("  → %s", regions)
            enriched_map[sid] = regions
            inferred += 1

        time.sleep(0.1)  # rate limit courtesy

    logger.info("Cache hits: %d, Newly inferred: %d, With regions: %d",
                skipped, inferred, len(enriched_map))

    if args.dry_run:
        logger.info("[dry-run] skipping corpus write")
        return 0

    output = []
    updated = 0
    for rec in corpus:
        sid = rec["source_id"]
        new_regions = enriched_map.get(sid)
        if new_regions:
            existing = _flatten_ids(rec.get("brain_regions") or [])
            merged = list(dict.fromkeys(existing + new_regions))
            # Tag as silver provenance
            rec = {
                **rec,
                "brain_regions": merged,
                "brain_regions_provenance": provenance,
            }
            updated += 1
        output.append(json.dumps(rec))

    CORPUS_PATH.write_text("\n".join(output) + "\n")
    logger.info("Updated %d records → %s", updated, CORPUS_PATH)

    # Report coverage
    refreshed = [json.loads(l) for l in CORPUS_PATH.read_text().strip().splitlines() if l.strip()]
    total = len(refreshed)
    with_region = sum(1 for r in refreshed if _has_region(r))
    logger.info("Brain region coverage: %d/%d = %d%%", with_region, total,
                round(100 * with_region / total))
    by_src: dict[str, list[int]] = {}
    for r in refreshed:
        s = r["source"]
        by_src.setdefault(s, [0, 0])
        by_src[s][1] += 1
        if _has_region(r):
            by_src[s][0] += 1
    for s, (w, t) in sorted(by_src.items()):
        logger.info("  %s: %d/%d = %d%%", s, w, t, round(100 * w / t))
    return 0


if __name__ == "__main__":
    sys.exit(main())
