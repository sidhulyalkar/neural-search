"""LLM-based brain region inference for corpus records without regions.

All inferred labels are silver-tier with provenance tracking — NOT human-adjudicated gold.

Provider priority (first available wins):
  1. Ollama      — local, free; set OLLAMA_BASE_URL + OLLAMA_MODEL
  2. LM Studio   — local, free; set LM_STUDIO_BASE_URL + LM_STUDIO_MODEL
  3. Anthropic   — Claude Haiku 4.5; set ANTHROPIC_API_KEY
  4. Gemini      — Flash variants; set GEMINI_API_KEY (+ KEY1..KEY6 for rotation)
  5. DeepSeek    — set DEEPSEEK_API_KEY
  6. OpenRouter  — set OPENROUTER_API_KEY

Parallel workers (7 Gemini keys → ~7x throughput):
  for i in $(seq 0 6); do
    GEMINI_KEY_IDX=$i python scripts/corpus/infer_regions_llm.py --dry-run --stride 7 --offset $i &
  done
  wait
  python scripts/corpus/infer_regions_llm.py  # apply cache to corpus

Local model (Ollama on M1 Max):
  ollama pull llama3.1        # or qwen2.5, mistral-nemo, etc.
  OLLAMA_MODEL=llama3.1 python scripts/corpus/infer_regions_llm.py

Usage:
  python scripts/corpus/infer_regions_llm.py [--dry-run] [--limit N]
  python scripts/corpus/infer_regions_llm.py --stride 7 --offset 0 --dry-run
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

_PROVENANCE_ANTHROPIC  = "claude_haiku_inferred_silver_not_human_gold"
_PROVENANCE_GEMINI     = "gemini_flash_inferred_silver_not_human_gold"
_PROVENANCE_DEEPSEEK   = "deepseek_inferred_silver_not_human_gold"
_PROVENANCE_OPENROUTER = "openrouter_inferred_silver_not_human_gold"
_PROVENANCE_LOCAL      = "local_llm_inferred_silver_not_human_gold"

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


# ─── cache helpers ────────────────────────────────────────────────────────────

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


# ─── env helpers ──────────────────────────────────────────────────────────────

def _load_api_key(env_var: str) -> str | None:
    key = os.environ.get(env_var)
    if key:
        return key
    for env_path in [Path(".env.local"), Path(".env")]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith(f"{env_var}="):
                    return line.split("=", 1)[1].strip()
    return None


def _load_env_var(env_var: str) -> str | None:
    return _load_api_key(env_var)


# ─── Anthropic provider ───────────────────────────────────────────────────────

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


# ─── Gemini provider (multi-key rotation) ────────────────────────────────────

_GEMINI_MODELS = [
    "models/gemini-2.5-flash-lite",
    "models/gemini-flash-lite-latest",
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
]

_QUOTA_EXHAUSTED = object()
_gemini_key_idx = 0
_gemini_model_idx = 0


def _load_gemini_keys() -> list[str]:
    key_envs = ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY{i}" for i in range(1, 10)]
    keys = [k for env in key_envs if (k := _load_api_key(env))]
    return keys


def _make_gemini_client(api_key: str):
    from google import genai as _genai
    return _genai.Client(api_key=api_key)


def _build_gemini_caller(keys: list[str]):
    """Returns a callable that rotates through all key×model combos before sleeping."""
    clients = [_make_gemini_client(k) for k in keys]
    state = {"key_idx": 0, "model_idx": 0}

    def call(title: str, description: str) -> list[str] | object:
        prompt = f"Title: {title}\nDescription: {description[:400] if description else '(none)'}"
        total_slots = len(keys) * len(_GEMINI_MODELS)
        for _attempt in range(total_slots):
            client = clients[state["key_idx"] % len(clients)]
            model = _GEMINI_MODELS[state["model_idx"] % len(_GEMINI_MODELS)]
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[{"role": "user", "parts": [{"text": _SYSTEM_PROMPT + "\n\n" + prompt}]}],
                )
                return _parse_llm_response(response.text)
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    logger.warning("Quota exhausted on key[%d]/%s, rotating",
                                   state["key_idx"] % len(clients), model)
                    # Advance model; when we've cycled all models on this key, advance key
                    state["model_idx"] += 1
                    if state["model_idx"] % len(_GEMINI_MODELS) == 0:
                        state["key_idx"] += 1
                    continue
                logger.warning("Gemini call failed: %s", e)
                return []
        logger.warning("All %d Gemini key×model slots exhausted — sleeping 65s", total_slots)
        time.sleep(65)
        # Reset to first key/model after backoff
        state["key_idx"] = 0
        state["model_idx"] = 0
        return _QUOTA_EXHAUSTED

    return call


# ─── OpenAI-compatible provider (Ollama / LM Studio / DeepSeek / OpenRouter) ──

def _build_openai_compat_caller(base_url: str, api_key: str, model: str, provenance_label: str):
    """Builds a caller for any OpenAI-compatible endpoint."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai SDK not installed; pip install openai")
        return None, None

    client = OpenAI(base_url=base_url, api_key=api_key)

    def call(title: str, description: str) -> list[str]:
        prompt = f"Title: {title}\nDescription: {description[:400] if description else '(none)'}"
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=256,
                temperature=0.0,
            )
            return _parse_llm_response(resp.choices[0].message.content or "")
        except Exception as e:
            logger.warning("%s call failed: %s", provenance_label, e)
            return []

    return call, provenance_label


# ─── main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Build cache but skip corpus write")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", default="all",
                        help="Filter by source name, or 'all'")
    parser.add_argument("--offset", type=int, default=0,
                        help="Parallel worker: start index (default 0)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Parallel worker: step size (default 1 = single worker)")
    parser.add_argument("--provider",
                        choices=["auto", "ollama", "lmstudio", "anthropic",
                                 "gemini", "deepseek", "openrouter"],
                        default="auto",
                        help="Force a specific provider (default: auto-detect)")
    args = parser.parse_args(argv)

    # ── provider selection ──────────────────────────────────────────────────
    call_llm = None
    provenance = None
    force = args.provider  # "auto" means try in priority order

    skip_local = _load_env_var("SKIP_LOCAL_PROVIDERS") == "1"

    # 1. Ollama (local) — only if reachable
    ollama_url = _load_env_var("OLLAMA_BASE_URL")
    ollama_model = _load_env_var("OLLAMA_MODEL") or "llama3.1"
    if ollama_url and call_llm is None and not skip_local and force in ("auto", "ollama"):
        try:
            import httpx as _httpx
            _httpx.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=2.0).raise_for_status()
            fn, prov = _build_openai_compat_caller(
                f"{ollama_url.rstrip('/')}/v1", "ollama", ollama_model, _PROVENANCE_LOCAL)
            if fn:
                call_llm, provenance = fn, prov
                logger.info("Provider: Ollama (%s @ %s)", ollama_model, ollama_url)
        except Exception:
            logger.debug("Ollama not reachable at %s — skipping", ollama_url)

    # 2. LM Studio (local) — only if reachable
    lms_url = _load_env_var("LM_STUDIO_BASE_URL")
    lms_model = _load_env_var("LM_STUDIO_MODEL") or "local-model"
    if lms_url and call_llm is None and not skip_local and force in ("auto", "lmstudio"):
        try:
            import httpx as _httpx
            _httpx.get(f"{lms_url.rstrip('/')}/v1/models", timeout=2.0).raise_for_status()
            fn, prov = _build_openai_compat_caller(
                f"{lms_url.rstrip('/')}/v1", "lm-studio", lms_model, _PROVENANCE_LOCAL)
            if fn:
                call_llm, provenance = fn, prov
                logger.info("Provider: LM Studio (%s @ %s)", lms_model, lms_url)
        except Exception:
            logger.debug("LM Studio not reachable at %s — skipping", lms_url)

    # 3. Anthropic
    anthropic_key = _load_api_key("ANTHROPIC_API_KEY")
    if anthropic_key and call_llm is None and force in ("auto", "anthropic"):
        try:
            import anthropic as _anthropic
            base_url = _load_env_var("ANTHROPIC_BASE_URL")
            client = _anthropic.Anthropic(
                api_key=anthropic_key,
                **({"base_url": base_url} if base_url else {}),
            )
            def anthropic_call(t: str, d: str) -> list[str]:
                return _call_anthropic(client, t, d)
            call_llm = anthropic_call
            provenance = _PROVENANCE_ANTHROPIC
            logger.info("Provider: Anthropic Claude Haiku 4.5")
        except ImportError:
            logger.warning("anthropic SDK not installed; pip install anthropic")

    # 4. Gemini (multi-key rotation)
    gemini_keys = _load_gemini_keys()
    # Allow GEMINI_KEY_IDX env to pin a single key for a parallel worker
    key_idx_pin = os.environ.get("GEMINI_KEY_IDX")
    if key_idx_pin is not None and gemini_keys:
        idx = int(key_idx_pin) % len(gemini_keys)
        gemini_keys = [gemini_keys[idx]]
        logger.info("Gemini key pinned to index %d (parallel worker mode)", idx)
    if gemini_keys and call_llm is None and force in ("auto", "gemini"):
        try:
            caller = _build_gemini_caller(gemini_keys)
            call_llm = caller
            provenance = _PROVENANCE_GEMINI
            logger.info("Provider: Gemini Flash (%d key(s) × %d models)",
                        len(gemini_keys), len(_GEMINI_MODELS))
        except ImportError:
            logger.warning("google-genai not installed; pip install google-genai")

    # 5. DeepSeek
    deepseek_key = _load_api_key("DEEPSEEK_API_KEY")
    deepseek_url = _load_env_var("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    if deepseek_key and call_llm is None and force in ("auto", "deepseek"):
        fn, prov = _build_openai_compat_caller(
            deepseek_url, deepseek_key, "deepseek-chat", _PROVENANCE_DEEPSEEK)
        if fn:
            call_llm, provenance = fn, prov
            logger.info("Provider: DeepSeek")

    # 6. OpenRouter
    openrouter_key = _load_api_key("OPENROUTER_API_KEY")
    openrouter_url = _load_env_var("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    if openrouter_key and call_llm is None and force in ("auto", "openrouter"):
        fn, prov = _build_openai_compat_caller(
            openrouter_url, openrouter_key,
            "google/gemini-flash-1.5", _PROVENANCE_OPENROUTER)
        if fn:
            call_llm, provenance = fn, prov
            logger.info("Provider: OpenRouter")

    if call_llm is None:
        logger.error(
            "No LLM provider found. Set one of:\n"
            "  OLLAMA_BASE_URL (+ optionally OLLAMA_MODEL)\n"
            "  ANTHROPIC_API_KEY\n"
            "  GEMINI_API_KEY [+ GEMINI_API_KEY1..6 for key rotation]\n"
            "  DEEPSEEK_API_KEY\n"
            "  OPENROUTER_API_KEY"
        )
        return 1

    # ── corpus + targets ────────────────────────────────────────────────────
    corpus = [
        json.loads(line)
        for line in CORPUS_PATH.read_text().strip().splitlines()
        if line.strip()
    ]
    llm_cache = _load_cache()

    targets = [
        r for r in corpus
        if not _has_region(r)
        and not _is_test_dataset(r)
        and (args.source == "all" or r.get("source") == args.source)
    ]
    logger.info("Records without regions (excl. test datasets): %d", len(targets))

    # Parallel worker slice
    if args.stride > 1:
        targets = targets[args.offset::args.stride]
        logger.info("Worker offset=%d stride=%d → %d records", args.offset, args.stride, len(targets))

    if args.limit:
        targets = targets[:args.limit]

    # ── inference loop ──────────────────────────────────────────────────────
    enriched_map: dict[str, list[str]] = {}
    skipped = 0
    inferred = 0

    for i, rec in enumerate(targets, 1):
        sid = rec["source_id"]
        if sid in llm_cache:
            regions = llm_cache[sid]
            if regions:
                enriched_map[sid] = regions
            skipped += 1
            continue

        title = rec.get("title") or ""
        description = rec.get("description") or ""
        if not title:
            _save_cache(sid, [])
            continue

        logger.info("[%d/%d] Inferring regions for %s: %s", i, len(targets), sid, title[:60])
        regions = call_llm(title, description)

        if regions is _QUOTA_EXHAUSTED:
            regions = call_llm(title, description)
            if regions is _QUOTA_EXHAUSTED:
                logger.error("Persistent quota failure on %s — skipping", sid)
                continue

        if regions is not None:
            _save_cache(sid, regions)  # always cache, even in --dry-run

        if regions:
            logger.info("  → %s", regions)
            enriched_map[sid] = regions
            inferred += 1

        time.sleep(0.1)

    logger.info("Cache hits: %d, Newly inferred: %d, With regions: %d",
                skipped, inferred, len(enriched_map))

    if args.dry_run:
        logger.info("[dry-run] cache written; skipping corpus write")
        return 0

    # ── apply cache to corpus ───────────────────────────────────────────────
    # Reload cache to pick up any results from parallel workers
    full_cache = _load_cache()
    enriched_map.update({sid: r for sid, r in full_cache.items() if r})

    output = []
    updated = 0
    for rec in corpus:
        sid = rec["source_id"]
        new_regions = enriched_map.get(sid)
        if new_regions:
            existing = _flatten_ids(rec.get("brain_regions") or [])
            merged = list(dict.fromkeys(existing + new_regions))
            rec = {**rec, "brain_regions": merged, "brain_regions_provenance": provenance}
            updated += 1
        output.append(json.dumps(rec))

    CORPUS_PATH.write_text("\n".join(output) + "\n")
    logger.info("Updated %d records → %s", updated, CORPUS_PATH)

    refreshed = [json.loads(line) for line in CORPUS_PATH.read_text().strip().splitlines() if line.strip()]
    total = len(refreshed)
    with_region = sum(1 for r in refreshed if _has_region(r))
    logger.info("Brain region coverage: %d/%d = %d%%", with_region, total,
                round(100 * with_region / total))
    by_src: dict[str, list[int]] = {}
    for r in refreshed:
        s = r.get("source", "?")
        by_src.setdefault(s, [0, 0])
        by_src[s][1] += 1
        if _has_region(r):
            by_src[s][0] += 1
    for s, (w, t) in sorted(by_src.items(), key=lambda x: -x[1][1]):
        logger.info("  %-25s: %d/%d = %d%%", s, w, t, round(100 * w / t))
    return 0


if __name__ == "__main__":
    sys.exit(main())
