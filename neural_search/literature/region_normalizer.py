"""Allen CCF + UBERON region normalization for neuroscience finding records.

Fetches the Allen Mouse Brain Atlas structure tree on first call and caches
it to data/ontology/allen_ccf.json. Subsequent calls read the cache.

UBERON_BRIDGE provides cross-species canonical mappings for terms that appear
in human/primate findings with different terminology than the Allen CCF mouse
atlas (e.g., "medial temporal lobe" → "hippocampal formation").
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

ALLEN_API_URL = (
    "https://api.brain-map.org/api/v2/data/query.json"
    "?criteria=model::Structure,rma::criteria,[graph_id$eq1]"
    ",rma::options[num_rows$eqall]"
    "[only$eq'id,name,acronym,parent_structure_id']"
)
DEFAULT_CACHE = Path("data/ontology/allen_ccf.json")

# Cross-species canonical mappings (human/primate → Allen CCF mouse equivalent).
# Extend this dict as new species are encountered in findings.
UBERON_BRIDGE: dict[str, str] = {
    "medial temporal lobe": "hippocampal formation",
    "mtl": "hippocampal formation",
    "temporal lobe": "temporal lobe",
    "subiculum": "subiculum",
    "entorhinal cortex": "entorhinal area",
    "ec": "entorhinal area",
    "dlpfc": "prefrontal cortex",
    "dorsolateral prefrontal cortex": "prefrontal cortex",
    "vlpfc": "prefrontal cortex",
    "ofc": "orbital frontal cortex",
    "orbitofrontal cortex": "orbital frontal cortex",
    "acc": "anterior cingulate area",
    "anterior cingulate cortex": "anterior cingulate area",
    "v1": "primary visual cortex",
    "primary visual cortex": "primary visual cortex",
    "m1": "primary motor cortex",
    "primary motor cortex": "primary motor cortex",
    "s1": "primary somatosensory area",
    "primary somatosensory cortex": "primary somatosensory area",
    "basolateral amygdala": "basolateral amygdaloid nucleus",
    "bla": "basolateral amygdaloid nucleus",
    "central amygdala": "central amygdaloid nucleus",
    "cea": "central amygdaloid nucleus",
    "dorsal striatum": "caudoputamen",
    "caudate putamen": "caudoputamen",
    "nucleus accumbens": "nucleus accumbens",
    "nac": "nucleus accumbens",
    "vta": "ventral tegmental area",
    "ventral tegmental area": "ventral tegmental area",
    "substantia nigra": "substantia nigra",
    "snr": "substantia nigra, reticular part",
    "locus coeruleus": "locus ceruleus",
    "lc": "locus ceruleus",
    "raphe": "raphe nuclei",
}


def fetch_allen_ccf(cache_path: Path = DEFAULT_CACHE) -> dict[str, dict[str, Any]]:
    """Return Allen CCF structure map {str(id): {name, acronym, parent_id}}.

    Downloads from Allen API on first call; reads cache on subsequent calls.
    """
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    logger.info("Fetching Allen CCF from %s", ALLEN_API_URL)
    response = httpx.get(ALLEN_API_URL, timeout=60)
    response.raise_for_status()
    payload = response.json()

    structures: dict[str, dict[str, Any]] = {}
    for s in payload.get("msg", []):
        structures[str(s["id"])] = {
            "name": s["name"].lower().strip(),
            "acronym": s["acronym"].lower().strip(),
            "parent_id": s.get("parent_structure_id"),
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(structures, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Cached %d structures to %s", len(structures), cache_path)
    return structures


def build_name_index(structures: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Return {lowercase_name_or_acronym: structure_id} for fast lookup."""
    index: dict[str, int] = {}
    for sid, s in structures.items():
        index[s["name"]] = int(sid)
        index[s["acronym"]] = int(sid)
    return index


def normalize_region(
    region: str,
    name_index: dict[str, int],
    structures: dict[str, dict[str, Any]],
) -> str:
    """Return canonical CCF name for region, or UBERON bridge term, or original string.

    Lookup order: (1) Allen CCF exact match, (2) UBERON_BRIDGE cross-species map,
    (3) original string lowercased.
    """
    cleaned = region.lower().strip()
    if cleaned in name_index:
        sid = name_index[cleaned]
        return structures[str(sid)]["name"]
    if cleaned in UBERON_BRIDGE:
        return UBERON_BRIDGE[cleaned]
    return cleaned


def get_parent_chain(
    structure_id: int,
    structures: dict[str, dict[str, Any]],
) -> list[str]:
    """Return list of ancestor names from structure up to (and including) root."""
    chain: list[str] = []
    sid: int | None = structure_id
    visited: set[int] = set()
    while sid is not None and sid not in visited:
        s = structures.get(str(sid))
        if s is None:
            break
        chain.append(s["name"])
        visited.add(sid)
        sid = s["parent_id"]
    return chain


def normalize_finding(
    finding: dict[str, Any],
    name_index: dict[str, int],
    structures: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return finding dict with added `regions_normalized` list.

    Original fields are preserved unchanged. `regions_normalized` contains
    canonical Allen CCF names (or original strings for unrecognized regions).
    """
    normalized = [
        normalize_region(r, name_index, structures)
        for r in (finding.get("regions") or [])
    ]
    return {**finding, "regions_normalized": normalized}
