"""CRCNS (Collaborative Research in Computational Neuroscience) ingestion adapter.

CRCNS hosts ~270 curated electrophysiology datasets organized by brain area.
No formal API; we scrape the category listing and individual dataset pages.

Site: https://crcns.org/data-sets
"""
from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.registry import register

logger = logging.getLogger(__name__)

CRCNS_BASE = "https://crcns.org"
CRCNS_LISTING = f"{CRCNS_BASE}/data-sets"

# Map CRCNS category slugs (as they appear on the site) to canonical ontology IDs.
# Modalities use the same IDs as the rest of the corpus (extracellular_ephys, not
# extracellular_electrophysiology).
_CATEGORY_META: dict[str, dict[str, Any]] = {
    "vc":           {"brain_regions": ["visual_cortex"],           "modalities": ["extracellular_ephys"]},
    "ac":           {"brain_regions": ["auditory_cortex"],          "modalities": ["extracellular_ephys"]},
    "ear":          {"brain_regions": ["auditory_cortex", "brainstem"], "modalities": ["extracellular_ephys"]},
    "fcx":          {"brain_regions": ["prefrontal_cortex"],        "modalities": ["extracellular_ephys"]},
    "pfc":          {"brain_regions": ["prefrontal_cortex"],        "modalities": ["extracellular_ephys"]},
    "pc":           {"brain_regions": ["piriform_cortex"],          "modalities": ["extracellular_ephys"]},
    "motor-cortex": {"brain_regions": ["motor_cortex"],             "modalities": ["extracellular_ephys"]},
    "ssc":          {"brain_regions": ["somatosensory_cortex"],     "modalities": ["extracellular_ephys"]},
    "hc":           {"brain_regions": ["hippocampus"],              "modalities": ["extracellular_ephys"]},
    "thalamus":     {"brain_regions": ["thalamus"],                 "modalities": ["extracellular_ephys"]},
    "ofc":          {"brain_regions": ["OFC"],                      "modalities": ["extracellular_ephys"]},
    "lgn":          {"brain_regions": ["lateral_geniculate_nucleus"], "modalities": ["extracellular_ephys"]},
    "bf":           {"brain_regions": ["basal_forebrain"],          "modalities": ["extracellular_ephys"]},
    "bst":          {"brain_regions": ["brainstem"],                "modalities": ["extracellular_ephys"]},
    "cb":           {"brain_regions": ["cerebellum"],               "modalities": ["extracellular_ephys"]},
    "sp":           {"brain_regions": ["spinal_cord"],              "modalities": ["extracellular_ephys"]},
    "retina":       {"brain_regions": ["retina"],                   "modalities": ["extracellular_ephys"]},
    # ia = inferior areas (inferotemporal cortex)
    "ia":           {"brain_regions": ["temporal_cortex"],          "modalities": ["extracellular_ephys"]},
    # eye tracking / oculomotor datasets
    "eye":          {"brain_regions": [],                           "modalities": ["eye_tracking"]},
    # No specific region
    "aa":           {"brain_regions": [],                           "modalities": ["extracellular_ephys"]},
    "aplysia":      {"brain_regions": [],                           "modalities": ["extracellular_ephys"]},
    "tunicates":    {"brain_regions": [],                           "modalities": ["extracellular_ephys"]},
    "movements":    {"brain_regions": [],                           "modalities": ["behavioral"]},
    "methods":      {"brain_regions": [],                           "modalities": ["extracellular_ephys"]},
    "tools":        {"brain_regions": [],                           "modalities": ["extracellular_ephys"]},
    "sim":          {"brain_regions": [],                           "modalities": []},
    "challenges":   {"brain_regions": [],                           "modalities": ["extracellular_ephys"]},
    "other":        {"brain_regions": [],                           "modalities": ["extracellular_ephys"]},
}

_DEFAULT_CAT_META: dict[str, Any] = {"brain_regions": [], "modalities": ["extracellular_ephys"]}

_SPECIES_WORDS = {
    "mouse", "mice", "rat", "rats", "human", "humans", "monkey", "monkeys",
    "macaque", "cat", "cats", "ferret", "ferrets", "primate", "primates",
    "zebrafish", "marmoset", "aplysia",
}
_SPECIES_MAP = {
    "mice": "mouse", "rats": "rat", "humans": "human", "cats": "cat",
    "ferrets": "ferret", "monkeys": "monkey", "primates": "primate",
    "macaque": "macaque",
}
_FORMAT_PATTERNS = {
    r"\bnwb\b": "NWB",
    r"\bmat\b|\bmatlab\b": "MATLAB",
    r"\bhdf5?\b": "HDF5",
    r"\bplexon\b|\bplx\b": "Plexon",
    r"\bspike2\b": "Spike2",
    r"\bnex\b": "NEX",
    r"\bcsv\b": "CSV",
}

_SKIP_PHRASES = frozenset({"Plone", "Skip to", "navigation", "CRCNS.org", "Creative Commons"})


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s).strip()


def _extract_description(html: str) -> str:
    """Extract the main description from a CRCNS dataset page.

    Prefers the .documentDescription element (Plone CMS field), falls back to
    the first substantive paragraph.
    """
    # Primary: Plone documentDescription field
    m = re.search(r'class="documentDescription"[^>]*>(.*?)</\w+>', html, re.DOTALL)
    if m:
        text = re.sub(r"\s+", " ", _strip_html(m.group(1))).strip()
        if len(text) > 30:
            return text

    # Fallback: first substantive paragraph
    paras = re.findall(r"<p[^>]*>(.*?)</p>", html, re.DOTALL)
    for para in paras:
        cleaned = re.sub(r"\s+", " ", _strip_html(para)).strip()
        if len(cleaned) > 60 and not any(p in cleaned for p in _SKIP_PHRASES):
            return cleaned
    return ""


def _extract_species(text: str) -> list[str]:
    words_found = {w.lower() for w in re.findall(r"\b\w+\b", text.lower()) if w in _SPECIES_WORDS}
    mapped = {_SPECIES_MAP.get(w, w) for w in words_found}
    return sorted(mapped)


def _extract_formats(text: str) -> list[str]:
    lower = text.lower()
    return [fmt for pattern, fmt in _FORMAT_PATTERNS.items() if re.search(pattern, lower)]


def _get_category_datasets(client: httpx.Client, category: str) -> list[str]:
    """Return all dataset page URLs for one CRCNS category."""
    url = f"{CRCNS_BASE}/data-sets/{category}"
    try:
        resp = client.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("CRCNS category %s failed: %s", category, exc)
        return []

    links: set[str] = set()
    # Absolute links
    for link in re.findall(
        rf'href="({re.escape(CRCNS_BASE)}/data-sets/{re.escape(category)}/[^/"]+)"', resp.text
    ):
        links.add(link)
    # Relative links
    for rel in re.findall(rf'href="(/data-sets/{re.escape(category)}/[^/"]+)"', resp.text):
        links.add(f"{CRCNS_BASE}{rel}")

    category_root = f"{CRCNS_BASE}/data-sets/{category}/"
    return sorted(
        lnk for lnk in links
        if lnk != category_root and "/sitemap" not in lnk and lnk.count("/") >= 5
    )


def _scrape_dataset(client: httpx.Client, url: str, category: str) -> dict[str, Any] | None:
    """Scrape one CRCNS dataset detail page and return a corpus record."""
    slug = url.rstrip("/").split("/")[-1]
    try:
        resp = client.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("CRCNS dataset %s failed: %s", url, exc)
        return None

    page_html = resp.text
    description = _extract_description(page_html)

    # Title: "<slug>: <first sentence of description>"
    if description:
        first = description.split(".")[0].strip()
        title = f"{slug}: {first}" if first else slug
    else:
        title = slug

    species = _extract_species(description)
    # Scan first 4KB of page HTML for format hints (MATLAB, NWB, etc.)
    formats = _extract_formats(description + " " + page_html[:4000])

    cat_meta = _CATEGORY_META.get(category, _DEFAULT_CAT_META)
    base_regions: list[str] = cat_meta["brain_regions"]
    base_modalities: list[str] = cat_meta["modalities"]

    extraction = extract_dataset_labels(
        title=title,
        description=description,
        file_paths=[],
        source_metadata={"formats": formats, "category": category},
        linked_paper_abstracts=[],
    )

    # Merge category defaults with extraction results; no modality renaming —
    # corpus uses extracellular_ephys throughout.
    modalities = sorted({*base_modalities, *(item.id for item in extraction.modalities)} - {""})
    brain_regions = sorted({*base_regions, *(item.id for item in extraction.brain_regions)} - {""})
    if not species:
        species = [item.id for item in extraction.species]

    data_standards = sorted({*formats, *(item.id for item in extraction.data_standards)} - {""})

    return {
        "source": "crcns",
        "source_id": slug,
        "title": title,
        "description": description,
        "url": url,
        "license": "CC-BY 4.0",
        "species": species,
        "modalities": modalities,
        "brain_regions": brain_regions,
        "tasks": [item.id for item in extraction.tasks],
        "behaviors": [item.id for item in extraction.behaviors],
        "data_standards": data_standards,
        "has_behavior": bool(extraction.behaviors),
        "has_trials": any(t in description.casefold() for t in ["trial", "epoch", "event", "stimulus"]),
        "has_raw_data": True,
        "has_processed_data": "processed" in description.casefold() or "sorted" in description.casefold(),
        "metadata_json": {
            "raw_source": "crcns",
            "category": category,
            "formats": formats,
        },
    }


def _collect_all_categories(client: httpx.Client) -> list[str]:
    """Return all CRCNS category slugs from the main data-sets listing."""
    try:
        resp = client.get(CRCNS_LISTING + "/", timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("CRCNS main listing failed: %s", exc)
        return list(_CATEGORY_META.keys())

    links = re.findall(
        r'href="(?:https://crcns\.org)?/data-sets/([a-z][a-z0-9\-]*)(?:/|")', resp.text
    )
    seen: set[str] = set()
    return [
        cat for cat in links
        if cat and cat not in ("sitemap",) and cat not in seen and not seen.add(cat)  # type: ignore[func-returns-value]
    ] or list(_CATEGORY_META.keys())


@register("crcns")
def fetch_crcns_records(limit: int = 300) -> list[dict[str, Any]]:
    """Scrape all CRCNS brain-area categories and return normalized dataset records."""
    records: list[dict[str, Any]] = []
    seen_slugs: set[str] = set()

    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        categories = _collect_all_categories(client)
        logger.info("CRCNS: found %d categories", len(categories))

        for category in categories:
            if len(records) >= limit:
                break
            dataset_urls = _get_category_datasets(client, category)
            logger.debug("CRCNS %s: %d datasets", category, len(dataset_urls))
            for url in dataset_urls:
                if len(records) >= limit:
                    break
                slug = url.rstrip("/").split("/")[-1]
                if slug in seen_slugs:
                    continue
                seen_slugs.add(slug)
                rec = _scrape_dataset(client, url, category)
                if rec:
                    records.append(rec)

    logger.info("CRCNS: fetched %d datasets", len(records))
    return records
