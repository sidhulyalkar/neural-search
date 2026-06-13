"""CRCNS (Collaborative Research in Computational Neuroscience) ingestion adapter.

CRCNS hosts ~270 curated electrophysiology datasets organized by brain area.
No formal API; we scrape the category listing and individual dataset pages.

Site: https://crcns.org/data-sets
"""
from __future__ import annotations

import html
import logging
import re
from typing import Any

import httpx

from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.registry import register

logger = logging.getLogger(__name__)

CRCNS_BASE = "https://crcns.org"
CRCNS_LISTING = f"{CRCNS_BASE}/data-sets"

# Map CRCNS category slugs to brain region + default modality
_CATEGORY_META: dict[str, dict[str, Any]] = {
    "vc": {"brain_regions": ["visual_cortex"], "modalities": ["extracellular_electrophysiology"]},
    "ac": {"brain_regions": ["auditory_cortex"], "modalities": ["extracellular_electrophysiology"]},
    "ear": {"brain_regions": ["auditory_cortex", "brainstem"], "modalities": ["extracellular_electrophysiology"]},
    "fcx": {"brain_regions": ["frontal_cortex"], "modalities": ["extracellular_electrophysiology"]},
    "pfc": {"brain_regions": ["prefrontal_cortex"], "modalities": ["extracellular_electrophysiology"]},
    "pc": {"brain_regions": ["piriform_cortex"], "modalities": ["extracellular_electrophysiology"]},
    "motor-cortex": {"brain_regions": ["motor_cortex"], "modalities": ["extracellular_electrophysiology"]},
    "ssc": {"brain_regions": ["somatosensory_cortex"], "modalities": ["extracellular_electrophysiology"]},
    "hc": {"brain_regions": ["hippocampus"], "modalities": ["extracellular_electrophysiology"]},
    "th": {"brain_regions": ["thalamus"], "modalities": ["extracellular_electrophysiology"]},
    "ofc": {"brain_regions": ["orbitofrontal_cortex"], "modalities": ["extracellular_electrophysiology"]},
    "lb": {"brain_regions": ["lateral_geniculate_nucleus"], "modalities": ["extracellular_electrophysiology"]},
    "bf": {"brain_regions": ["basal_forebrain"], "modalities": ["extracellular_electrophysiology"]},
    "bst": {"brain_regions": ["brainstem"], "modalities": ["extracellular_electrophysiology"]},
    "cb": {"brain_regions": ["cerebellum"], "modalities": ["extracellular_electrophysiology"]},
    "sp": {"brain_regions": ["spinal_cord"], "modalities": ["extracellular_electrophysiology"]},
    "retina": {"brain_regions": ["retina"], "modalities": ["extracellular_electrophysiology"]},
    "eye": {"brain_regions": ["eye"], "modalities": ["eye_tracking"]},
    "aa": {"brain_regions": [], "modalities": ["extracellular_electrophysiology"]},
    "aplysia": {"brain_regions": [], "modalities": ["extracellular_electrophysiology"]},
    "crcns": {"brain_regions": [], "modalities": ["extracellular_electrophysiology"]},
    "challenges": {"brain_regions": [], "modalities": ["extracellular_electrophysiology"]},
    "other": {"brain_regions": [], "modalities": ["extracellular_electrophysiology"]},
}

_SPECIES_WORDS = {
    "mouse", "mice", "rat", "rats", "human", "humans", "monkey", "monkeys",
    "macaque", "cat", "cats", "ferret", "ferrets", "primate", "primates",
    "zebrafish", "marmoset", "aplysia",
}
_SPECIES_MAP = {
    "mice": "mouse", "rats": "rat", "humans": "human", "cats": "cat",
    "ferrets": "ferret", "monkeys": "monkey", "primates": "primate",
    "macaque": "monkey",
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


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s).strip()


def _extract_description(html: str) -> str:
    """Extract the main description from a CRCNS dataset page."""
    paras = re.findall(r"<p[^>]*>(.*?)</p>", html, re.DOTALL)
    for para in paras:
        cleaned = _strip_html(para).strip()
        # Skip navigation and footer blurbs
        if len(cleaned) > 40 and "Plone" not in cleaned and "Skip to" not in cleaned:
            return cleaned
    return ""


def _extract_species(text: str) -> list[str]:
    words_found = {w.lower() for w in re.findall(r"\b\w+\b", text.lower()) if w in _SPECIES_WORDS}
    mapped = {_SPECIES_MAP.get(w, w) for w in words_found}
    return sorted(mapped)


def _extract_formats(text: str) -> list[str]:
    formats = []
    lower = text.lower()
    for pattern, fmt in _FORMAT_PATTERNS.items():
        if re.search(pattern, lower):
            formats.append(fmt)
    return formats


def _get_category_datasets(client: httpx.Client, category: str) -> list[str]:
    """Return dataset slugs for one CRCNS category."""
    url = f"{CRCNS_BASE}/data-sets/{category}"
    try:
        resp = client.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("CRCNS category %s failed: %s", category, exc)
        return []
    # Dataset links: /data-sets/{category}/{dataset-slug}
    pattern = rf'href="({re.escape(CRCNS_BASE)}/data-sets/{re.escape(category)}/[^/\"]+)"'
    links = set(re.findall(pattern, resp.text))
    # Also relative links
    rel_pattern = rf'href="(/data-sets/{re.escape(category)}/[^/\"]+)"'
    rel_links = re.findall(rel_pattern, resp.text)
    links.update(f"{CRCNS_BASE}{l}" for l in rel_links)
    # Filter out category root and non-dataset pages
    return [
        l for l in sorted(links)
        if l != f"{CRCNS_BASE}/data-sets/{category}/"
        and "/sitemap" not in l
        and l.count("/") >= 5  # has dataset slug after category
    ]


def _scrape_dataset(client: httpx.Client, url: str, category: str) -> dict[str, Any] | None:
    """Scrape one CRCNS dataset detail page and return a flat corpus record."""
    slug = url.rstrip("/").split("/")[-1]
    try:
        resp = client.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("CRCNS dataset %s failed: %s", url, exc)
        return None

    page_html = resp.text
    description = _extract_description(page_html)
    # Build a meaningful title: first sentence of description, prefixed with slug
    if description:
        first_sentence = description.split(".")[0].strip()
        title = f"{slug}: {first_sentence}" if first_sentence else slug
    else:
        title = slug
    # Extract species/formats from description only to avoid navigation pollution
    species = _extract_species(description)
    formats = _extract_formats(description + " " + page_html[:2000])

    cat_meta = _CATEGORY_META.get(category, {"brain_regions": [], "modalities": ["extracellular_electrophysiology"]})
    base_regions = cat_meta["brain_regions"]
    base_modalities = cat_meta["modalities"]

    extraction = extract_dataset_labels(
        title=title,
        description=description,
        file_paths=[],
        source_metadata={"formats": formats, "brain_region": category},
        linked_paper_abstracts=[],
    )

    # extracellular_ephys is a synonym — normalize to extracellular_electrophysiology
    raw_mods = {m.replace("extracellular_ephys", "extracellular_electrophysiology")
                for m in [*base_modalities, *(item.id for item in extraction.modalities)]}
    modalities = sorted(raw_mods)
    brain_regions = sorted({*base_regions, *(item.id for item in extraction.brain_regions)})
    if not species:
        species = [item.id for item in extraction.species]

    data_standards = sorted({*formats, *(item.id for item in extraction.data_standards)})

    return {
        "source": "crcns",
        "source_id": slug,
        "source_type": "canonical_dataset",
        "identifier": slug,
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
        "analysis_affordances": ["spike_analysis", "single_unit_recording", "population_coding"],
        "metadata_json": {
            "raw_source": "crcns",
            "category": category,
            "formats": formats,
        },
    }


def _collect_all_categories(client: httpx.Client) -> list[str]:
    """Return all CRCNS category slugs from the main data-sets page."""
    try:
        resp = client.get(CRCNS_LISTING + "/", timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("CRCNS main listing failed: %s", exc)
        return list(_CATEGORY_META.keys())
    # Category links: /data-sets/{category}
    links = re.findall(r'href="(?:https://crcns\.org)?/data-sets/([a-z][a-z0-9\-]*)(?:/|\")', resp.text)
    seen: set[str] = set()
    categories: list[str] = []
    for cat in links:
        if cat and cat not in seen and cat not in ("sitemap",):
            seen.add(cat)
            categories.append(cat)
    return categories or list(_CATEGORY_META.keys())


@register("crcns")
def fetch_crcns_records(limit: int = 300) -> list[dict[str, Any]]:
    """Scrape all CRCNS brain-area categories and return dataset records."""
    records: list[dict[str, Any]] = []
    seen_slugs: set[str] = set()

    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        categories = _collect_all_categories(client)
        logger.info("CRCNS: found %d categories", len(categories))

        for category in categories:
            if len(records) >= limit:
                break
            dataset_urls = _get_category_datasets(client, category)
            logger.debug("CRCNS category %s: %d datasets", category, len(dataset_urls))
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

    logger.info("CRCNS: accepted %d datasets", len(records))
    return records
