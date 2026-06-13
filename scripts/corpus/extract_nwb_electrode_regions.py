"""Extract brain region labels from NWB electrode tables via DANDI streaming.

For DANDI ephys records that still have no brain_regions after metadata enrichment,
this script samples one NWB asset per dandiset and reads the electrode table's
'location' column to extract brain region labels.

Usage:
    python scripts/corpus/extract_nwb_electrode_regions.py [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("nwb_region_extractor")

CORPUS_PATH = Path("data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl")
ELECTRODE_CACHE_PATH = Path("data/raw/dandi/nwb_electrode_locations.jsonl")

# Map NWB electrode location strings → canonical brain region IDs
# These appear in electrode_table.location from real NWB files
_NWB_LOCATION_MAP: dict[str, str] = {
    # Visual cortex
    "v1": "v1",
    "v1l": "v1",
    "v1r": "v1",
    "visp": "v1",
    "primary visual cortex": "v1",
    "v2": "v2",
    "visual cortex": "visual_cortex",
    "visual area": "visual_cortex",
    "mt": "area_mt",
    "area mt": "area_mt",
    # Motor cortex
    "m1": "motor_cortex",
    "motor cortex": "motor_cortex",
    "primary motor cortex": "motor_cortex",
    "m1l": "motor_cortex",
    "m1r": "motor_cortex",
    "mog": "motor_cortex",
    # Premotor
    "premotor cortex": "premotor_cortex",
    "sma": "premotor_cortex",
    "supplementary motor area": "premotor_cortex",
    "pmd": "PMd",
    "pmc": "premotor_cortex",
    # Prefrontal
    "pfc": "prefrontal_cortex",
    "prefrontal cortex": "prefrontal_cortex",
    "frontal cortex": "prefrontal_cortex",
    "mpfc": "mPFC",
    "dlpfc": "dlPFC",
    "ofc": "OFC",
    "acc": "ACC",
    "anterior cingulate": "ACC",
    # Somatosensory
    "s1": "somatosensory_cortex",
    "somatosensory cortex": "somatosensory_cortex",
    "primary somatosensory cortex": "somatosensory_cortex",
    "barrel cortex": "somatosensory_cortex",
    "s1bf": "somatosensory_cortex",
    "s2": "somatosensory_area_2",
    # Parietal
    "ppc": "posterior_parietal_cortex",
    "posterior parietal cortex": "posterior_parietal_cortex",
    "area 2": "somatosensory_area_2",
    # Hippocampal formation
    "ca1": "ca1",
    "ca2": "ca2",
    "ca3": "ca3",
    "dg": "dentate_gyrus",
    "dentate gyrus": "dentate_gyrus",
    "hippocampus": "hippocampus",
    "hip": "hippocampus",
    "hipp": "hippocampus",
    "hippocampal formation": "hippocampus",
    "subiculum": "subiculum",
    "sub": "subiculum",
    # Entorhinal
    "ec": "entorhinal_cortex",
    "entorhinal cortex": "entorhinal_cortex",
    "mec": "medial_entorhinal_cortex",
    "medial entorhinal cortex": "medial_entorhinal_cortex",
    # Amygdala
    "amygdala": "amygdala",
    "bla": "basolateral_amygdala",
    "basolateral amygdala": "basolateral_amygdala",
    "cea": "central_amygdala",
    "central amygdala": "central_amygdala",
    # Thalamus
    "thalamus": "thalamus",
    "thal": "thalamus",
    "lgn": "lateral_geniculate_nucleus",
    "lateral geniculate nucleus": "lateral_geniculate_nucleus",
    "md": "mediodorsal_thalamus",
    "mediodorsal thalamus": "mediodorsal_thalamus",
    "vpl": "thalamus",
    "vpm": "thalamus",
    "lp": "thalamus",
    # Striatum
    "striatum": "striatum",
    "dls": "dorsolateral_striatum",
    "dms": "dorsomedial_striatum",
    "dorsal striatum": "dorsal_striatum",
    "ventral striatum": "ventral_striatum",
    "nac": "nucleus_accumbens",
    "nacc": "nucleus_accumbens",
    "nucleus accumbens": "nucleus_accumbens",
    "caudate": "caudate",
    "putamen": "putamen",
    "caudoputamen": "caudate",
    # Globus pallidus
    "gp": "globus_pallidus",
    "gpe": "globus_pallidus",
    "gpi": "globus_pallidus",
    "globus pallidus": "globus_pallidus",
    # Substantia nigra / VTA
    "sn": "substantia_nigra",
    "snc": "substantia_nigra",
    "snr": "substantia_nigra",
    "substantia nigra": "substantia_nigra",
    "vta": "vta",
    "ventral tegmental area": "vta",
    # Colliculi
    "sc": "superior_colliculus",
    "superior colliculus": "superior_colliculus",
    "ic": "inferior_colliculus",
    "inferior colliculus": "inferior_colliculus",
    # Brainstem
    "brainstem": "brainstem",
    "lc": "locus_coeruleus",
    "locus coeruleus": "locus_coeruleus",
    "pag": "periaqueductal_gray",
    # Cerebellum
    "cerebellum": "cerebellum",
    "cb": "cerebellum",
    "purkinje": "cerebellum",
    # Hypothalamus
    "hypothalamus": "hypothalamus",
    "hyp": "hypothalamus",
    "lh": "hypothalamus",
    "lateral hypothalamus": "hypothalamus",
    # Septum
    "septum": "septum",
    "ls": "septum",
    "ms": "septum",
    "lateral septum": "septum",
    "medial septum": "septum",
    # Olfactory
    "ob": "olfactory_bulb",
    "olfactory bulb": "olfactory_bulb",
    "pir": "piriform_cortex",
    "piriform cortex": "piriform_cortex",
    # Retrosplenial
    "rsc": "retrosplenial_cortex",
    "retrosplenial cortex": "retrosplenial_cortex",
    # Insula
    "insula": "insula",
    "ins": "insula",
    # Auditory
    "ac": "auditory_cortex",
    "auditory cortex": "auditory_cortex",
    "a1": "auditory_cortex",
    # Neocortex broad
    "cortex": "neocortex",
    "neocortex": "neocortex",
    "isocortex": "neocortex",
    # Allen Brain Atlas mouse abbreviations
    "visp": "v1",
    "visam": "visual_cortex",
    "visal": "visual_cortex",
    "vispm": "visual_cortex",
    "visrl": "visual_cortex",
    "visli": "visual_cortex",
    "vispor": "visual_cortex",
    "vism": "area_mt",
    "mop": "motor_cortex",
    "mos": "premotor_cortex",
    "ssp": "somatosensory_cortex",
    "ssp-bfd": "somatosensory_cortex",
    "ssbbfd": "somatosensory_cortex",
    "sss": "somatosensory_cortex",
    "tea": "temporal_cortex",
    "aud": "auditory_cortex",
    "aud-v": "auditory_cortex",
    "audc": "auditory_cortex",
    "audp": "auditory_cortex",
    "audpo": "auditory_cortex",
    "audd": "auditory_cortex",
    "rsp": "retrosplenial_cortex",
    "rspagl": "retrosplenial_cortex",
    "rspd": "retrosplenial_cortex",
    "rspl": "retrosplenial_cortex",
    "ent": "entorhinal_cortex",
    "mec": "medial_entorhinal_cortex",
    "ect": "entorhinal_cortex",
    "pir": "piriform_cortex",
    "aob": "olfactory_bulb",
    "olfb": "olfactory_bulb",
    "ai": "insula",
    "aip": "insula",
    "aiv": "insula",
    "cp": "caudate",
    "str": "striatum",
    "aca": "ACC",
    "pl": "mPFC",
    "ils": "mPFC",
    "orb": "OFC",
    "orbl": "OFC",
    "orbm": "OFC",
    "orbvl": "OFC",
    "dg": "dentate_gyrus",
    "entorhinal": "entorhinal_cortex",
    "sub": "subiculum",
    "hip": "hippocampus",
    "apn": "thalamus",
    "pf": "thalamus",
    "vpm": "thalamus",
    "vpl": "thalamus",
    "vpmpvc": "thalamus",
    "vpmpc": "thalamus",
    "lpn": "thalamus",
    "lp": "thalamus",
    "lg": "lateral_geniculate_nucleus",
    "lgv": "lateral_geniculate_nucleus",
    "lgd": "lateral_geniculate_nucleus",
    "md": "mediodorsal_thalamus",
    "mdt": "mediodorsal_thalamus",
    "imf": "thalamus",
    "imd": "thalamus",
    "pvt": "thalamus",
    "av": "thalamus",
    "ad": "thalamus",
    "am": "thalamus",
    "pva": "thalamus",
    "pf": "thalamus",
    "mb": "brainstem",
    "snc": "substantia_nigra",
    "snr": "substantia_nigra",
    "vta": "vta",
    "scm": "superior_colliculus",
    "sci": "superior_colliculus",
    "scs": "superior_colliculus",
    "scig": "superior_colliculus",
    "scop": "superior_colliculus",
    "ic": "inferior_colliculus",
    "ext": "brainstem",
    "dcn": "cerebellum",
    "cbn": "cerebellum",
    "cbp": "cerebellum",
    "cbx": "cerebellum",
    "bla": "basolateral_amygdala",
    "cea": "central_amygdala",
    "la": "basolateral_amygdala",
    "ba": "basolateral_amygdala",
    "bma": "amygdala",
    "me": "amygdala",
    "coa": "amygdala",
    "nac": "nucleus_accumbens",
    "ot": "olfactory_bulb",
    "aon": "olfactory_bulb",
    "tt": "olfactory_bulb",
    "peri": "piriform_cortex",
    "lh": "hypothalamus",
    "vmh": "hypothalamus",
    "dmh": "hypothalamus",
    "pvh": "hypothalamus",
    "ah": "hypothalamus",
    "ls": "septum",
    "ms": "septum",
    "sf": "septum",
    # Unknown/none — skip
    "none": "",
    "unknown": "",
    "n/a": "",
    "na": "",
    "not specified": "",
    "": "",
}


def _map_nwb_locations(locations: list[str]) -> list[str]:
    """Map NWB electrode location strings to canonical region IDs.

    Handles Allen Brain Atlas abbreviations, hemisphere prefixes (Left/Right),
    layer suffixes (Layer II/III), and partial compound names.
    """
    import re as _re
    # Strip hemisphere prefixes and layer/stratum suffixes
    _PREFIX = _re.compile(r"^(left|right|ipsilateral|contralateral|bilateral)\s+", _re.I)
    _SUFFIX = _re.compile(r"\s*[-,]\s*(layer\s+\w+|\d+|stratum\s+\w+|superficial|deep|dorsal|ventral).*$", _re.I)

    canonical: set[str] = set()
    for loc in locations:
        raw = str(loc).strip()
        # Normalize: strip hemisphere and layer notation
        cleaned = _PREFIX.sub("", raw)
        cleaned = _SUFFIX.sub("", cleaned).lower().strip()

        # Try exact match first
        region_id = _NWB_LOCATION_MAP.get(cleaned) or _NWB_LOCATION_MAP.get(raw.lower().strip())
        if region_id:
            canonical.add(region_id)
            continue

        # Substring match: check if any key appears in the location string
        for key, rid in _NWB_LOCATION_MAP.items():
            if key and rid and len(key) >= 3 and key in cleaned:
                canonical.add(rid)
                break

    canonical.discard("")  # remove empty strings from unknown/none entries
    return sorted(canonical)


def _load_electrode_cache() -> dict[str, list[str]]:
    """Load cached NWB electrode locations: source_id → list of canonical region IDs."""
    cache: dict[str, list[str]] = {}
    if ELECTRODE_CACHE_PATH.exists():
        for line in ELECTRODE_CACHE_PATH.read_text().splitlines():
            try:
                entry = json.loads(line)
                sid = entry.get("source_id")
                if sid:
                    cache[sid] = entry.get("regions", [])
            except Exception:
                pass
    return cache


def _save_electrode_cache(source_id: str, regions: list[str]) -> None:
    with open(ELECTRODE_CACHE_PATH, "a") as f:
        f.write(json.dumps({"source_id": source_id, "regions": regions}) + "\n")


def _fetch_nwb_electrode_regions(source_id: str) -> list[str]:
    """Sample one NWB asset from the dandiset and read electrode locations.

    Returns a list of canonical brain region IDs, or [] on failure.
    """
    try:
        from dandi.dandiapi import DandiAPIClient
        import remfile
        import h5py

        with DandiAPIClient() as client:
            dandiset = client.get_dandiset(source_id, "draft")
            assets = list(dandiset.get_assets_by_glob("*.nwb"))
            if not assets:
                # Try with path filter
                assets = list(dandiset.get_assets())
                assets = [a for a in assets if a.path.endswith(".nwb")]
            if not assets:
                logger.debug("%s: no NWB assets found", source_id)
                return []

            # Use the first asset (smallest or first alphabetically)
            asset = assets[0]
            url = asset.get_content_url(follow_redirects=1, strip_query=True)
            logger.info("  %s: streaming %s (%s)", source_id, asset.path, asset.size)

            # Stream the NWB file header without downloading
            rfile = remfile.File(url)
            with h5py.File(rfile, "r") as f:
                locations: list[str] = []
                # Try standard NWB electrode table location
                if "general/extracellular_ephys/electrodes/location" in f:
                    loc_data = f["general/extracellular_ephys/electrodes/location"][:]
                    locations = [loc.decode("utf-8") if isinstance(loc, bytes) else str(loc) for loc in loc_data]
                elif "electrodes/location" in f:
                    loc_data = f["electrodes/location"][:]
                    locations = [loc.decode("utf-8") if isinstance(loc, bytes) else str(loc) for loc in loc_data]

                if not locations:
                    logger.debug("%s: no electrode location column found", source_id)
                    return []

                # Deduplicate and map
                unique_locs = list(dict.fromkeys(locations))
                regions = _map_nwb_locations(unique_locs)
                logger.info("  %s: electrode locations %s → regions %s", source_id, unique_locs[:5], regions)
                return regions

    except ImportError as e:
        logger.warning("Missing dependency for NWB streaming (%s); skipping NWB extraction", e)
        return []
    except Exception as e:
        logger.debug("%s: NWB extraction failed: %s", source_id, e)
        return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    corpus = [json.loads(l) for l in CORPUS_PATH.read_text().strip().splitlines() if l.strip()]
    electrode_cache = _load_electrode_cache()

    ephys_mods = {"extracellular_ephys", "lfp", "neuropixels"}

    def _get_mod_ids(field):
        if not field:
            return set()
        return {(v.get("id") if isinstance(v, dict) else v) for v in field if v}

    targets = [
        r for r in corpus
        if r.get("source") == "dandi"
        and not r.get("brain_regions")
        and bool(_get_mod_ids(r.get("modalities") or {}) & ephys_mods)
    ]
    logger.info("DANDI ephys records without brain regions: %d", len(targets))

    if args.limit:
        targets = targets[: args.limit]

    enriched_map: dict[str, list[str]] = {}
    for i, rec in enumerate(targets, 1):
        source_id = rec["source_id"]
        if source_id in electrode_cache:
            regions = electrode_cache[source_id]
            logger.info("[%d/%d] Cache hit %s → %s", i, len(targets), source_id, regions)
        else:
            logger.info("[%d/%d] Fetching NWB electrode locations for %s…", i, len(targets), source_id)
            regions = _fetch_nwb_electrode_regions(source_id)
            if not args.dry_run:
                _save_electrode_cache(source_id, regions)
            time.sleep(0.5)

        if regions:
            enriched_map[source_id] = regions

    logger.info("Records with electrode-derived regions: %d", len(enriched_map))

    if args.dry_run:
        logger.info("[dry-run] skipping corpus write")
        return 0

    output = []
    updated = 0
    for rec in corpus:
        sid = rec.get("source_id", "")
        if rec.get("source") == "dandi" and sid in enriched_map:
            existing = rec.get("brain_regions") or []
            merged = list(dict.fromkeys(enriched_map[sid] + existing))
            output.append({**rec, "brain_regions": merged})
            updated += 1
        else:
            output.append(rec)

    CORPUS_PATH.write_text("\n".join(json.dumps(r) for r in output) + "\n")
    logger.info("Updated %d records → %s", updated, CORPUS_PATH)

    with_r = sum(1 for r in output if r.get("brain_regions"))
    logger.info("Brain region coverage: %d/%d = %d%%", with_r, len(output), 100 * with_r // len(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
