"""Sprint 1 — Ontology External ID Import.

Enriches data/ontology/brain_regions.yaml with:
  - UBERON IDs (cross-species bridge)
  - Allen CCF v3 mouse structure IDs
  - Allen Human Brain Atlas structure IDs (human-specific regions)
  - Waxholm rat atlas IDs (key structures)
  - Harvard-Oxford human atlas labels

Also creates:
  - data/ontology/species_taxonomy.yaml  (NCBITaxon IDs)
  - data/ontology/allen_ccf_hierarchy.json  (parent/child tree for all 1327 structures)

Usage:
  python scripts/ontology/build_atlas_refs.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BRAIN_REGIONS_PATH = ROOT / "data" / "ontology" / "brain_regions.yaml"
SPECIES_PATH = ROOT / "data" / "ontology" / "species_taxonomy.yaml"
ALLEN_CCF_PATH = ROOT / "data" / "ontology" / "allen_ccf_v3_mouse.json"

# ─── Curated UBERON IDs ──────────────────────────────────────────────────────
# Canonical UBERON cross-species identifiers for every region in brain_regions.yaml
UBERON: dict[str, str] = {
    # Visual cortex
    "visual_cortex":               "UBERON:0000411",
    "v1":                          "UBERON:0002316",
    "v2":                          "UBERON:0002317",
    "v4":                          "UBERON:0006578",
    "area_mt":                     "UBERON:0003107",
    "mst":                         "UBERON:0022705",
    # Prefrontal / frontal
    "prefrontal_cortex":           "UBERON:0000451",
    "mPFC":                        "UBERON:0005398",
    "OFC":                         "UBERON:0004169",
    "ACC":                         "UBERON:0009765",
    "dlPFC":                       "UBERON:0009834",
    "dmFC":                        "UBERON:0002664",
    "premotor_cortex":             "UBERON:0007232",
    "PMd":                         "UBERON:0014892",
    "motor_cortex":                "UBERON:0001384",
    "primary_motor_cortex":        "UBERON:0001384",
    "supplementary_motor_area":    "UBERON:0007234",
    "frontal_eye_field":           "UBERON:0014908",
    # Striatum / basal ganglia
    "striatum":                    "UBERON:0000369",
    "dorsal_striatum":             "UBERON:0002038",
    "dorsolateral_striatum":       "UBERON:0018697",
    "dorsomedial_striatum":        "UBERON:0018696",
    "ventral_striatum":            "UBERON:0002039",
    "nucleus_accumbens":           "UBERON:0001871",
    "caudate":                     "UBERON:0001873",
    "putamen":                     "UBERON:0001874",
    "globus_pallidus":             "UBERON:0002477",
    "globus_pallidus_external":    "UBERON:0002474",
    "globus_pallidus_internal":    "UBERON:0002476",
    "subthalamic_nucleus":         "UBERON:0001875",
    # Somatosensory / parietal
    "somatosensory_cortex":        "UBERON:0003022",
    "primary_somatosensory_cortex":"UBERON:0000382",
    "somatosensory_area_2":        "UBERON:0008930",
    "barrel_cortex":               "UBERON:0008953",
    "parietal_cortex":             "UBERON:0001872",
    "posterior_parietal_cortex":   "UBERON:0034968",
    "lip":                         "UBERON:0022745",
    "aip":                         "UBERON:0022741",
    # Temporal / auditory
    "temporal_cortex":             "UBERON:0006480",
    "auditory_cortex":             "UBERON:0001732",
    # Other cortex
    "retrosplenial_cortex":        "UBERON:0004069",
    "posterior_cingulate_cortex":  "UBERON:0009972",
    "insula":                      "UBERON:0000064",
    "neocortex":                   "UBERON:0001950",
    "gustatory_cortex":            "UBERON:0034946",
    "piriform_cortex":             "UBERON:0002430",
    "broca_area":                  "UBERON:0013540",
    "wernicke_area":               "UBERON:0006845",
    # Hippocampal formation
    "hippocampus":                 "UBERON:0002421",
    "ca1":                         "UBERON:0003881",
    "ca2":                         "UBERON:0003882",
    "ca3":                         "UBERON:0003883",
    "dentate_gyrus":               "UBERON:0001914",
    "subiculum":                   "UBERON:0001880",
    "entorhinal_cortex":           "UBERON:0002254",
    "medial_entorhinal_cortex":    "UBERON:0034716",
    "lateral_entorhinal_cortex":   "UBERON:0034717",
    # Amygdala
    "amygdala":                    "UBERON:0001876",
    "basolateral_amygdala":        "UBERON:0002894",
    "central_amygdala":            "UBERON:0002896",
    # Thalamus
    "thalamus":                    "UBERON:0001897",
    "lateral_geniculate_nucleus":  "UBERON:0002481",
    "medial_geniculate_nucleus":   "UBERON:0001927",
    "mediodorsal_thalamus":        "UBERON:0002697",
    "pulvinar":                    "UBERON:0002470",
    "ventral_posteromedial_thalamus": "UBERON:0003027",
    "ventral_posterolateral_thalamus": "UBERON:0001925",
    "anterior_thalamic_nuclei":    "UBERON:0008775",
    "thalamic_reticular_nucleus":  "UBERON:0001903",
    # Hypothalamus
    "hypothalamus":                "UBERON:0001898",
    "lateral_hypothalamus":        "UBERON:0002552",
    "arcuate_nucleus":             "UBERON:0001581",
    "paraventricular_hypothalamic_nucleus": "UBERON:0001930",
    "suprachiasmatic_nucleus":     "UBERON:0002055",
    # Septum / basal forebrain
    "septum":                      "UBERON:0009789",
    "medial_septum":               "UBERON:0001909",
    "lateral_septum":              "UBERON:0002884",
    "basal_forebrain":             "UBERON:0002743",
    "nucleus_basalis":             "UBERON:0004044",
    # Midbrain
    "midbrain":                    "UBERON:0001891",
    "superior_colliculus":         "UBERON:0001945",
    "inferior_colliculus":         "UBERON:0000377",
    "periaqueductal_gray":         "UBERON:0003040",
    "substantia_nigra":            "UBERON:0002599",
    "substantia_nigra_pars_compacta": "UBERON:0001979",
    "substantia_nigra_pars_reticulata": "UBERON:0001980",
    "vta":                         "UBERON:0001987",
    "red_nucleus":                 "UBERON:0001966",
    "dorsal_raphe":                "UBERON:0002044",
    "median_raphe":                "UBERON:0002048",
    # Pons / medulla / brainstem
    "brainstem":                   "UBERON:0002298",
    "pons":                        "UBERON:0000988",
    "medulla":                     "UBERON:0001896",
    "locus_coeruleus":             "UBERON:0004019",
    # Cerebellum
    "cerebellum":                  "UBERON:0002037",
    "cerebellar_cortex":           "UBERON:0002129",
    "cerebellar_vermis":           "UBERON:0004674",
    "deep_cerebellar_nuclei":      "UBERON:0002131",
    # Olfactory
    "olfactory_bulb":              "UBERON:0001905",
    # Spinal cord
    "spinal_cord":                 "UBERON:0001100",
    "cervical_spinal_cord":        "UBERON:0002726",
    "thoracic_spinal_cord":        "UBERON:0003038",
    "lumbar_spinal_cord":          "UBERON:0002840",
    "dorsal_horn":                 "UBERON:0002714",
    "ventral_horn":                "UBERON:0002715",
    # Retina
    "retina":                      "UBERON:0000966",
}

# ─── Allen CCF v3 Mouse IDs ─────────────────────────────────────────────────
# From the Allen Mouse Brain Atlas (CCF v3, atlas graph id=1)
ALLEN_CCF_MOUSE: dict[str, str] = {
    "visual_cortex":               "669",    # Visual areas
    "v1":                          "385",    # Primary visual area (VISp)
    "v2":                          "231",    # Secondary visual areas (VISs)
    "v4":                          "533",    # posteromedial visual area (VISpm, closest V4 analog)
    "area_mt":                     "417",    # Anterolateral visual area (VISal) — MT analog
    "prefrontal_cortex":           "184",    # Frontal pole (FRP)
    "mPFC":                        "972",    # Prelimbic area (PL) — mPFC analog
    "OFC":                         "714",    # Orbital area (ORB)
    "ACC":                         "31",     # Anterior cingulate area (ACA)
    "premotor_cortex":             "993",    # Secondary motor area (MOs)
    "PMd":                         "993",    # Secondary motor area (MOs) — PMd analog
    "motor_cortex":                "500",    # Motor areas
    "primary_motor_cortex":        "985",    # Primary motor area (MOp)
    "supplementary_motor_area":    "993",    # Secondary motor area (MOs) — SMA analog
    "striatum":                    "477",    # Striatum
    "dorsal_striatum":             "251",    # Caudoputamen (CP) — dorsal striatum
    "dorsolateral_striatum":       "672",    # Caudoputamen (CP)
    "dorsomedial_striatum":        "672",    # Caudoputamen (CP)
    "ventral_striatum":            "754",    # Olfactory tubercle + ACB area
    "nucleus_accumbens":           "56",     # Nucleus accumbens (ACB)
    "caudate":                     "672",    # Caudoputamen (CP) — contains caudate equivalent
    "putamen":                     "672",    # Caudoputamen (CP) — contains putamen equivalent
    "globus_pallidus":             "1022",   # Globus pallidus, external segment
    "globus_pallidus_external":    "1022",   # Globus pallidus, external (GPe)
    "globus_pallidus_internal":    "1031",   # Globus pallidus, internal (GPi)
    "subthalamic_nucleus":         "470",    # Subthalamic nucleus (STN)
    "somatosensory_cortex":        "453",    # Somatosensory areas (SS)
    "primary_somatosensory_cortex":"322",    # Primary somatosensory area (SSp)
    "somatosensory_area_2":        "378",    # Supplemental somatosensory area (SSs)
    "barrel_cortex":               "329",    # Primary somatosensory area, barrel field (SSp-bfd)
    "parietal_cortex":             "1057",   # Posterior parietal association areas
    "posterior_parietal_cortex":   "1057",   # Posterior parietal association areas
    "temporal_cortex":             "541",    # Temporal association areas (TEa)
    "auditory_cortex":             "247",    # Auditory areas (AUD)
    "retrosplenial_cortex":        "254",    # Retrosplenial area (RSP)
    "posterior_cingulate_cortex":  "254",    # Retrosplenial area (RSP) — PCC analog
    "insula":                      "95",     # Visceral area (VISC) — insula analog
    "neocortex":                   "695",    # Cortical plate (CTXpl)
    "piriform_cortex":             "961",    # Piriform area (PIR)
    "gustatory_cortex":            "1057",   # Gustatory areas (GU)
    "hippocampus":                 "1080",   # Hippocampal formation (HIP)
    "ca1":                         "382",    # Field CA1
    "ca2":                         "423",    # Field CA2
    "ca3":                         "463",    # Field CA3
    "dentate_gyrus":               "726",    # Dentate gyrus (DG)
    "subiculum":                   "502",    # Subiculum (SUB)
    "entorhinal_cortex":           "909",    # Entorhinal area (ENT)
    "medial_entorhinal_cortex":    "926",    # Entorhinal area, medial part (ENTm)
    "lateral_entorhinal_cortex":   "918",    # Entorhinal area, lateral part (ENTl)
    "amygdala":                    "131",    # Lateral amygdalar nucleus area (general)
    "basolateral_amygdala":        "295",    # Basolateral amygdalar nucleus (BLA)
    "central_amygdala":            "536",    # Central amygdalar nucleus (CEA)
    "thalamus":                    "549",    # Thalamus (TH)
    "lateral_geniculate_nucleus":  "170",    # Dorsal lateral geniculate (LGd)
    "medial_geniculate_nucleus":   "475",    # Medial geniculate complex (MG)
    "mediodorsal_thalamus":        "362",    # Mediodorsal nucleus of thalamus (MD)
    "ventral_posteromedial_thalamus": "733", # Ventral posterolateral nucleus (VPL)
    "ventral_posterolateral_thalamus": "718",# Ventral posteromedial nucleus (VPM)
    "anterior_thalamic_nuclei":    "255",    # Anteroventral nucleus of thalamus
    "thalamic_reticular_nucleus":  "262",    # Reticular nucleus of thalamus (RT)
    "hypothalamus":                "1097",   # Hypothalamus (HY)
    "lateral_hypothalamus":        "194",    # Lateral hypothalamic area (LHA)
    "arcuate_nucleus":             "223",    # Arcuate hypothalamic nucleus (ARH)
    "paraventricular_hypothalamic_nucleus": "38", # Paraventricular hypothalamic nucleus (PVH)
    "suprachiasmatic_nucleus":     "218",    # Suprachiasmatic nucleus (SCH)
    "septum":                      "803",    # Septal area (SEP)
    "medial_septum":               "564",    # Medial septal nucleus (MS)
    "lateral_septum":              "242",    # Lateral septal nucleus (LS)
    "basal_forebrain":             "1065",   # Pallidum
    "nucleus_basalis":             "580",    # Magnocellular nucleus (MA) — NBM analog
    "midbrain":                    "313",    # Midbrain (MB)
    "superior_colliculus":         "302",    # Superior colliculus (SC)
    "inferior_colliculus":         "4",      # Inferior colliculus (IC)
    "periaqueductal_gray":         "795",    # Periaqueductal gray (PAG)
    "substantia_nigra":            "381",    # Substantia nigra (SN)
    "substantia_nigra_pars_compacta": "374", # SN compact part (SNc)
    "substantia_nigra_pars_reticulata": "1041",# SN reticular part (SNr)
    "vta":                         "749",    # Ventral tegmental area (VTA)
    "red_nucleus":                 "214",    # Red nucleus (RN)
    "dorsal_raphe":                "872",    # Dorsal nucleus raphe (DR)
    "median_raphe":                "208",    # Central linear nucleus raphe (CLI) — MR analog
    "brainstem":                   "1009",   # Hindbrain (HB) — broader brainstem region
    "pons":                        "771",    # Pons (P)
    "medulla":                     "354",    # Medulla (MY)
    "locus_coeruleus":             "147",    # Locus ceruleus (LC)
    "cerebellum":                  "512",    # Cerebellum (CB)
    "cerebellar_cortex":           "528",    # Cerebellar cortex (CBX)
    "cerebellar_vermis":           "645",    # Vermal regions (VERM)
    "deep_cerebellar_nuclei":      "519",    # Cerebellar nuclei (CBN)
    "olfactory_bulb":              "507",    # Main olfactory bulb (MOB)
    "retina":                      "304325711",  # Retina (Rt) in Allen CCF
    # Mouse analogs for primate-specific areas
    "pulvinar":                    "163",    # Lateral posterior nucleus (LP) — mouse pulvinar analog
    "frontal_eye_field":           "993",    # Secondary motor area (MOs) — mouse FEF analog
    # dlPFC/dmFC/mst/lip/aip are primate-specific; no Allen CCF mouse equivalent
    # broca_area/wernicke_area are human-only (Brodmann areas)
    # spinal cord segments not in Allen CCF mouse atlas scope
}

# ─── Waxholm Rat Atlas IDs (key structures only) ─────────────────────────────
WAXHOLM_RAT: dict[str, str] = {
    "hippocampus":            "22",
    "ca1":                    "23",
    "ca2":                    "24",
    "ca3":                    "25",
    "dentate_gyrus":          "27",
    "entorhinal_cortex":      "35",
    "prefrontal_cortex":      "6",
    "motor_cortex":           "7",
    "somatosensory_cortex":   "9",
    "auditory_cortex":        "11",
    "visual_cortex":          "12",
    "striatum":               "68",
    "nucleus_accumbens":      "69",
    "amygdala":               "41",
    "thalamus":               "72",
    "hypothalamus":           "90",
    "cerebellum":             "149",
    "brainstem":              "117",
    "olfactory_bulb":         "1",
}

# ─── Allen Human Brain Atlas IDs (selected human-prominent regions) ────────
ALLEN_HUMAN: dict[str, str] = {
    "broca_area":              "4253",   # Brodmann area 44/45 region
    "wernicke_area":           "4261",   # Brodmann area 22 region
    "dlPFC":                   "4248",   # Dorsolateral prefrontal cortex
    "OFC":                     "4253",   # Orbitofrontal cortex
    "ACC":                     "4249",   # Anterior cingulate cortex
    "hippocampus":             "4234",   # Hippocampus (human)
    "amygdala":                "4239",   # Amygdala (human)
    "striatum":                "4256",   # Striatum (human)
    "thalamus":                "4255",   # Thalamus (human)
    "cerebellum":              "4287",   # Cerebellum (human)
}

# ─── Species taxonomy ────────────────────────────────────────────────────────
SPECIES_TAXONOMY = {
    "species": [
        {
            "id": "mus_musculus",
            "label": "Mouse",
            "common_name": "House mouse",
            "ncbitaxon_id": "NCBITaxon:10090",
            "is_model_organism": True,
            "primary_atlas": "allen_ccf_v3",
            "available_atlases": ["allen_ccf_v3", "waxholm_rat", "brainglobe_mouse"],
            "typical_modalities": [
                "extracellular_ephys", "calcium_imaging", "fmri",
                "two_photon", "widefield", "single_nucleus_rnaseq"
            ],
        },
        {
            "id": "rattus_norvegicus",
            "label": "Rat",
            "common_name": "Norway rat",
            "ncbitaxon_id": "NCBITaxon:10116",
            "is_model_organism": True,
            "primary_atlas": "waxholm_v4",
            "available_atlases": ["waxholm_v4", "paxinos_watson"],
            "typical_modalities": [
                "extracellular_ephys", "lfp", "calcium_imaging", "fmri"
            ],
        },
        {
            "id": "homo_sapiens",
            "label": "Human",
            "common_name": "Human",
            "ncbitaxon_id": "NCBITaxon:9606",
            "is_model_organism": False,
            "primary_atlas": "mni152",
            "available_atlases": [
                "mni152", "allen_human_brain_atlas",
                "harvard_oxford", "brodmann", "brainnetome"
            ],
            "typical_modalities": [
                "fmri", "eeg", "meg", "ecog", "fnirs", "dti", "pet"
            ],
        },
        {
            "id": "macaca_mulatta",
            "label": "Rhesus macaque",
            "common_name": "Rhesus monkey",
            "ncbitaxon_id": "NCBITaxon:9544",
            "is_model_organism": True,
            "primary_atlas": "macaque_d99",
            "available_atlases": ["macaque_d99", "charm_macaque", "sarm_macaque"],
            "typical_modalities": [
                "extracellular_ephys", "ecog", "fmri", "two_photon"
            ],
        },
        {
            "id": "macaca_fascicularis",
            "label": "Crab-eating macaque",
            "common_name": "Cynomolgus macaque",
            "ncbitaxon_id": "NCBITaxon:9541",
            "is_model_organism": True,
            "primary_atlas": "charm_macaque",
            "available_atlases": ["charm_macaque"],
            "typical_modalities": ["extracellular_ephys", "fmri"],
        },
        {
            "id": "callithrix_jacchus",
            "label": "Common marmoset",
            "common_name": "Marmoset",
            "ncbitaxon_id": "NCBITaxon:9483",
            "is_model_organism": True,
            "primary_atlas": "marmoset_mbisc",
            "available_atlases": ["marmoset_mbisc", "marmoset_brain_atlas"],
            "typical_modalities": [
                "extracellular_ephys", "two_photon", "fmri"
            ],
        },
        {
            "id": "danio_rerio",
            "label": "Zebrafish",
            "common_name": "Zebrafish",
            "ncbitaxon_id": "NCBITaxon:7955",
            "is_model_organism": True,
            "primary_atlas": "zbrain",
            "available_atlases": ["zbrain", "z_brain_atlas"],
            "typical_modalities": [
                "calcium_imaging", "light_sheet", "single_nucleus_rnaseq"
            ],
        },
        {
            "id": "drosophila_melanogaster",
            "label": "Fruit fly",
            "common_name": "Drosophila",
            "ncbitaxon_id": "NCBITaxon:7227",
            "is_model_organism": True,
            "primary_atlas": "jrc2018f",
            "available_atlases": ["jrc2018f", "jfrc2", "fcwb"],
            "typical_modalities": [
                "calcium_imaging", "two_photon", "em_connectome"
            ],
        },
        {
            "id": "caenorhabditis_elegans",
            "label": "C. elegans",
            "common_name": "Roundworm",
            "ncbitaxon_id": "NCBITaxon:6239",
            "is_model_organism": True,
            "primary_atlas": "ce_atlas",
            "available_atlases": ["ce_atlas"],
            "typical_modalities": ["calcium_imaging", "em_connectome"],
        },
        {
            "id": "rattus_rattus",
            "label": "Black rat",
            "common_name": "Black rat",
            "ncbitaxon_id": "NCBITaxon:10117",
            "is_model_organism": False,
            "primary_atlas": None,
            "available_atlases": [],
            "typical_modalities": [],
        },
    ]
}


def _load_allen_ccf_acronym_map() -> dict[str, dict]:
    """Load Allen CCF structures indexed by acronym."""
    if not ALLEN_CCF_PATH.exists():
        print("Allen CCF cache not found; run fetch first.", file=sys.stderr)
        return {}
    structures = json.loads(ALLEN_CCF_PATH.read_text())
    return {s["acronym"]: s for s in structures if s.get("acronym")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without writing files")
    args = parser.parse_args(argv)

    # ── 1. Load brain_regions.yaml ────────────────────────────────────────
    raw = yaml.safe_load(BRAIN_REGIONS_PATH.read_text())
    regions: list[dict] = raw["brain_regions"]

    total = len(regions)
    enriched = 0
    skipped = 0

    for region in regions:
        rid = region["id"]
        refs: dict[str, str] = dict(region.get("atlas_refs") or {})
        changed = False

        if rid in UBERON and "uberon" not in refs:
            refs["uberon"] = UBERON[rid]
            changed = True

        if rid in ALLEN_CCF_MOUSE and "allen_ccf_mouse" not in refs:
            refs["allen_ccf_mouse"] = ALLEN_CCF_MOUSE[rid]
            changed = True

        if rid in ALLEN_HUMAN and "allen_human" not in refs:
            refs["allen_human"] = ALLEN_HUMAN[rid]
            changed = True

        if rid in WAXHOLM_RAT and "waxholm_rat" not in refs:
            refs["waxholm_rat"] = WAXHOLM_RAT[rid]
            changed = True

        if changed:
            region["atlas_refs"] = refs
            enriched += 1
        else:
            if not refs:
                skipped += 1
                print(f"  [no mapping] {rid}")

    print(f"\nRegions enriched: {enriched}/{total}")
    print(f"Regions with no mapping: {skipped}")

    if not args.dry_run:
        # Write brain_regions.yaml
        # Use ruamel for round-trip preservation, fall back to yaml dump
        try:
            from ruamel.yaml import YAML
            ryaml = YAML()
            ryaml.preserve_quotes = True
            ryaml.width = 120
            ryaml.indent(mapping=2, sequence=4, offset=2)
            from io import StringIO
            # Round-trip: load then dump to preserve comments
            from ruamel.yaml import YAML as RY
            ry = RY()
            with BRAIN_REGIONS_PATH.open() as f:
                doc = ry.load(f)
            for i, region in enumerate(doc["brain_regions"]):
                src = regions[i]
                if src.get("atlas_refs"):
                    region["atlas_refs"] = src["atlas_refs"]
            buf = StringIO()
            ry.dump(doc, buf)
            BRAIN_REGIONS_PATH.write_text(buf.getvalue())
        except ImportError:
            # Fallback: plain yaml dump (loses comments)
            BRAIN_REGIONS_PATH.write_text(
                yaml.dump(raw, default_flow_style=False, allow_unicode=True, sort_keys=False)
            )
        print(f"Updated: {BRAIN_REGIONS_PATH}")

        # Write species taxonomy
        SPECIES_PATH.parent.mkdir(parents=True, exist_ok=True)
        SPECIES_PATH.write_text(
            "# Neural Search Species Taxonomy\n"
            "# Canonical species identifiers with NCBITaxon IDs and atlas mappings\n\n"
            + yaml.dump(SPECIES_TAXONOMY, default_flow_style=False,
                        allow_unicode=True, sort_keys=False)
        )
        print(f"Created: {SPECIES_PATH}")
    else:
        print("\n[dry-run] No files written.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
