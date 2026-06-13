"""Conservative alias normalization for concept names in the concept memory module."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Alias map definition
# ---------------------------------------------------------------------------

# Each inner list is: [alias1, alias2, ..., canonical]
# The last element is the canonical form.
_ALIAS_GROUPS: list[tuple[list[str], str]] = [
    # --- Modalities ---
    (
        ["neuropixels", "neuropixel", "npx", "neuropixels probe"],
        "neuropixels",
    ),
    (
        [
            "calcium imaging",
            "2p imaging",
            "two-photon calcium imaging",
            "2-photon imaging",
            "two photon imaging",
            "ca2+ imaging",
            "ca imaging",
        ],
        "calcium_imaging",
    ),
    (
        ["fmri", "functional mri", "functional magnetic resonance imaging", "bold fmri"],
        "fmri",
    ),
    (
        ["eeg", "electroencephalography", "electroencephalogram"],
        "eeg",
    ),
    (
        ["ecog", "electrocorticography", "electrocorticogram"],
        "ecog",
    ),
    (
        ["lfp", "local field potential"],
        "lfp",
    ),
    (
        [
            "extracellular_ephys",
            "extracellular electrophysiology",
            "extracellular ephys",
            "ephys",
        ],
        "extracellular_ephys",
    ),
    (
        ["miniscope", "mini-scope", "mini scope", "miniscope imaging"],
        "miniscope",
    ),
    # --- Methods ---
    (
        ["spike sorting", "spike-sorting", "spikesort", "kilosort", "phy spike sorting"],
        "spike_sorting",
    ),
    (
        ["deeplabcut", "dlc", "deep lab cut"],
        "deeplabcut",
    ),
    (
        ["suite2p", "suite 2p", "suite-2p"],
        "suite2p",
    ),
    (
        ["lfads", "latent factor analysis via dynamical systems"],
        "lfads",
    ),
    (
        ["pca", "principal component analysis"],
        "pca",
    ),
    (
        ["decoding", "neural decoding", "population decoding"],
        "decoding",
    ),
    # --- Species ---
    (
        ["mouse", "mice", "mus musculus"],
        "mouse",
    ),
    (
        ["rat", "rattus norvegicus", "rats"],
        "rat",
    ),
    (
        ["human", "homo sapiens", "humans", "human subjects"],
        "human",
    ),
    (
        [
            "macaque",
            "rhesus macaque",
            "macaca mulatta",
            "non-human primate",
            "nhp",
        ],
        "macaque",
    ),
    (
        ["zebrafish", "danio rerio", "zebra fish"],
        "zebrafish",
    ),
    (
        ["drosophila", "fly", "fruit fly", "d. melanogaster"],
        "drosophila",
    ),
    # --- Tasks ---
    (
        ["go/no-go", "go-nogo", "go nogo", "go no go"],
        "go_nogo",
    ),
    (
        ["delay period", "working memory delay", "delayed match to sample"],
        "working_memory",
    ),
    (
        ["visual discrimination", "visual task", "orientation discrimination"],
        "visual_discrimination",
    ),
]


def build_alias_map() -> dict[str, str]:
    """Return the full raw→canonical alias map.

    Keys are lowercased alias strings; values are canonical concept names.
    """
    mapping: dict[str, str] = {}
    for aliases, canonical in _ALIAS_GROUPS:
        for alias in aliases:
            mapping[alias.lower()] = canonical
        # Ensure the canonical itself maps to itself
        mapping[canonical.lower()] = canonical
    return mapping


# Module-level singleton — built once per import.
_ALIAS_MAP: dict[str, str] = build_alias_map()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_concept_name(raw: str, alias_map: dict[str, str] | None = None) -> str:
    """Normalize a raw concept name to its canonical form.

    Steps:
    1. Lowercase and strip whitespace.
    2. Look up in alias map → return canonical if found.
    3. Otherwise return the lowercased+stripped version with spaces replaced by
       underscores.

    Conservative: only normalizes things present in the alias map.
    Callers should preserve the original alias alongside the canonical.
    """
    _map = alias_map if alias_map is not None else _ALIAS_MAP
    key = raw.lower().strip()
    if key in _map:
        return _map[key]
    return key.replace(" ", "_")


def is_same_concept(name_a: str, name_b: str, alias_map: dict[str, str] | None = None) -> bool:
    """Return True if both names normalize to the same canonical form."""
    return normalize_concept_name(name_a, alias_map) == normalize_concept_name(name_b, alias_map)


def concept_aliases(raw: str, alias_map: dict[str, str] | None = None) -> list[str]:
    """Return all known aliases for a concept name.

    Looks up both the raw form and its normalized form to find all keys that
    map to the same canonical. Returns a deduplicated list that always includes
    the canonical.
    """
    _map = alias_map if alias_map is not None else _ALIAS_MAP
    canonical = normalize_concept_name(raw, _map)
    seen: set[str] = {canonical}
    result: list[str] = [canonical]
    for alias, target in _map.items():
        if target == canonical and alias not in seen:
            seen.add(alias)
            result.append(alias)
    return result
