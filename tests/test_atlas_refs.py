"""Tests for Sprint 1 atlas-ref helpers and enriched brain_regions.yaml."""


from neural_search.ontology.loader import (
    get_brain_regions,
    get_region_allen_ccf_id,
    get_region_atlas_refs,
    get_region_id_by_alias,
    get_region_uberon_id,
    get_regions_by_allen_ccf,
    get_regions_by_uberon,
)


class TestBrainRegionAtlasRefs:
    def test_all_106_regions_have_atlas_refs(self):
        regions = get_brain_regions()
        assert len(regions) >= 106
        for region in regions:
            assert region.atlas_refs, f"Region {region.id!r} has no atlas_refs"

    def test_all_regions_have_uberon_id(self):
        regions = get_brain_regions()
        missing = [r.id for r in regions if "uberon" not in r.atlas_refs]
        assert not missing, f"Regions missing UBERON ID: {missing}"

    # Primate-specific and spinal-cord regions have no Allen CCF mouse equivalent:
    HUMAN_PRIMATE_ONLY = frozenset({
        "dlPFC", "dmFC", "mst", "lip", "aip",
        "broca_area", "wernicke_area",
        # Primate neocortical gyri with no direct mouse Allen CCF homologue
        "superior_temporal_gyrus", "inferior_frontal_gyrus", "middle_temporal_gyrus",
        # Spinal cord segments not in Allen CCF mouse atlas
        "spinal_cord", "cervical_spinal_cord", "thoracic_spinal_cord",
        "lumbar_spinal_cord", "dorsal_horn", "ventral_horn",
    })

    def test_mouse_regions_have_allen_ccf_id(self):
        regions = get_brain_regions()
        missing = [
            r.id for r in regions
            if "allen_ccf_mouse" not in r.atlas_refs
            and r.id not in self.HUMAN_PRIMATE_ONLY
        ]
        assert not missing, f"Cross-species regions missing Allen CCF mouse ID: {missing}"

    def test_allen_ccf_coverage_above_85_percent(self):
        regions = get_brain_regions()
        with_ccf = sum(1 for r in regions if "allen_ccf_mouse" in r.atlas_refs)
        pct = with_ccf / len(regions)
        assert pct >= 0.85, f"Allen CCF coverage too low: {pct:.0%} ({with_ccf}/{len(regions)})"

    def test_uberon_ids_start_with_uberon_prefix(self):
        regions = get_brain_regions()
        for r in regions:
            uberon = r.atlas_refs.get("uberon", "")
            assert uberon.startswith("UBERON:"), (
                f"Region {r.id!r} has malformed UBERON ID: {uberon!r}"
            )

    def test_allen_ccf_ids_are_numeric_or_known_special(self):
        regions = get_brain_regions()
        for r in regions:
            ccf = r.atlas_refs.get("allen_ccf_mouse", "")
            assert ccf.lstrip("-").isdigit() or ccf in ("", "SC"), (
                f"Region {r.id!r} has non-numeric Allen CCF ID: {ccf!r}"
            )


class TestGetRegionAtlasRefs:
    def test_ca1_refs(self):
        refs = get_region_atlas_refs("ca1")
        assert refs["uberon"] == "UBERON:0003881"
        assert refs["allen_ccf_mouse"] == "382"

    def test_hippocampus_refs(self):
        refs = get_region_atlas_refs("hippocampus")
        assert refs["uberon"] == "UBERON:0002421"

    def test_vta_allen_ccf(self):
        assert get_region_allen_ccf_id("vta") == "749"

    def test_motor_cortex_uberon(self):
        assert get_region_uberon_id("motor_cortex") == "UBERON:0001384"

    def test_unknown_region_returns_empty(self):
        assert get_region_atlas_refs("not_a_real_region") == {}

    def test_uberon_helper_returns_none_for_unknown(self):
        assert get_region_uberon_id("not_a_real_region") is None

    def test_allen_helper_returns_none_for_unknown(self):
        assert get_region_allen_ccf_id("not_a_real_region") is None


class TestGetRegionIdByAlias:
    def test_canonical_id_resolves_to_itself(self):
        assert get_region_id_by_alias("hippocampus") == "hippocampus"

    def test_case_insensitive_match(self):
        assert get_region_id_by_alias("Hippocampus") == "hippocampus"

    def test_label_resolves_to_canonical_id(self):
        regions = {r.id: r for r in get_brain_regions()}
        ca1 = regions["ca1"]
        assert get_region_id_by_alias(ca1.label) == "ca1"

    def test_alias_resolves_to_canonical_id(self):
        regions = {r.id: r for r in get_brain_regions()}
        vta = regions["vta"]
        assert vta.aliases, "expected VTA to have at least one alias for this test"
        assert get_region_id_by_alias(vta.aliases[0]) == "vta"

    def test_unknown_text_returns_none(self):
        assert get_region_id_by_alias("not a real region at all") is None

    def test_whitespace_is_stripped(self):
        assert get_region_id_by_alias("  hippocampus  ") == "hippocampus"


class TestReverseAtlasLookup:
    def test_uberon_reverse_lookup_hippocampus(self):
        regions = get_regions_by_uberon("UBERON:0002421")
        ids = {r.id for r in regions}
        assert "hippocampus" in ids

    def test_uberon_reverse_lookup_ca1(self):
        regions = get_regions_by_uberon("UBERON:0003881")
        assert any(r.id == "ca1" for r in regions)

    def test_allen_reverse_lookup_vta(self):
        regions = get_regions_by_allen_ccf("749")
        assert any(r.id == "vta" for r in regions)

    def test_reverse_lookup_unknown_returns_empty(self):
        assert get_regions_by_uberon("UBERON:9999999") == []
        assert get_regions_by_allen_ccf("99999") == []

    def test_motor_cortex_and_primary_share_uberon(self):
        regions = get_regions_by_uberon("UBERON:0001384")
        ids = {r.id for r in regions}
        assert "motor_cortex" in ids
        assert "primary_motor_cortex" in ids


class TestSpeciesTaxonomy:
    def test_species_taxonomy_file_exists(self):
        from pathlib import Path
        path = Path("data/ontology/species_taxonomy.yaml")
        assert path.exists(), "species_taxonomy.yaml not found"

    def test_species_taxonomy_has_key_species(self):
        from pathlib import Path

        import yaml
        raw = yaml.safe_load(Path("data/ontology/species_taxonomy.yaml").read_text())
        ids = {s["id"] for s in raw["species"]}
        assert {"mus_musculus", "rattus_norvegicus", "homo_sapiens",
                "macaca_mulatta", "danio_rerio"} <= ids

    def test_all_species_have_ncbitaxon_id(self):
        from pathlib import Path

        import yaml
        raw = yaml.safe_load(Path("data/ontology/species_taxonomy.yaml").read_text())
        for s in raw["species"]:
            assert s["ncbitaxon_id"].startswith("NCBITaxon:"), (
                f"Species {s['id']!r} has malformed NCBITaxon ID: {s['ncbitaxon_id']!r}"
            )
