from scripts.eval.build_corpus_from_packets import build_corpus_records


def test_dedup_and_field_mapping():
    packets = [
        {"dataset_id": "dandi:000003", "title": "Hippocampus ephys",
         "description": "theta", "dataset_modalities": ["extracellular_ephys"],
         "dataset_species": ["mouse"], "dataset_tasks": ["spatial_navigation"],
         "dataset_brain_regions": ["hippocampus"], "data_standards": ["nwb"],
         "source_archive": "dandi", "query_id": "q_1"},
        {"dataset_id": "dandi:000003", "title": "Hippocampus ephys",
         "description": "theta", "dataset_modalities": ["extracellular_ephys"],
         "dataset_species": ["mouse"], "dataset_tasks": ["spatial_navigation"],
         "dataset_brain_regions": ["hippocampus"], "data_standards": ["nwb"],
         "source_archive": "dandi", "query_id": "q_2"},
    ]
    records = build_corpus_records(packets)
    assert len(records) == 1
    r = records[0]
    assert r["dataset_id"] == "dandi:000003"
    assert r["source"] == "dandi"
    assert r["source_id"] == "000003"
    assert r["modalities"] == ["extracellular_ephys"]
    assert r["species"] == ["mouse"]
    assert r["tasks"] == ["spatial_navigation"]
    assert r["brain_regions"] == ["hippocampus"]
    assert r["data_standards"] == ["nwb"]
    assert r["title"] == "Hippocampus ephys"
    assert r["description"] == "theta"
