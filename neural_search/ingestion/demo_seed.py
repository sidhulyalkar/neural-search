"""Small deterministic seed dataset for local demos and CLI smoke tests."""

from __future__ import annotations

import json

from neural_search.extraction import extract_dataset_labels


def build_demo_seed() -> list[dict]:
    datasets = [
        {
            "id": "DEMO",
            "source": "demo",
            "source_id": "DEMO",
            "title": "Demo reversal learning NWB dataset with reward omission",
            "description": (
                "Mouse OFC Neuropixels recordings during probabilistic reversal "
                "learning with choice, reward, omission, and trial outcomes."
            ),
            "url": "https://example.org/demo",
            "license": "CC-BY-4.0",
            "species": ["mouse"],
            "modalities": ["neuropixels"],
            "brain_regions": ["OFC"],
            "tasks": ["reversal_learning"],
            "behaviors": ["choice", "reward", "omission"],
            "data_standards": ["NWB"],
            "has_behavior": True,
            "has_trials": True,
            "has_raw_data": True,
            "has_processed_data": True,
            "metadata_json": {
                "license": "CC-BY-4.0",
                "trial_columns": ["choice", "reward", "omission", "reversal_point"],
                "has_processed_data": True,
            },
        },
        {
            "id": "DEMO_GONOGO",
            "source": "demo",
            "source_id": "DEMO_GONOGO",
            "title": "Demo Go/NoGo lick event calcium imaging dataset",
            "description": (
                "Mouse mPFC calcium imaging during a Go/NoGo response inhibition "
                "task with lick, hit, false alarm, and reward events."
            ),
            "url": "https://example.org/demo-gonogo",
            "license": "CC0-1.0",
            "species": ["mouse"],
            "modalities": ["calcium_imaging"],
            "brain_regions": ["mPFC"],
            "tasks": ["go_nogo"],
            "behaviors": ["lick", "reward"],
            "data_standards": ["NWB"],
            "has_behavior": True,
            "has_trials": True,
            "has_raw_data": True,
            "has_processed_data": False,
            "metadata_json": {"license": "CC0-1.0", "trial_columns": ["hit", "false_alarm"]},
        },
    ]
    assets = [
        [
            {
                "id": "DEMO",
                "dataset_id": "DEMO",
                "path": "data/seed/demo_reversal_learning.nwb",
                "asset_type": "nwb",
                "file_format": "nwb",
                "modality": "neuropixels",
            }
        ],
        [
            {
                "id": "DEMO_GONOGO_ASSET",
                "dataset_id": "DEMO_GONOGO",
                "path": "data/seed/demo_gonogo.nwb",
                "asset_type": "nwb",
                "file_format": "nwb",
                "modality": "calcium_imaging",
            }
        ],
    ]
    papers = [
        [
            {
                "id": "PAPER_DEMO",
                "title": "Orbitofrontal activity during reversal learning",
                "abstract": (
                    "A linked abstract describing reward omission, choice outcomes, "
                    "and contingency reversal in mice."
                ),
            }
        ],
        [],
    ]
    records: list[dict] = []
    for dataset, dataset_assets, linked_papers in zip(datasets, assets, papers, strict=True):
        extraction = extract_dataset_labels(
            title=dataset["title"],
            description=dataset["description"],
            file_paths=[asset["path"] for asset in dataset_assets],
            source_metadata=dataset,
            linked_paper_abstracts=[paper["abstract"] for paper in linked_papers],
        )
        records.append(
            {
                "dataset": dataset,
                "assets": dataset_assets,
                "papers": linked_papers,
                "extraction": extraction,
            }
        )
    return records


def main() -> int:
    payload = []
    for record in build_demo_seed():
        item = dict(record)
        item["extraction"] = record["extraction"].model_dump(mode="json")
        payload.append(item)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
