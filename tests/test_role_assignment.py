"""Tests for dataset role assignment."""
from neural_search.retrieval.role_assignment import (
    DatasetRole,
    assign_role,
    RoleAssignment,
)


def test_anchor_role():
    datasets = [
        {"dataset_id": "ds1", "usefulness_score": 0.9, "sub_query_matches": 4,
         "modalities": ["neuropixels"], "tasks": ["working_memory", "decision_making"],
         "species": ["mouse"], "brain_regions": ["prefrontal_cortex"]},
    ]
    role = assign_role(datasets[0], datasets, anchor_id=None)
    assert role.role == DatasetRole.ANCHOR


def test_replication_role():
    anchor = {"dataset_id": "anchor", "usefulness_score": 0.9, "sub_query_matches": 4,
              "tasks": ["working_memory"], "species": ["mouse"], "modalities": ["neuropixels"]}
    candidate = {"dataset_id": "rep", "usefulness_score": 0.7, "sub_query_matches": 2,
                 "tasks": ["working_memory"], "species": ["mouse"], "modalities": ["calcium_imaging"]}
    role = assign_role(candidate, [anchor, candidate], anchor_id="anchor")
    assert role.role == DatasetRole.REPLICATION


def test_cross_species_role():
    anchor = {"dataset_id": "anchor", "tasks": ["reversal_learning"], "species": ["mouse"],
              "modalities": ["neuropixels"], "usefulness_score": 0.9, "sub_query_matches": 4}
    candidate = {"dataset_id": "human_ds", "tasks": ["reversal_learning"], "species": ["human"],
                 "modalities": ["fmri"], "usefulness_score": 0.7, "sub_query_matches": 2}
    role = assign_role(candidate, [anchor, candidate], anchor_id="anchor")
    assert role.role == DatasetRole.CROSS_SPECIES_COMPARATOR


def test_no_role_excluded():
    anchor = {"dataset_id": "anchor", "tasks": ["working_memory"], "species": ["mouse"],
              "modalities": ["neuropixels"], "usefulness_score": 0.9, "sub_query_matches": 4}
    candidate = {"dataset_id": "unrelated", "tasks": [], "species": [],
                 "modalities": [], "usefulness_score": 0.1, "sub_query_matches": 0,
                 "affordances": []}
    role = assign_role(candidate, [anchor, candidate], anchor_id="anchor")
    assert role.role == DatasetRole.UNASSIGNABLE
