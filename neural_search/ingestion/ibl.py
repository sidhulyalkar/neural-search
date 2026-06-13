"""International Brain Laboratory (IBL) data adapter.

Combines curated Brain Wide Map release records with per-session records
from subject session parquet tables in the public IBL S3 bucket.

No credentials required — all data is publicly accessible.
"""
from __future__ import annotations

import hashlib
import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx
import pandas as pd
from defusedxml import ElementTree as ET

from neural_search.ingestion.registry import register

logger = logging.getLogger(__name__)

IBL_S3_BUCKET = "https://ibl-brain-wide-map-public.s3.amazonaws.com"
IBL_FLATIRONINSTITUTE = "https://ibl.flatironinstitute.org/public"

# Ordered by contribution size; we enumerate these for session-level records
IBL_LABS = [
    "angelakilab", "churchlandlab", "cortexlab", "danlab", "hausserlab",
    "hoferlab", "mainenlab", "mrsicflogellab", "steinmetzlab", "wittenlab",
    "zadorlab", "churchlandlab_ucla",
]

# Curated project-level records — one per major IBL data release
_IBL_PROJECTS: list[dict[str, Any]] = [
    {
        "id": "ibl_brain_wide_map_2022",
        "title": "IBL Brain Wide Map 2022 Q4 Release",
        "description": (
            "International Brain Laboratory Brain Wide Map first public data release (2022 Q4). "
            "Neuropixels recordings across the mouse brain during the IBL decision-making task "
            "(biased choice world). Includes spike times, spike amplitudes, cluster labels, "
            "trial-aligned behavioral data, and brain region assignments via IBL alignment GUI. "
            "Data from 12 collaborating labs, ~250 sessions, covering frontal cortex, parietal "
            "cortex, hippocampus, thalamus, striatum, and hindbrain."
        ),
        "url": "https://doi.org/10.1038/s41586-023-06490-7",
        "s3_prefix": "aggregates/2022_Q4_IBL_et_al_BWM/",
        "modalities": ["extracellular_ephys", "neuropixels", "behavior"],
        "brain_regions": ["prefrontal_cortex", "parietal_cortex", "hippocampus", "thalamus", "striatum", "brainstem"],
        "tasks": ["decision_making", "two_alternative_forced_choice"],
        "behaviors": ["licking", "wheel_movement", "body_camera"],
        "data_standards": ["ALF", "NWB"],
    },
    {
        "id": "ibl_brain_wide_map_2023",
        "title": "IBL Brain Wide Map 2023 Q4 Release",
        "description": (
            "International Brain Laboratory Brain Wide Map second public data release (2023 Q4). "
            "Extended Neuropixels dataset with improved spike sorting, additional sessions, and "
            "updated brain region assignments. Covers the full mouse brain during the IBL "
            "perceptual decision-making task. ~700 sessions from 12 labs. "
            "Includes multi-region population dynamics data for decoding analyses."
        ),
        "url": "https://ibl-brain-wide-map-public.s3.amazonaws.com/aggregates/2023_Q4_IBL_et_al_BWM/clusters.pqt",
        "s3_prefix": "aggregates/2023_Q4_IBL_et_al_BWM_2/",
        "modalities": ["extracellular_ephys", "neuropixels", "behavior"],
        "brain_regions": ["prefrontal_cortex", "parietal_cortex", "hippocampus", "thalamus", "striatum", "brainstem"],
        "tasks": ["decision_making", "two_alternative_forced_choice"],
        "behaviors": ["licking", "wheel_movement", "pupil_tracking", "body_camera"],
        "data_standards": ["ALF", "NWB"],
    },
    {
        "id": "ibl_brain_wide_map_2024",
        "title": "IBL Brain Wide Map 2024 Q2 Release",
        "description": (
            "International Brain Laboratory Brain Wide Map third public data release (2024 Q2). "
            "Largest IBL release: ~1,000 sessions, full brain coverage, refined histology "
            "and probe track reconstruction. Added mesoscale calcium imaging subset and fiber "
            "photometry sessions. Includes motion energy, pose estimation (SLEAP), and "
            "passive replay sessions for comparing active vs passive stimulus responses."
        ),
        "url": "https://ibl-brain-wide-map-public.s3.amazonaws.com/aggregates/2024_Q2_IBL_et_al_BWM/clusters.pqt",
        "s3_prefix": "aggregates/2024_Q2_IBL_et_al_BWM/",
        "modalities": ["extracellular_ephys", "neuropixels", "calcium_imaging", "fiber_photometry", "behavior"],
        "brain_regions": ["prefrontal_cortex", "parietal_cortex", "hippocampus", "thalamus", "striatum", "brainstem"],
        "tasks": ["decision_making", "two_alternative_forced_choice", "passive_viewing"],
        "behaviors": ["licking", "wheel_movement", "pupil_tracking", "body_camera", "pose_estimation"],
        "data_standards": ["ALF", "NWB"],
    },
    {
        "id": "ibl_brain_wide_map_2026",
        "title": "IBL Brain Wide Map 2026 Q2 Release",
        "description": (
            "International Brain Laboratory Brain Wide Map latest public data release (2026 Q2). "
            "Includes cluster waveforms, autocorrelogram data, and updated cluster quality metrics. "
            "Covers the full multi-lab Neuropixels dataset with standardized preprocessing "
            "pipeline and cross-lab quality control."
        ),
        "url": "https://ibl-brain-wide-map-public.s3.amazonaws.com/aggregates/2026_Q2_IBL_et_al_BWM/clusters.pqt",
        "s3_prefix": "aggregates/2026_Q2_IBL_et_al_BWM/",
        "modalities": ["extracellular_ephys", "neuropixels", "behavior"],
        "brain_regions": ["prefrontal_cortex", "parietal_cortex", "hippocampus", "thalamus", "striatum", "brainstem"],
        "tasks": ["decision_making", "two_alternative_forced_choice"],
        "behaviors": ["licking", "wheel_movement", "pupil_tracking"],
        "data_standards": ["ALF", "NWB"],
    },
    {
        "id": "ibl_repeated_site",
        "title": "IBL Repeated Site Dataset — Reproducible Ephys",
        "description": (
            "IBL repeated-site dataset: standardized Neuropixels probe insertion at an identical "
            "brain location across 10 labs, 59 mice. Designed to test reproducibility of in-vivo "
            "electrophysiology across labs, experimenters, and equipment. Probe traverses motor "
            "cortex (MOs/MOp), retrosplenial cortex, thalamus (VPM/VPL/LP), and hippocampus (CA1/DG). "
            "Data includes spike sorting output, LFP, and behavioral readouts."
        ),
        "url": "https://figshare.com/articles/dataset/Data_release_-_Reproducibility_of_in-vivo_electrophysiological_measurements_in_mice/16624551",
        "s3_prefix": None,
        "modalities": ["extracellular_ephys", "neuropixels", "lfp", "behavior"],
        "brain_regions": ["motor_cortex", "retrosplenial_cortex", "thalamus", "hippocampus"],
        "tasks": ["passive_viewing", "spontaneous_activity"],
        "behaviors": ["pupil_tracking", "body_camera"],
        "data_standards": ["ALF", "NWB"],
    },
    {
        "id": "ibl_np1_np2_kim",
        "title": "IBL NP1 vs NP2 Neuropixels Comparison (Kim et al.)",
        "description": (
            "IBL dataset comparing Neuropixels 1.0 and 2.0 probes in the same brain regions. "
            "Recordings from visual cortex, hippocampus, and brainstem. Provides matched NP1/NP2 "
            "sessions for benchmarking spike sorting algorithms and probe comparisons."
        ),
        "url": "https://ibl-brain-wide-map-public.s3.amazonaws.com/aggregates/np1_np2_kim_recordings/",
        "s3_prefix": "aggregates/np1_np2_kim_recordings/",
        "modalities": ["extracellular_ephys", "neuropixels"],
        "brain_regions": ["visual_cortex", "hippocampus", "brainstem"],
        "tasks": ["passive_viewing"],
        "behaviors": [],
        "data_standards": ["ALF"],
    },
]


def _source_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _normalize_project(project: dict[str, Any]) -> dict[str, Any]:
    pid = project["id"]
    return {
        "source": "ibl",
        "source_id": pid,
        "source_type": "canonical_dataset",
        "record_type": "dataset_collection",
        "title": project["title"],
        "description": project["description"],
        "url": project["url"],
        "access_url": (
            f"{IBL_S3_BUCKET}/{project['s3_prefix']}"
            if project.get("s3_prefix")
            else project["url"]
        ),
        "license": "CC-BY 4.0",
        "species": ["mouse"],
        "modalities": project["modalities"],
        "brain_regions": project["brain_regions"],
        "tasks": project["tasks"],
        "behaviors": project["behaviors"],
        "data_standards": project["data_standards"],
        "has_behavior": True,
        "has_trials": True,
        "has_raw_data": True,
        "has_processed_data": True,
        "analysis_affordances": [
            "spike_sorting",
            "population_decoding",
            "cross_lab_reproducibility",
            "brain_wide_mapping",
            "decision_variable_tracking",
        ],
        "metadata_json": {
            "raw_source": "ibl",
            "record_subtype": "bwm_release",
            "s3_prefix": project.get("s3_prefix"),
        },
    }


_PROTOCOL_TO_TASKS: dict[str, list[str]] = {
    "biasedChoiceWorld": ["decision_making", "two_alternative_forced_choice"],
    "ephysChoiceWorld": ["decision_making", "two_alternative_forced_choice"],
    "trainingChoiceWorld": ["decision_making", "training"],
    "passiveChoiceWorld": ["passive_viewing"],
    "habituation": ["habituation"],
}


def _protocol_tasks(protocol: str) -> list[str]:
    for key, tasks in _PROTOCOL_TO_TASKS.items():
        if key in protocol:
            return tasks
    return ["decision_making"]


def _list_subject_session_tables(client: httpx.Client) -> list[str]:
    """Return all _ibl_subjectSessions.table.*.pqt keys in S3."""
    keys: list[str] = []
    marker = ""
    while True:
        params: dict[str, Any] = {"prefix": "aggregates/Subjects/", "max-keys": "1000"}
        if marker:
            params["marker"] = marker
        try:
            resp = client.get(IBL_S3_BUCKET + "/", params=params, timeout=15)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("IBL S3 list error: %s", exc)
            break
        root = ET.fromstring(resp.content)
        ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
        batch = [
            k.find(f"{ns}Key").text
            for k in root.findall(f"{ns}Contents")
            if "_ibl_subjectSessions.table." in (k.find(f"{ns}Key").text or "")
        ]
        keys.extend(batch)
        if root.find(f"{ns}IsTruncated").text == "false":
            break
        all_keys_batch = [k.find(f"{ns}Key").text for k in root.findall(f"{ns}Contents")]
        marker = all_keys_batch[-1] if all_keys_batch else ""
    return keys


def _fetch_session_table(key: str) -> pd.DataFrame | None:
    """Download and parse one subject session parquet table."""
    try:
        resp = httpx.get(f"{IBL_S3_BUCKET}/{key}", timeout=20, follow_redirects=True)
        if resp.status_code == 200:
            return pd.read_parquet(io.BytesIO(resp.content))
    except Exception as exc:
        logger.warning("IBL parquet fetch error for %s: %s", key, exc)
    return None


def _normalize_session_row(row: Any, session_uuid: str) -> dict[str, Any]:
    """Build a corpus record for one IBL session row from the sessions table."""
    lab = str(row.get("lab", ""))
    subject = str(row.get("subject", ""))
    date = str(row.get("date", ""))
    session_num = str(int(row.get("number", 1)))
    protocol = str(row.get("task_protocol", ""))
    project = str(row.get("projects", ""))

    is_ephys = "biasedChoiceWorld" in protocol or "ephysChoiceWorld" in protocol
    title = (
        f"IBL Brain Wide Map: {lab}/{subject} {date}"
        + (" — Neuropixels recording" if is_ephys else "")
    )
    description = (
        f"International Brain Laboratory {project} session. "
        f"Lab: {lab}. Subject: {subject}. Date: {date}. Session {session_num}. "
        f"Task protocol: {protocol}. "
        + ("Neuropixels multi-region ephys + behavior (biased choice world perceptual decision task). "
           if is_ephys else "Behavioral training session. ")
        + "Data in ALF format accessible via ONE API (session UUID: "
        + f"{session_uuid})."
    )
    return {
        "source": "ibl",
        "source_id": f"session_{session_uuid[:16]}",
        "source_type": "canonical_dataset",
        "record_type": "dataset",
        "title": title,
        "description": description,
        "url": f"https://ibl-brain-wide-map-public.s3.amazonaws.com/aggregates/Subjects/{lab}/{subject}/{date}/{session_num}/",
        "identifier": session_uuid,
        "license": "CC-BY 4.0",
        "species": ["mouse"],
        "modalities": (
            ["extracellular_ephys", "neuropixels", "behavior"]
            if is_ephys else ["behavior"]
        ),
        "brain_regions": ["prefrontal_cortex", "parietal_cortex", "hippocampus", "thalamus", "striatum", "brainstem"],
        "tasks": _protocol_tasks(protocol),
        "behaviors": ["licking", "wheel_movement", "pupil_tracking"],
        "data_standards": ["ALF"],
        "has_behavior": True,
        "has_trials": True,
        "has_raw_data": True,
        "has_processed_data": True,
        "analysis_affordances": [
            "spike_sorting",
            "population_decoding",
            "decision_variable_tracking",
            "cross_lab_reproducibility",
        ] if is_ephys else ["behavioral_decoding"],
        "metadata_json": {
            "raw_source": "ibl",
            "lab": lab,
            "subject": subject,
            "date": date,
            "session_number": int(session_num),
            "task_protocol": protocol,
            "project": project,
            "session_uuid": session_uuid,
            "record_subtype": "session",
        },
    }


def fetch_ibl_sessions(limit: int = 200) -> list[dict[str, Any]]:
    """Build IBL session records by reading subject session parquet tables from S3."""
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        table_keys = _list_subject_session_tables(client)

    logger.info("IBL: found %d subject session tables", len(table_keys))

    # Fetch all tables concurrently (they're small ~few KB each)
    all_dfs: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(_fetch_session_table, k): k for k in table_keys}
        for future in as_completed(futures):
            df = future.result()
            if df is not None and not df.empty:
                all_dfs.append(df)

    if not all_dfs:
        logger.warning("IBL: no session tables fetched")
        return []

    combined = pd.concat(all_dfs)
    # Filter to BWM ephys sessions (exclude pure training)
    bwm_mask = combined["projects"].str.contains("ibl_neuropixel_brainwide", na=False)
    ephys_mask = combined["task_protocol"].str.contains("biasedChoiceWorld|ephysChoiceWorld", na=False)
    filtered = combined[bwm_mask & ephys_mask]

    # Sample evenly across labs
    labs = filtered["lab"].unique()
    per_lab = max(1, limit // len(labs)) if len(labs) > 0 else limit
    sampled_parts = []
    for lab in sorted(labs):
        lab_rows = filtered[filtered["lab"] == lab]
        sampled_parts.append(lab_rows.head(per_lab))
    sampled = pd.concat(sampled_parts).head(limit)

    records: list[dict[str, Any]] = []
    for session_uuid, row in sampled.iterrows():
        try:
            records.append(_normalize_session_row(row, str(session_uuid)))
        except Exception as exc:
            logger.warning("IBL session normalize error for %s: %s", session_uuid, exc)

    logger.info("IBL sessions: %d records from %d tables (%d BWM ephys total)", len(records), len(all_dfs), len(filtered))
    return records


@register("ibl")
def fetch_ibl_records(limit: int = 200) -> list[dict[str, Any]]:
    """Registry adapter: curated IBL releases + per-session S3 enumeration."""
    records: list[dict[str, Any]] = [_normalize_project(p) for p in _IBL_PROJECTS]
    session_limit = max(0, limit - len(records))
    if session_limit > 0:
        sessions = fetch_ibl_sessions(limit=session_limit)
        records.extend(sessions)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for rec in records:
        sid = str(rec.get("source_id", ""))
        if sid and sid not in seen:
            seen.add(sid)
            deduped.append(rec)
    logger.info("IBL: %d total records (%d curated + %d sessions)", len(deduped), len(_IBL_PROJECTS), len(deduped) - len(_IBL_PROJECTS))
    return deduped[:limit]
