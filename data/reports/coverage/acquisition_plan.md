# Corpus Acquisition Plan

Generated from DuckDB coverage ledger (7,176 datasets).
Items ranked by estimated coverage impact.

**Total action items:** 45

## Low-Coverage Sources

### [P0] neurovault
**Impact:** 0.874 | **Est. new datasets:** 699

neurovault has 2,000 datasets but only 13% have brain region labels. Structured metadata enrichment or additional adapter fields could unlock 1,748 more.

**Action:** NeuroVault whole-brain fMRI maps rarely specify recording regions — by design. Region coverage here represents ROI analyses only. Parse `contrast_definition` field for region mentions instead.

### [P1] gin
**Impact:** 0.592 | **Est. new datasets:** 118

gin has 408 datasets but only 28% have brain region labels. Structured metadata enrichment or additional adapter fields could unlock 295 more.

**Action:** GIN repositories often link to publications. Run CrossRef DOI resolver on linked papers to extract region mentions. Estimate +100-150 region labels.

### [P2] openneuro
**Impact:** 0.478 | **Est. new datasets:** 95

openneuro has 300 datasets but only 20% have brain region labels. Structured metadata enrichment or additional adapter fields could unlock 239 more.

**Action:** OpenNeuro BIDS datasets include `participants.tsv` (species) and electrode location files. Run NWB/BIDS electrode extractor on the 239 uncovered datasets.

### [P2] brain_image_library
**Impact:** 0.396 | **Est. new datasets:** 79

brain_image_library has 200 datasets but only 1% have brain region labels. Structured metadata enrichment or additional adapter fields could unlock 198 more.

**Action:** BIL datasets use a structured JSON manifest. Parse `general/subject/species` and `general/technique` fields to extract brain_regions and modalities. Expected yield: ~180 additional structured records.

## Dark Region × Modality Pairs

### [P0] retina × fmri
**Impact:** 1.000 | **Est. new datasets:** 17

No datasets combine retina with fmri. 87 retina datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for retina+fmri co-recorded datasets.

### [P0] septum × fmri
**Impact:** 1.000 | **Est. new datasets:** 2

No datasets combine septum with fmri. 11 septum datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for septum+fmri co-recorded datasets.

### [P0] medial_entorhinal_cortex × fmri
**Impact:** 1.000 | **Est. new datasets:** 2

No datasets combine medial_entorhinal_cortex with fmri. 11 medial_entorhinal_cortex datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for medial_entorhinal_cortex+fmri co-recorded datasets.

### [P0] barrel_cortex × fmri
**Impact:** 1.000 | **Est. new datasets:** 1

No datasets combine barrel_cortex with fmri. 9 barrel_cortex datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for barrel_cortex+fmri co-recorded datasets.

### [P0] piriform_cortex × fmri
**Impact:** 1.000 | **Est. new datasets:** 1

No datasets combine piriform_cortex with fmri. 7 piriform_cortex datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for piriform_cortex+fmri co-recorded datasets.

### [P0] lateral_geniculate_nucleus × fmri
**Impact:** 1.000 | **Est. new datasets:** 1

No datasets combine lateral_geniculate_nucleus with fmri. 6 lateral_geniculate_nucleus datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for lateral_geniculate_nucleus+fmri co-recorded datasets.

### [P0] dorsolateral_striatum × fmri
**Impact:** 1.000 | **Est. new datasets:** 1

No datasets combine dorsolateral_striatum with fmri. 6 dorsolateral_striatum datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for dorsolateral_striatum+fmri co-recorded datasets.

### [P0] dorsomedial_striatum × fmri
**Impact:** 1.000 | **Est. new datasets:** 1

No datasets combine dorsomedial_striatum with fmri. 5 dorsomedial_striatum datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for dorsomedial_striatum+fmri co-recorded datasets.

### [P0] dorsal_raphe × fmri
**Impact:** 1.000 | **Est. new datasets:** 1

No datasets combine dorsal_raphe with fmri. 4 dorsal_raphe datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for dorsal_raphe+fmri co-recorded datasets.

### [P0] arcuate_nucleus × fmri
**Impact:** 1.000 | **Est. new datasets:** 1

No datasets combine arcuate_nucleus with fmri. 4 arcuate_nucleus datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for arcuate_nucleus+fmri co-recorded datasets.

### [P0] lateral_entorhinal_cortex × fmri
**Impact:** 1.000 | **Est. new datasets:** 1

No datasets combine lateral_entorhinal_cortex with fmri. 3 lateral_entorhinal_cortex datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for lateral_entorhinal_cortex+fmri co-recorded datasets.

### [P0] cerebellar_cortex × fmri
**Impact:** 1.000 | **Est. new datasets:** 1

No datasets combine cerebellar_cortex with fmri. 3 cerebellar_cortex datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for cerebellar_cortex+fmri co-recorded datasets.

### [P0] somatosensory_area_2 × fmri
**Impact:** 1.000 | **Est. new datasets:** 1

No datasets combine somatosensory_area_2 with fmri. 3 somatosensory_area_2 datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for somatosensory_area_2+fmri co-recorded datasets.

### [P0] lateral_septum × fmri
**Impact:** 1.000 | **Est. new datasets:** 1

No datasets combine lateral_septum with fmri. 2 lateral_septum datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for lateral_septum+fmri co-recorded datasets.

### [P0] anterior_thalamic_nuclei × fmri
**Impact:** 1.000 | **Est. new datasets:** 1

No datasets combine anterior_thalamic_nuclei with fmri. 2 anterior_thalamic_nuclei datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for anterior_thalamic_nuclei+fmri co-recorded datasets.

### [P0] mst × fmri
**Impact:** 1.000 | **Est. new datasets:** 1

No datasets combine mst with fmri. 1 mst datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for mst+fmri co-recorded datasets.

### [P0] cervical_spinal_cord × fmri
**Impact:** 1.000 | **Est. new datasets:** 1

No datasets combine cervical_spinal_cord with fmri. 1 cervical_spinal_cord datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for cervical_spinal_cord+fmri co-recorded datasets.

### [P0] thoracic_spinal_cord × fmri
**Impact:** 1.000 | **Est. new datasets:** 1

No datasets combine thoracic_spinal_cord with fmri. 1 thoracic_spinal_cord datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for thoracic_spinal_cord+fmri co-recorded datasets.

### [P0] seizure_focus × fmri
**Impact:** 1.000 | **Est. new datasets:** 1

No datasets combine seizure_focus with fmri. 1 seizure_focus datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for seizure_focus+fmri co-recorded datasets.

### [P0] frontal_eye_field × fmri
**Impact:** 1.000 | **Est. new datasets:** 1

No datasets combine frontal_eye_field with fmri. 1 frontal_eye_field datasets and 2473 fmri datasets exist separately — cross-species or protocol bridging could close this gap.

**Action:** Search OpenNeuro, NeuroVault for frontal_eye_field+fmri co-recorded datasets.

## Underrepresented Species

### [P2]  × 
**Impact:** 0.468 | **Est. new datasets:** 12

other has only 6 datasets. Targeted crawl of species-specific repositories could expand coverage.

**Sources:** dandi, zenodo, figshare

### [P2]  × 
**Impact:** 0.444 | **Est. new datasets:** 18

c_elegans has only 9 datasets. Targeted crawl of species-specific repositories could expand coverage.

**Sources:** dandi, zenodo, figshare

### [P2]  × 
**Impact:** 0.432 | **Est. new datasets:** 22

marmoset has only 11 datasets. Targeted crawl of species-specific repositories could expand coverage.

**Sources:** dandi, gin, zenodo

### [P2]  × 
**Impact:** 0.432 | **Est. new datasets:** 22

cat has only 11 datasets. Targeted crawl of species-specific repositories could expand coverage.

**Sources:** dandi, zenodo, figshare

### [P2]  × 
**Impact:** 0.344 | **Est. new datasets:** 86

drosophila has only 43 datasets. Targeted crawl of species-specific repositories could expand coverage.

**Sources:** dandi, zenodo, figshare
