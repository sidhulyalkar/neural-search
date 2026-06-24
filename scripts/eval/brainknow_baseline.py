"""BrainKnow-style same-sentence co-occurrence baseline.

Builds a lightweight concept co-occurrence graph from literature findings
(or OpenAlex abstracts if available), following the method described in:
  Wang et al. (2024) BrainKnow: arXiv:2403.04346

Differences from BrainKnow:
  - BrainKnow: 1.8M PubMed papers, 37K concepts, 3.6M edges
  - This baseline: ~80K tier-1 findings from findings_tier1_ollama.jsonl
    (or findings_tier1_normalized.jsonl), same-sentence concept pairs

Output:
  - reports/eval/brainknow_baseline_graph.json
  - reports/eval/brainknow_baseline_summary.md

Usage:
    python scripts/eval/brainknow_baseline.py
    python scripts/eval/brainknow_baseline.py --max-records 5000   # fast smoke test
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Use the normalized findings which have structured region/task/modality fields
FINDINGS_NORMALIZED = ROOT / "artifacts/literature/findings_tier1_normalized.jsonl"
FINDINGS_OLLAMA = ROOT / "artifacts/literature/findings_tier1_ollama.jsonl"

GRAPH_OUT = ROOT / "reports/eval/brainknow_baseline_graph.json"
SUMMARY_MD = ROOT / "reports/eval/brainknow_baseline_summary.md"

# ---------------------------------------------------------------------------
# Expanded concept vocabulary with synonym → canonical mapping.
# Each tuple is (regex_pattern, canonical_label, concept_type).
# Patterns are matched case-insensitively against the full finding text.
# Multiple aliases can map to the same canonical label.
# ---------------------------------------------------------------------------

CONCEPT_VOCAB: list[tuple[str, str, str]] = [
    # -----------------------------------------------------------------------
    # BRAIN REGIONS — cortical
    # -----------------------------------------------------------------------
    (r"\bhippocampus\b|\bhippocampal\b|\bHPC\b", "hippocampus", "region"),
    (r"\bprefrontal\s+cortex\b|\bPFC\b|\bdlPFC\b|\bvmPFC\b|\bdmPFC\b|\bmPFC\b", "prefrontal cortex", "region"),
    (r"\banterior\s+cingulate\b|\bACC\b|\bdACC\b|\bpACC\b", "anterior cingulate", "region"),
    (r"\borbitofrontal\s+cortex\b|\bOFC\b|\bvmPFC\b", "orbitofrontal cortex", "region"),
    (r"\bmotor\s+cortex\b|\bM1\b|\bprimary\s+motor\b", "motor cortex", "region"),
    (r"\bsomatosensory\s+cortex\b|\bS1\b|\bS2\b|\bprimary\s+somatosensory\b", "somatosensory cortex", "region"),
    (r"\bvisual\s+cortex\b|\bV1\b|\bV2\b|\bMT\b|\bstriate\s+cortex\b", "visual cortex", "region"),
    (r"\bauditory\s+cortex\b|\bA1\b|\bprimary\s+auditory\b", "auditory cortex", "region"),
    (r"\binsula\b|\binsular\s+cortex\b", "insula", "region"),
    (r"\bparietal\s+cortex\b|\bparietal\s+lobe\b|\bLIP\b|\bAIP\b|\bVIP\b|\bIPL\b", "parietal cortex", "region"),
    (r"\btemporal\s+cortex\b|\btemporal\s+lobe\b|\bMTL\b|\bITC\b|\bSTC\b", "temporal cortex", "region"),
    (r"\boccipital\s+cortex\b|\boccipital\s+lobe\b|\boccipital\b", "occipital cortex", "region"),
    (r"\bparahippocampal\b|\bPHC\b|\bparahippocampal\s+gyrus\b", "parahippocampal", "region"),
    (r"\bentorhinal\s+cortex\b|\bEC\b|\bMEC\b|\bLEC\b", "entorhinal cortex", "region"),
    (r"\bperirhinal\s+cortex\b|\bPRC\b", "perirhinal cortex", "region"),
    (r"\bposterior\s+cingulate\b|\bPCC\b|\bretrosp\w+\s+cortex\b|\bRSC\b", "posterior cingulate", "region"),
    (r"\bsupplementary\s+motor\b|\bSMA\b|\bpre-SMA\b", "supplementary motor area", "region"),
    (r"\bpremotor\s+cortex\b|\bPMC\b|\bPM\b", "premotor cortex", "region"),
    (r"\bfrontal\s+eye\s+field\b|\bFEF\b", "frontal eye field", "region"),
    (r"\bdefault\s+mode\b|\bDMN\b", "default mode network", "region"),

    # -----------------------------------------------------------------------
    # BRAIN REGIONS — hippocampal subfields
    # -----------------------------------------------------------------------
    (r"\bCA1\b|\bCA\s*1\b", "CA1", "region"),
    (r"\bCA2\b|\bCA\s*2\b", "CA2", "region"),
    (r"\bCA3\b|\bCA\s*3\b", "CA3", "region"),
    (r"\bdentate\s+gyrus\b|\bDG\b", "dentate gyrus", "region"),
    (r"\bsubiculum\b|\bsub\b", "subiculum", "region"),

    # -----------------------------------------------------------------------
    # BRAIN REGIONS — subcortical
    # -----------------------------------------------------------------------
    (r"\bamygdala\b|\bBLA\b|\bbasolateral\s+amygdala\b|\bCeA\b|\bcentral\s+amygdala\b", "amygdala", "region"),
    (r"\bstriatum\b|\bcaudate\b|\bputamen\b|\bdorsal\s+striatum\b", "striatum", "region"),
    (r"\bnucleus\s+accumbens\b|\bNAc\b|\bNAcc\b|\bventral\s+striatum\b", "nucleus accumbens", "region"),
    (r"\bbasal\s+ganglia\b|\bBG\b", "basal ganglia", "region"),
    (r"\bglobus\s+pallidus\b|\bGP\b|\bGPe\b|\bGPi\b", "globus pallidus", "region"),
    (r"\bsubthalamic\s+nucleus\b|\bSTN\b", "subthalamic nucleus", "region"),
    (r"\bthalamus\b|\bthalamic\b|\bMD\b|\bLGN\b|\bMGN\b|\bVPL\b|\bPVT\b", "thalamus", "region"),
    (r"\bhypothalamus\b|\bhypothalamic\b|\bLH\b|\bDMH\b|\bVMH\b|\bARC\b", "hypothalamus", "region"),
    (r"\bsubstantia\s+nigra\b|\bSNc\b|\bSNr\b|\bSN\b", "substantia nigra", "region"),
    (r"\bventral\s+tegmental\s+area\b|\bVTA\b", "VTA", "region"),
    (r"\bdorsal\s+raphe\b|\braphe\b|\bDRN\b|\bMRN\b", "raphe nucleus", "region"),
    (r"\blocus\s+coeruleus\b|\bLC\b", "locus coeruleus", "region"),
    (r"\bhabenula\b|\bLHb\b|\bMHb\b", "habenula", "region"),
    (r"\bseptal\s+nucleus\b|\bseptum\b|\bmedial\s+septum\b|\bMS\b", "medial septum", "region"),
    (r"\bbed\s+nucleus\b|\bBNST\b", "BNST", "region"),
    (r"\bclaustrum\b", "claustrum", "region"),
    (r"\bbrainstem\b|\bpons\b|\bmedulla\b|\bmidbrain\b", "brainstem", "region"),
    (r"\bcerebellum\b|\bcerebellar\b|\bpurkinje\s+cell\b", "cerebellum", "region"),
    (r"\bolfactory\s+bulb\b|\bOB\b|\bMOB\b", "olfactory bulb", "region"),

    # -----------------------------------------------------------------------
    # CELL TYPES
    # -----------------------------------------------------------------------
    (r"\bpyramidal\s+cell\b|\bpyramidal\s+neuron\b|\bpyramidal\b", "pyramidal cell", "cell_type"),
    (r"\bparvalbumin\b|\bPV\s*\+|\bPV\s+interneuron\b|\bfast.spiking\b", "parvalbumin interneuron", "cell_type"),
    (r"\bsomatostatin\b|\bSST\s*\+|\bSST\s+interneuron\b", "somatostatin interneuron", "cell_type"),
    (r"\bVIP\s*\+|\bVIP\s+interneuron\b", "VIP interneuron", "cell_type"),
    (r"\binterneuron\b|\bGABAergic\s+neuron\b|\binhibitory\s+neuron\b", "interneuron", "cell_type"),
    (r"\bplace\s+cell\b|\bplace.modulated\b", "place cell", "cell_type"),
    (r"\bgrid\s+cell\b", "grid cell", "cell_type"),
    (r"\bhead.direction\s+cell\b|\bHD\s+cell\b", "head direction cell", "cell_type"),
    (r"\bborder\s+cell\b", "border cell", "cell_type"),
    (r"\bspeed\s+cell\b", "speed cell", "cell_type"),
    (r"\bdopamine\s+neuron\b|\bdopaminergic\b|\bDA\s+neuron\b", "dopaminergic neuron", "cell_type"),
    (r"\bserotonergic\b|\b5-HT\s+neuron\b|\braphe\s+neuron\b", "serotonergic neuron", "cell_type"),
    (r"\bcholinergic\b|\bcholine\s+acetyltransferase\b|\bChAT\b", "cholinergic neuron", "cell_type"),
    (r"\bnoradrenergic\b|\bnorepinephrine\s+neuron\b|\bLC\s+neuron\b", "noradrenergic neuron", "cell_type"),
    (r"\bastrocyte\b|\bGFAP\b|\bglial\b", "astrocyte", "cell_type"),
    (r"\bmicroglia\b|\bmicroglial\b|\bIba1\b", "microglia", "cell_type"),
    (r"\bolivodendrocyte\b|\bMBP\b|\bmyelin\b", "oligodendrocyte", "cell_type"),
    (r"\bgranule\s+cell\b|\bDG\s+granule\b", "granule cell", "cell_type"),

    # -----------------------------------------------------------------------
    # FREQUENCY BANDS / OSCILLATIONS
    # -----------------------------------------------------------------------
    (r"\btheta\s+oscillation\b|\btheta\s+rhythm\b|\btheta\s+wave\b|\btheta\b", "theta", "signal"),
    (r"\bgamma\s+oscillation\b|\bgamma\s+band\b|\bgamma\b", "gamma", "signal"),
    (r"\bbeta\s+oscillation\b|\bbeta\s+band\b|\bbeta\b", "beta", "signal"),
    (r"\balpha\s+oscillation\b|\balpha\s+band\b|\balpha\b", "alpha", "signal"),
    (r"\bdelta\s+oscillation\b|\bdelta\s+band\b|\bdelta\b", "delta", "signal"),
    (r"\bripple\b|\bsharp.wave\s+ripple\b|\bSWR\b", "ripple", "signal"),
    (r"\bsigma\b|\bsleep\s+spindle\b|\bspindle\b", "sleep spindle", "signal"),
    (r"\bcross.frequency\s+coupling\b|\bphase.amplitude\s+coupling\b|\bCFC\b|\bPAC\b", "cross-frequency coupling", "signal"),

    # -----------------------------------------------------------------------
    # NEURAL SIGNALS / RECORDING MODALITIES
    # -----------------------------------------------------------------------
    (r"\bLFP\b|\blocal\s+field\s+potential\b|\bfield\s+potential\b", "LFP", "signal"),
    (r"\bspike\s+rate\b|\bspike\s+train\b|\bspike\b|\bunit\s+activity\b|\bfiring\s+rate\b", "spike", "signal"),
    (r"\bEEG\b|\belectroencephalog\w*\b", "EEG", "signal"),
    (r"\bfMRI\b|\bBOLD\b|\bfunctional\s+MRI\b", "fMRI/BOLD", "signal"),
    (r"\bcalcium\s+imag\w*\b|\bCa2\+\s+imag\w*\b|\bGCaMP\b|\bGCamp\b", "calcium imaging", "signal"),
    (r"\bneuropixels\b|\bsilicon\s+probe\b|\bpolytrode\b", "Neuropixels", "signal"),
    (r"\bpatch.clamp\b|\bwhole.cell\s+recording\b", "patch clamp", "signal"),
    (r"\bvoltage\s+imag\w*\b|\bVSDI\b", "voltage imaging", "signal"),
    (r"\bECoG\b|\belectrocorticog\w*\b", "ECoG", "signal"),
    (r"\btetrode\b|\bstereotrode\b", "tetrode", "signal"),
    (r"\btwo.photon\b|\b2p\s+imag\w*\b|\b2-photon\b", "two-photon imaging", "signal"),
    (r"\bcoherence\b|\bspike.field\s+coherence\b|\bLFP.spike\b", "coherence", "signal"),
    (r"\boscillation\b|\boscillatory\b|\brhythm\b", "oscillation", "signal"),
    (r"\bsynchrony\b|\bsynchroniz\w*\b|\bcoordinated\s+activ\w*\b", "synchrony", "signal"),
    (r"\bpower\s+spectrum\b|\bspectral\s+power\b|\bpower\b", "spectral power", "signal"),

    # -----------------------------------------------------------------------
    # NEUROTRANSMITTERS / NEUROMODULATORS
    # -----------------------------------------------------------------------
    (r"\bdopamine\b|\bDA\b|\bD1\b|\bD2\s+receptor\b", "dopamine", "neuromodulator"),
    (r"\bserotonin\b|\b5-HT\b|\bSSRI\b", "serotonin", "neuromodulator"),
    (r"\bacetylcholine\b|\bACh\b|\bnicotinic\b|\bmuscarinic\b", "acetylcholine", "neuromodulator"),
    (r"\bnorepinephrine\b|\bnoradrenaline\b|\bNE\b|\bNOR\b", "norepinephrine", "neuromodulator"),
    (r"\bGABA\b|\bgamma.amino\w*\b|\binhibitory\s+transmitter\b", "GABA", "neuromodulator"),
    (r"\bglutamate\b|\bNMDA\b|\bAMPA\b|\bglutamatergic\b", "glutamate", "neuromodulator"),
    (r"\boxytocin\b|\bOT\b", "oxytocin", "neuromodulator"),
    (r"\bendocannabinoid\b|\bCB1\b|\bTHC\b|\beCB\b", "endocannabinoid", "neuromodulator"),
    (r"\bopioid\b|\bmorphine\b|\bmu.opioid\b|\bendorphin\b", "opioid", "neuromodulator"),
    (r"\bcortisol\b|\bstress\s+hormone\b|\bCRH\b|\bHPA\b", "cortisol/HPA", "neuromodulator"),

    # -----------------------------------------------------------------------
    # SYNAPTIC PLASTICITY
    # -----------------------------------------------------------------------
    (r"\bLTP\b|\blong.term\s+potentiation\b|\bpotentiat\w*\b", "LTP", "plasticity"),
    (r"\bLTD\b|\blong.term\s+depression\b|\bdepressed\s+synapse\b", "LTD", "plasticity"),
    (r"\bSTDP\b|\bspike.timing.dependent\s+plasticity\b", "STDP", "plasticity"),
    (r"\bHebbian\b|\bassociative\s+plasticity\b", "Hebbian plasticity", "plasticity"),
    (r"\bsynaptic\s+facilitation\b|\bshort.term\s+facilitation\b|\bSTP\b", "short-term plasticity", "plasticity"),
    (r"\bhomeostatic\s+plasticity\b|\bsynaptic\s+scaling\b", "homeostatic plasticity", "plasticity"),
    (r"\bdendritic\s+spine\b|\bspine\s+densit\w*\b|\bspinogenesis\b", "structural plasticity", "plasticity"),
    (r"\bneurogenesis\b|\bnewborn\s+neuron\b|\badult.born\b", "neurogenesis", "plasticity"),

    # -----------------------------------------------------------------------
    # TASKS / BEHAVIORAL PARADIGMS
    # -----------------------------------------------------------------------
    (r"\bworking\s+memory\b|\bdelay\s+match\w*\b|\bdelay\s+period\b", "working memory", "task"),
    (r"\bfear\s+conditioning\b|\bfear\s+learning\b|\bCFC\b|\bcontextual\s+fear\b", "fear conditioning", "task"),
    (r"\bextinction\b|\bfear\s+extinction\b|\bconditioned\s+fear\b", "extinction", "task"),
    (r"\bnavigation\b|\bspatial\s+navigation\b|\bMorris\s+water\s+maze\b|\bMWM\b", "navigation", "task"),
    (r"\bspatial\s+memory\b|\bplace\s+memory\b|\bspatial\s+task\b", "spatial memory", "task"),
    (r"\bdecision\s+making\b|\bchoice\s+behavior\b|\bdecision\b", "decision making", "task"),
    (r"\battention\b|\bsustained\s+attention\b|\bselectiv\w+\s+attention\b", "attention", "task"),
    (r"\breward\s+learning\b|\breward.based\b|\breward\s+processing\b|\breward\b", "reward", "task"),
    (r"\bprediction\s+error\b|\bRPE\b|\bvalue\s+prediction\b", "prediction error", "task"),
    (r"\breinforcement\s+learning\b|\bRL\b|\bQ.learning\b|\btemporal\s+difference\b", "reinforcement learning", "task"),
    (r"\brecognition\s+memory\b|\bnovel\s+object\b|\bNOR\b", "recognition memory", "task"),
    (r"\bepis\w*\s+memory\b|\bepisodic\b", "episodic memory", "task"),
    (r"\bsemantic\s+memory\b|\bconcept\s+learning\b", "semantic memory", "task"),
    (r"\boperant\s+conditioning\b|\blever\s+press\b|\binstrumental\b", "operant conditioning", "task"),
    (r"\bclassical\s+conditioning\b|\bpavlovian\b|\bCS\b|\bUS\b", "classical conditioning", "task"),
    (r"\breversal\s+learning\b|\brule\s+switching\b|\bcognitive\s+flexibility\b", "reversal learning", "task"),
    (r"\bgo\s+no.go\b|\bresponse\s+inhibition\b|\bstop.signal\b|\bGNG\b", "go/no-go", "task"),
    (r"\bpath\s+integration\b|\bself.motion\b|\bdeadreckon\w*\b", "path integration", "task"),
    (r"\bsleep\s+consolidation\b|\bmemory\s+consolidation\b|\bconsolidation\b", "memory consolidation", "task"),
    (r"\breconsolidation\b|\bmemory\s+update\b", "reconsolidation", "task"),
    (r"\blearning\b|\bconditioned\b|\bacquisition\b", "learning", "task"),

    # -----------------------------------------------------------------------
    # BEHAVIORAL MEASURES
    # -----------------------------------------------------------------------
    (r"\baccurac\w*\b|\bhit\s+rate\b|\bcorrect\s+response\b", "accuracy", "behavior"),
    (r"\breaction\s+time\b|\bresponse\s+time\b|\blatency\b|\bRT\b", "reaction time", "behavior"),
    (r"\bfreezing\b|\bimmobilit\w*\b|\bfear\s+response\b", "freezing", "behavior"),
    (r"\blocomotion\b|\brunning\s+speed\b|\bvelocity\b|\bmovement\s+speed\b", "locomotion", "behavior"),
    (r"\bsocial\s+behavior\b|\bsocial\s+interact\w*\b", "social behavior", "behavior"),
    (r"\bsleep\b|\bNREM\b|\bREM\s+sleep\b|\bwake\b", "sleep/wake", "behavior"),
    (r"\barousal\b|\bvigilance\b|\balertness\b", "arousal", "behavior"),
    (r"\banxiety\b|\bopen\s+field\b|\belevated\s+plus\s+maze\b|\bEPM\b", "anxiety", "behavior"),
    (r"\baggression\b|\bfight\b|\bterritorial\b", "aggression", "behavior"),
    (r"\bpain\b|\bnociception\b|\bhyperalges\w*\b", "pain", "behavior"),

    # -----------------------------------------------------------------------
    # EXPERIMENTAL METHODS
    # -----------------------------------------------------------------------
    (r"\boptogenetics\b|\boptogenetic\b|\bchannel\s*rhodopsin\b|\bChR2\b|\bArchT\b|\bHalorhodopsin\b", "optogenetics", "method"),
    (r"\bchemogenetics\b|\bDREADD\b|\bhM3Dq\b|\bhM4Di\b|\bCNO\b", "chemogenetics", "method"),
    (r"\bpharmacolog\w*\b|\binfusion\b|\binjection\b|\bdrug\b|\bantagonist\b|\bagonist\b", "pharmacology", "method"),
    (r"\blesion\b|\bablation\b|\binactivation\b|\bknockout\b|\bknockdown\b", "lesion/inactivation", "method"),
    (r"\bCRISPR\b|\bgene\s+editing\b|\bCas9\b", "CRISPR", "method"),
    (r"\btransgenic\b|\bknock.in\b|\bcre.dependent\b|\bcre-lox\b", "transgenic", "method"),
    (r"\bfMRI\s+decoding\b|\bMVPA\b|\bclassif\w+\s+fMRI\b", "fMRI decoding", "method"),
    (r"\bSVM\b|\bsupport\s+vector\b", "SVM decoder", "method"),
    (r"\bLDA\b|\blinear\s+discriminant\b", "LDA decoder", "method"),
    (r"\bBCI\b|\bbrain.computer\s+interface\b|\bneuroprosthetic\b", "BCI", "method"),
    (r"\bpopulation\s+decod\w*\b|\bensemble\s+decod\w*\b", "population decoding", "method"),
    (r"\bcross.validation\b|\bleave.one.out\b|\bk.fold\b", "cross-validation", "method"),
    (r"\bprincipal\s+component\b|\bPCA\b", "PCA", "method"),
    (r"\bindependent\s+component\b|\bICA\b", "ICA", "method"),
    (r"\bGranger\s+causalit\w*\b|\bGranger\b", "Granger causality", "method"),

    # -----------------------------------------------------------------------
    # COMPUTATIONAL MODELS
    # -----------------------------------------------------------------------
    (r"\bintegrate.and.fire\b|\bIF\s+neuron\b|\bIAF\b", "integrate-and-fire", "model"),
    (r"\bHodgkin.Huxley\b|\bconductance.based\s+model\b", "Hodgkin-Huxley", "model"),
    (r"\brecurrent\s+neural\s+network\b|\bRNN\b|\bLSTM\b", "RNN", "model"),
    (r"\battractor\s+dynamics\b|\battractor\s+network\b|\bHopfield\b", "attractor model", "model"),
    (r"\bdrift.diffusion\b|\bDDM\b|\baccumulator\s+model\b", "drift-diffusion model", "model"),
    (r"\bBayesian\s+model\b|\bBayesian\s+brain\b|\bprobabilistic\s+inference\b", "Bayesian model", "model"),
    (r"\bpredictive\s+coding\b|\bfree.energy\s+principle\b|\bprecision.weighted\b", "predictive coding", "model"),
    (r"\bmean.field\b|\brate\s+model\b|\bneural\s+mass\b|\bWilson.Cowan\b", "mean-field model", "model"),
    (r"\bspiking\s+network\b|\bspiking\s+neuron\b", "spiking network", "model"),
    (r"\breservoir\s+computing\b|\becho\s+state\b", "reservoir computing", "model"),

    # -----------------------------------------------------------------------
    # NEUROLOGICAL / PSYCHIATRIC CONDITIONS
    # -----------------------------------------------------------------------
    (r"\bAlzheimer\w*\b|\bAD\b|\bamyloid\b|\btau\s+pathol\w*\b", "Alzheimer's disease", "disorder"),
    (r"\bParkinson\w*\b|\bPD\b|\bsymptom\s+rigidity\b|\btremor\b|\blewy\s+body\b", "Parkinson's disease", "disorder"),
    (r"\bdepression\b|\bMDD\b|\bmelancholia\b|\bdepressive\b", "depression", "disorder"),
    (r"\banxiety\s+disorder\b|\bGAD\b|\bPTSD\b|\btrauma\b", "anxiety disorder", "disorder"),
    (r"\bschizophrenia\b|\bpsychosis\b|\bhallucination\b|\bpsychotic\b", "schizophrenia", "disorder"),
    (r"\bepileps\w*\b|\bseizure\b|\bictal\b|\binter.ictal\b", "epilepsy", "disorder"),
    (r"\bautism\b|\bASD\b|\bautism\s+spectrum\b", "autism", "disorder"),
    (r"\bOCD\b|\bobsessive.compulsive\b", "OCD", "disorder"),
    (r"\baddiction\b|\bsubstance\s+use\b|\bdrug\s+dependence\b|\bcraving\b", "addiction", "disorder"),
    (r"\bstroke\b|\bischemia\b|\bcerebral\s+infar\w*\b|\bTIA\b", "stroke/ischemia", "disorder"),
    (r"\bTBI\b|\btraumatic\s+brain\s+injur\w*\b|\bconcussion\b", "TBI", "disorder"),
    (r"\bADHD\b|\battention\s+deficit\b|\bhyperactiv\w*\b", "ADHD", "disorder"),

    # -----------------------------------------------------------------------
    # SPECIES
    # -----------------------------------------------------------------------
    (r"\bmouse\b|\bmice\b|\bmurine\b|\bMus\s+musculus\b", "mouse", "species"),
    (r"\brat\b|\brats\b|\bRattus\b|\bWistar\b|\bSprague.Dawley\b", "rat", "species"),
    (r"\bprimate\b|\bmonkey\b|\bmacaque\b|\brhesus\b|\bcynomolgus\b", "primate", "species"),
    (r"\bhuman\b|\bhuman\s+subject\b|\bparticipant\b|\bpatient\b", "human", "species"),
    (r"\bzebrafish\b|\bdanio\s+rerio\b", "zebrafish", "species"),
    (r"\bDrosophila\b|\bfly\b|\bfruit\s+fly\b", "Drosophila", "species"),
]

# Compile patterns once for efficiency
_COMPILED: list[tuple[re.Pattern, str, str]] = [
    (re.compile(pat, re.IGNORECASE), label, ctype)
    for pat, label, ctype in CONCEPT_VOCAB
]

# All canonical labels and their types (for graph node typing)
ALL_CONCEPTS = [label for _, label, _ in CONCEPT_VOCAB]
_LABEL_TYPE: dict[str, str] = {label: ctype for _, label, ctype in CONCEPT_VOCAB}


def _relpath(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def tokenize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def extract_concepts(text: str) -> list[str]:
    """Return deduplicated canonical concept labels found in text."""
    seen: set[str] = set()
    found = []
    for compiled_pat, label, _ in _COMPILED:
        if label not in seen and compiled_pat.search(text):
            seen.add(label)
            found.append(label)
    return found


def load_findings(path: Path, max_records: int) -> list[dict]:
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_records:
                break
            if line.strip():
                records.append(json.loads(line))
    return records


def build_cooccurrence(records: list[dict]) -> tuple[dict, Counter]:
    """Build same-field concept co-occurrence from structured finding records."""
    edge_weights: Counter = Counter()
    concept_counts: Counter = Counter()

    for r in records:
        # Aggregate all concept-bearing fields into one sentence context
        text_parts = [
            r.get("finding_text") or "",
            " ".join(r.get("regions") or []),
            " ".join(r.get("tasks") or []),
            " ".join(r.get("modalities") or []),
            " ".join(r.get("species") or []),
        ]
        text = " ".join(text_parts)
        concepts = extract_concepts(text)
        for c in concepts:
            concept_counts[c] += 1
        for a, b in combinations(sorted(concepts), 2):
            edge_weights[(a, b)] += 1

    return concept_counts, edge_weights


def build_graph_json(concept_counts: Counter, edge_weights: Counter, min_edge_weight: int = 2) -> dict:
    nodes = [
        {"id": c, "count": n, "type": _LABEL_TYPE.get(c, "unknown")}
        for c, n in concept_counts.most_common()
    ]
    edges = [
        {"source": a, "target": b, "weight": w, "relation": "co_occurs_in_finding"}
        for (a, b), w in edge_weights.items()
        if w >= min_edge_weight
    ]
    return {"nodes": nodes, "edges": edges}


def what_this_baseline_cannot_answer() -> list[str]:
    return [
        "Which dataset can I download to test this claim? (no dataset nodes)",
        "Does this paper report increase or decrease? (unsigned edges)",
        "Did co-occurrence result from A inhibiting B vs. A activating B? (untyped)",
        "What analysis can I run on this dataset? (no affordance nodes)",
        "Is this evidence strong or a null result? (no negation modeling)",
        "Does this dataset have raw or processed data? (no data format nodes)",
        "Was the finding replicated or contradicted? (no polarity)",
        "Is this a hard negative for my query? (no hard-negative inference)",
        "What species / frequency band / temporal pattern? (not extracted into graph)",
    ]


def render_summary(
    n_records: int,
    n_concepts: int,
    n_edges: int,
    top_concepts: list[tuple[str, int]],
    top_edges: list[tuple[tuple[str, str], int]],
    cannot_answer: list[str],
    path_in: Path,
) -> str:
    lines = [
        "# BrainKnow-Style Co-Occurrence Baseline",
        f"\n_Generated: {datetime.now(UTC).isoformat()}_",
        "",
        "## Method",
        "",
        "Same-sentence concept co-occurrence following Wang et al. (2024) BrainKnow.",
        f"Source: `{_relpath(path_in)}`",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Input records | {n_records:,} |",
        f"| Concept vocabulary | {len(CONCEPT_VOCAB)} patterns → {len(set(ALL_CONCEPTS))} canonical concepts (regions, signals, tasks, cell types, neuromodulators, disorders, species, methods, models) |",
        f"| Unique concepts found | {n_concepts:,} |",
        f"| Undirected weighted edges (≥2 co-occurrences) | {n_edges:,} |",
        "",
        "## Top Concepts",
        "",
        "| Concept | Count |",
        "|---|---|",
    ]
    for c, n in top_concepts[:20]:
        lines.append(f"| {c} | {n:,} |")

    lines += [
        "",
        "## Top Co-Occurring Pairs",
        "",
        "| Pair | Weight |",
        "|---|---|",
    ]
    for (a, b), w in top_edges[:20]:
        lines.append(f"| {a} ↔ {b} | {w:,} |")

    lines += [
        "",
        "## What This Baseline Cannot Answer (vs. Neural Search Typed Graph)",
        "",
        "These are queries that co-occurrence alone cannot resolve,",
        "illustrating the motivation for Neural Search's typed relation graph:",
        "",
    ]
    for q in cannot_answer:
        lines.append(f"- {q}")

    lines += [
        "",
        "## BrainKnow Comparison",
        "",
        "| Dimension | BrainKnow | This Baseline | Neural Search |",
        "|---|---|---|---|",
        "| Scale | 3.6M edges / 1.8M papers | ~edges from findings | typed graph 31,920 edges |",
        "| Edge type | undirected co-occurrence | undirected co-occurrence | typed (supports/contradicts/records/affords…) |",
        "| Dataset nodes | no | no | yes (7,171 datasets) |",
        "| Polarity/negation | no | no | partially implemented |",
        "| Analysis affordances | no | no | yes (21 types) |",
        "| Frequency/temporal | no | no | planned (Task 8) |",
        "| Retrieval target | concepts | concepts | reusable datasets |",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-records", type=int, default=80_000, help="Max findings to process")
    parser.add_argument("--min-edge-weight", type=int, default=2, help="Min co-occurrence count for edge")
    parser.add_argument("--file", type=str, default=None, help="Path to findings JSONL (overrides auto-detect)")
    args = parser.parse_args(argv)

    if args.file:
        path_in = Path(args.file)
    else:
        # Prefer ollama (larger) over normalized
        path_in = FINDINGS_OLLAMA if FINDINGS_OLLAMA.exists() else FINDINGS_NORMALIZED
    if not path_in.exists():
        print(f"✗ No findings file found. Tried:\n  {FINDINGS_OLLAMA}\n  {FINDINGS_NORMALIZED}")
        return

    print(f"Loading up to {args.max_records:,} records from {_relpath(path_in)} ...")
    records = load_findings(path_in, args.max_records)
    print(f"  Loaded {len(records):,} records")

    print("Building co-occurrence graph ...")
    concept_counts, edge_weights = build_cooccurrence(records)
    print(f"  Concepts found: {len(concept_counts)}")
    print(f"  Raw pairs: {len(edge_weights)}")

    graph = build_graph_json(concept_counts, edge_weights, args.min_edge_weight)
    n_edges = len(graph["edges"])
    n_nodes = len(graph["nodes"])
    print(f"  Graph: {n_nodes} nodes / {n_edges} edges (min_weight={args.min_edge_weight})")

    top_concepts = concept_counts.most_common(20)
    top_edges = edge_weights.most_common(20)
    cannot_answer = what_this_baseline_cannot_answer()

    GRAPH_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_OUT, "w") as f:
        json.dump({
            "meta": {
                "generated_at": datetime.now(UTC).isoformat(),
                "method": "same_field_co_occurrence",
                "source": _relpath(path_in),
                "n_records": len(records),
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "min_edge_weight": args.min_edge_weight,
                "concept_vocabulary_size": len(ALL_CONCEPTS),
                "cannot_answer": cannot_answer,
            },
            **graph,
        }, f, indent=2)

    summary = render_summary(
        len(records), n_nodes, n_edges,
        top_concepts, top_edges, cannot_answer, path_in
    )
    with open(SUMMARY_MD, "w") as f:
        f.write(summary)

    print(f"✓ Graph   → {_relpath(GRAPH_OUT)}")
    print(f"✓ Summary → {_relpath(SUMMARY_MD)}")
    print(f"\nTop concept pairs:")
    for (a, b), w in top_edges[:5]:
        print(f"  {a} ↔ {b}: {w}")


if __name__ == "__main__":
    main()
