#!/bin/bash
# Corpus Expansion Sprint - Batch Ingestion Script
# Usage: ./scripts/run_corpus_expansion.sh [phase]
# Phases: 1 (high priority), 2 (medium priority), 3 (gap filling), all

set -e

PHASE=${1:-1}
LOG_DIR="data/logs/ingestion"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"
mkdir -p "data/raw/dandi"
mkdir -p "data/raw/openneuro"
mkdir -p "data/raw/openalex"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

run_ingestion() {
    local source=$1
    local query=$2
    local limit=$3
    local log_file="$LOG_DIR/${source}_${TIMESTAMP}.log"

    log "Running: $source --query \"$query\" --limit $limit"

    if python -m neural_search.ingestion.$source --query "$query" --limit $limit --save-raw 2>&1 | tee -a "$log_file"; then
        success "Completed: $source query='$query'"
    else
        error "Failed: $source query='$query' - check $log_file"
        echo "FAIL,$source,$query,$limit,$(date -Iseconds)" >> "$LOG_DIR/failures_${TIMESTAMP}.csv"
    fi

    # Rate limiting pause
    sleep 2
}

# =============================================================================
# PHASE 1: HIGH PRIORITY
# =============================================================================
run_phase_1() {
    log "=========================================="
    log "PHASE 1: HIGH PRIORITY INGESTION"
    log "=========================================="

    log "--- DANDI: Decision Making Core ---"
    run_ingestion dandi "go nogo" 25
    run_ingestion dandi "reversal learning" 25
    run_ingestion dandi "visual decision making" 25
    run_ingestion dandi "delay discounting" 15

    log "--- OpenNeuro: Motor & Clinical Core ---"
    run_ingestion openneuro "motor imagery eeg" 25
    run_ingestion openneuro "ieeg seizure" 25
    run_ingestion openneuro "epilepsy" 25
    run_ingestion openneuro "BCI" 15

    log "--- OpenAlex: Foundation Papers ---"
    run_ingestion openalex "reversal learning electrophysiology" 100
    run_ingestion openalex "go nogo task neural" 100
    run_ingestion openalex "brain computer interface motor cortex" 100

    success "Phase 1 complete!"
}

# =============================================================================
# PHASE 2: MEDIUM PRIORITY
# =============================================================================
run_phase_2() {
    log "=========================================="
    log "PHASE 2: MEDIUM PRIORITY INGESTION"
    log "=========================================="

    log "--- DANDI: Extended Coverage ---"
    run_ingestion dandi "motor cortex" 20
    run_ingestion dandi "reaching grasping" 15
    run_ingestion dandi "visual discrimination" 15
    run_ingestion dandi "two-alternative forced choice" 15
    run_ingestion dandi "brain computer interface" 10
    run_ingestion dandi "perceptual decision" 10

    log "--- OpenNeuro: Extended Coverage ---"
    run_ingestion openneuro "motor execution" 20
    run_ingestion openneuro "ecog" 20
    run_ingestion openneuro "reinforcement learning" 20
    run_ingestion openneuro "stop signal" 15
    run_ingestion openneuro "decision making" 15

    log "--- OpenAlex: Extended Papers ---"
    run_ingestion openalex "intracranial eeg epilepsy" 100
    run_ingestion openalex "neural decoding movement" 100
    run_ingestion openalex "visual decision making neural" 100

    success "Phase 2 complete!"
}

# =============================================================================
# PHASE 3: GAP FILLING
# =============================================================================
run_phase_3() {
    log "=========================================="
    log "PHASE 3: GAP FILLING INGESTION"
    log "=========================================="

    log "--- DANDI: Gap Filling ---"
    run_ingestion dandi "probabilistic learning" 10
    run_ingestion dandi "reward prediction" 10
    run_ingestion dandi "epilepsy ieeg" 10
    run_ingestion dandi "neuropixels" 15
    run_ingestion dandi "calcium imaging decision" 10

    log "--- OpenNeuro: Gap Filling ---"
    run_ingestion openneuro "reward learning" 15
    run_ingestion openneuro "intracranial" 10
    run_ingestion openneuro "working memory" 10
    run_ingestion openneuro "attention" 10

    log "--- OpenAlex: Gap Filling ---"
    run_ingestion openalex "delay discounting neuroimaging" 50
    run_ingestion openalex "motor imagery classification" 50
    run_ingestion openalex "ecog seizure prediction" 50
    run_ingestion openalex "neuropixels decision" 50

    success "Phase 3 complete!"
}

# =============================================================================
# SUMMARY REPORT
# =============================================================================
generate_summary() {
    log "=========================================="
    log "GENERATING INGESTION SUMMARY"
    log "=========================================="

    local summary_file="$LOG_DIR/summary_${TIMESTAMP}.md"

    cat > "$summary_file" << EOF
# Corpus Expansion Ingestion Summary

**Timestamp**: $TIMESTAMP
**Phase(s) Run**: $PHASE

## Results

### Log Files
$(ls -la "$LOG_DIR"/*_${TIMESTAMP}.log 2>/dev/null || echo "No logs found")

### Failures
$(cat "$LOG_DIR/failures_${TIMESTAMP}.csv" 2>/dev/null || echo "No failures recorded")

### Raw Data Saved
- DANDI: $(ls data/raw/dandi/*.json 2>/dev/null | wc -l) files
- OpenNeuro: $(ls data/raw/openneuro/*.json 2>/dev/null | wc -l) files
- OpenAlex: $(ls data/raw/openalex/*.json 2>/dev/null | wc -l) files

## Next Steps
1. Run deduplication: \`make corpus-dedupe\`
2. Generate cards: \`make corpus-cards\`
3. Check coverage: \`make corpus-coverage\`
EOF

    log "Summary written to: $summary_file"
    cat "$summary_file"
}

# =============================================================================
# MAIN
# =============================================================================

log "Corpus Expansion Sprint - Batch Ingestion"
log "Phase: $PHASE"
log "Log directory: $LOG_DIR"

case $PHASE in
    1)
        run_phase_1
        ;;
    2)
        run_phase_2
        ;;
    3)
        run_phase_3
        ;;
    all)
        run_phase_1
        run_phase_2
        run_phase_3
        ;;
    *)
        echo "Usage: $0 [1|2|3|all]"
        echo "  1   - Run Phase 1 (High Priority)"
        echo "  2   - Run Phase 2 (Medium Priority)"
        echo "  3   - Run Phase 3 (Gap Filling)"
        echo "  all - Run all phases"
        exit 1
        ;;
esac

generate_summary

log "=========================================="
log "INGESTION COMPLETE"
log "=========================================="
