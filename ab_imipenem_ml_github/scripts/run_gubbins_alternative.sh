#!/bin/bash

# Set SAMR_PROJECT_ROOT to override the repository root.
PROJECT_ROOT="${SAMR_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Run Gubbins recombination filtering on all 6 core alignments
# Run this in a screen session: screen -S gubbins

SNIPPY_CORE=${PROJECT_ROOT}/outputs/snippy_core
OUTDIR=${PROJECT_ROOT}/outputs/gubbins

mkdir -p $OUTDIR
cd $OUTDIR

echo "Starting Gubbins runs at $(date)"

# Universal — all 801 genomes (slowest)
echo "Running universal..."
run_gubbins.py \
    --prefix universal \
    --threads 16 \
    ${SNIPPY_CORE}/universal.aln
echo "Universal done at $(date)"

# ST1
echo "Running ST1..."
run_gubbins.py \
    --prefix ST1 \
    --threads 16 \
    ${SNIPPY_CORE}/ST1.aln
echo "ST1 done at $(date)"

# ST2
echo "Running ST2..."
run_gubbins.py \
    --prefix ST2 \
    --threads 16 \
    ${SNIPPY_CORE}/ST2.aln
echo "ST2 done at $(date)"

# ST3
echo "Running ST3..."
run_gubbins.py \
    --prefix ST3 \
    --threads 16 \
    ${SNIPPY_CORE}/ST3.aln
echo "ST3 done at $(date)"

# ST25
echo "Running ST25..."
run_gubbins.py \
    --prefix ST25 \
    --threads 16 \
    ${SNIPPY_CORE}/ST25.aln
echo "ST25 done at $(date)"

# ST32
echo "Running ST32..."
run_gubbins.py \
    --prefix ST32 \
    --threads 16 \
    ${SNIPPY_CORE}/ST32.aln
echo "ST32 done at $(date)"

echo "All Gubbins runs complete at $(date)"

# Check outputs
echo ""
echo "Output files:"
for ref in universal ST1 ST2 ST3 ST25 ST32; do
    fasta="$OUTDIR/${ref}.filtered_polymorphic_sites.fasta"
    if [ -f "$fasta" ]; then
        seqs=$(grep -c ">" "$fasta")
        echo "  $ref: $seqs sequences"
    else
        echo "  $ref: MISSING"
    fi
done
