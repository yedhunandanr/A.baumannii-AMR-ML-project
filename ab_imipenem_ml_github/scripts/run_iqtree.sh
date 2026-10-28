#!/bin/bash

# Set SAMR_PROJECT_ROOT to override the repository root.
PROJECT_ROOT="${SAMR_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Run IQ-TREE2 on all 6 Gubbins filtered alignments
# Run in screen: screen -S iqtree

GUBBINS=${PROJECT_ROOT}/outputs/gubbins
OUTDIR=${PROJECT_ROOT}/outputs/iqtree

mkdir -p $OUTDIR

echo "Starting IQ-TREE2 runs at $(date)"

# Universal — all 801 genomes
echo "Running universal..."
iqtree \
    -s ${GUBBINS}/universal.filtered_polymorphic_sites.fasta \
    -st DNA \
    -m GTR+G \
    -nt 16 \
    -pre ${OUTDIR}/universal \
    -bb 1000 \
    -safe \
    -quiet
echo "Universal done at $(date)"

# ST1
echo "Running ST1..."
iqtree \
    -s ${GUBBINS}/ST1.filtered_polymorphic_sites.fasta \
    -st DNA \
    -m GTR+G \
    -nt 16 \
    -pre ${OUTDIR}/ST1 \
    -bb 1000 \
    -safe \
    -quiet
echo "ST1 done at $(date)"

# ST2
echo "Running ST2..."
iqtree \
    -s ${GUBBINS}/ST2.filtered_polymorphic_sites.fasta \
    -st DNA \
    -m GTR+G \
    -nt 16 \
    -pre ${OUTDIR}/ST2 \
    -bb 1000 \
    -safe \
    -quiet
echo "ST2 done at $(date)"

# ST3
echo "Running ST3..."
iqtree \
    -s ${GUBBINS}/ST3.filtered_polymorphic_sites.fasta \
    -st DNA \
    -m GTR+G \
    -nt 16 \
    -pre ${OUTDIR}/ST3 \
    -bb 1000 \
    -safe \
    -quiet
echo "ST3 done at $(date)"

# ST25
echo "Running ST25..."
iqtree \
    -s ${GUBBINS}/ST25.filtered_polymorphic_sites.fasta \
    -st DNA \
    -m GTR+G \
    -nt 16 \
    -pre ${OUTDIR}/ST25 \
    -bb 1000 \
    -safe \
    -quiet
echo "ST25 done at $(date)"

# ST32
echo "Running ST32..."
iqtree \
    -s ${GUBBINS}/ST32.filtered_polymorphic_sites.fasta \
    -st DNA \
    -m GTR+G \
    -nt 16 \
    -pre ${OUTDIR}/ST32 \
    -bb 1000 \
    -safe \
    -quiet
echo "ST32 done at $(date)"

echo "All IQ-TREE2 runs complete at $(date)"

# Check outputs
echo ""
echo "Output trees:"
for ref in universal ST1 ST2 ST3 ST25 ST32; do
    tree="${OUTDIR}/${ref}.treefile"
    if [ -f "$tree" ]; then
        size=$(du -h "$tree" | cut -f1)
        echo "  $ref: $size ✓"
    else
        echo "  $ref: MISSING"
    fi
done
