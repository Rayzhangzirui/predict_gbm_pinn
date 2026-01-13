#!/bin/bash
# Test PINNGBM using Singularity
# Assumes SIF file is already built (e.g., at /scratch/ziruz16/pinngbm.sif)

set -e

# Check if Singularity is available
if ! command -v singularity &> /dev/null; then
    echo "Error: Singularity is not installed or not in PATH"
    exit 1
fi

# Source directory (contains files like gm_pbmap.nii.gz, wm_pbmap.nii.gz, etc.)
SOURCE_DIR="${1:-../data/predict_gbm_datasets_3/GLIODIL/data_001}"
SUBJECT_ID="${2:-00000}"

# SIF file location (default to cluster location, can be overridden)
SIF_FILE="${SIF_FILE:-/scratch/ziruz16/pinngbm.sif}"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory not found: $SOURCE_DIR"
    echo "Usage: $0 [source_dir] [subject_id] [sif_file]"
    echo "  source_dir: Directory containing gm_pbmap.nii.gz, wm_pbmap.nii.gz, etc."
    echo "  subject_id: Subject ID (default: 00000)"
    echo "  sif_file: Path to SIF file (default: /scratch/ziruz16/pinngbm.sif)"
    echo "            Can also set via SIF_FILE environment variable"
    exit 1
fi

if [ ! -f "$SIF_FILE" ]; then
    echo "Error: Singularity image not found: $SIF_FILE"
    echo "Please build it first:"
    echo "  singularity build $SIF_FILE docker://ghcr.io/rayzhangzirui/pinngbm:gpu-latest"
    exit 1
fi

# Set up directories
WORK_DIR="${WORK_DIR:-/scratch/ziruz16/pinngbm_test}"
INPUT_DIR="${WORK_DIR}/input"
OUTPUT_DIR="${WORK_DIR}/output"
PATIENT_DIR="$INPUT_DIR/Patient-${SUBJECT_ID}"

# Clean up old input directory
rm -rf "$INPUT_DIR"
mkdir -p "$PATIENT_DIR" "$OUTPUT_DIR"

echo "============================================================"
echo "Preparing data for Singularity container"
echo "============================================================"
echo "Source directory: $SOURCE_DIR"
echo "Subject ID: $SUBJECT_ID"
echo "SIF file: $SIF_FILE"
echo ""

# Copy and rename files according to Docker interface
# Expected: Patient-00000/00000-{modality}.nii.gz
copy_and_rename() {
    local src_file="$1"
    local dst_file="$2"
    local src_path="$SOURCE_DIR/$src_file"
    local dst_path="$PATIENT_DIR/$dst_file"
    
    if [ -f "$src_path" ]; then
        cp "$src_path" "$dst_path"
        echo "  Copied: $src_file -> $dst_file"
        return 0
    else
        echo "  Warning: $src_file not found in $SOURCE_DIR"
        return 1
    fi
}

missing_files=()
copy_and_rename "gm_pbmap.nii.gz" "${SUBJECT_ID}-gm.nii.gz" || missing_files+=("gm_pbmap.nii.gz")
copy_and_rename "wm_pbmap.nii.gz" "${SUBJECT_ID}-wm.nii.gz" || missing_files+=("wm_pbmap.nii.gz")
copy_and_rename "csf_pbmap.nii.gz" "${SUBJECT_ID}-csf.nii.gz" || missing_files+=("csf_pbmap.nii.gz")
copy_and_rename "tumor_seg.nii.gz" "${SUBJECT_ID}-tumorseg.nii.gz" || missing_files+=("tumor_seg.nii.gz")

if [ ${#missing_files[@]} -gt 0 ]; then
    echo ""
    echo "Error: Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    exit 1
fi

echo ""
echo "============================================================"
echo "Testing PINNGBM with Singularity"
echo "============================================================"
echo "Singularity image: $SIF_FILE"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run Singularity container
export SINGULARITYENV_PINNGBM_SUBJECT_ID="${SUBJECT_ID}"
export SINGULARITYENV_PINNGBM_FORCE_CPU=0  # Use GPU on cluster

singularity run --nv \
    --bind "$INPUT_DIR:/mlcube_io0:ro" \
    --bind "$OUTPUT_DIR:/mlcube_io1:rw" \
    "$SIF_FILE" \
    "${@:3}"  # Pass through any additional arguments

echo ""
echo "============================================================"
echo "Singularity test completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "============================================================"
echo ""
echo "To clean up temporary input directory:"
echo "  rm -rf $INPUT_DIR"
