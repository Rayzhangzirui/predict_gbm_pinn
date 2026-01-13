#!/bin/bash
# Test PINNGBM Docker container locally
# Copies and renames data files to match Docker interface requirements

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Docker image exists
IMAGE_NAME="pinngbm_cpu"
IMAGE_TAG="latest"
FULL_TAG="${IMAGE_NAME}:${IMAGE_TAG}"

if ! docker image inspect ${FULL_TAG} &>/dev/null; then
    echo "Error: Docker image ${FULL_TAG} not found."
    echo "Please build the image first using: pinngbmtorch/docker/build_cpu.sh"
    exit 1
fi

# Source directory (contains files like gm_pbmap.nii.gz, wm_pbmap.nii.gz, etc.)
SOURCE_DIR="${1:-../data/predict_gbm_datasets_3/GLIODIL/data_001}"
SUBJECT_ID="${2:-00000}"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory not found: $SOURCE_DIR"
    echo "Usage: $0 [source_dir] [subject_id]"
    echo "  source_dir: Directory containing gm_pbmap.nii.gz, wm_pbmap.nii.gz, etc."
    echo "  subject_id: Subject ID (default: 00000)"
    exit 1
fi

# Set up directories
INPUT_DIR="$PWD/tmp_docker_input"
OUTPUT_DIR="$PWD/tmp_docker_out"
PATIENT_DIR="$INPUT_DIR/Patient-${SUBJECT_ID}"

# Clean up old input directory
rm -rf "$INPUT_DIR"
mkdir -p "$PATIENT_DIR"

echo "============================================================"
echo "Preparing data for Docker container"
echo "============================================================"
echo "Source directory: $SOURCE_DIR"
echo "Subject ID: $SUBJECT_ID"
echo ""

# Copy and rename files according to Docker interface
# Expected: Patient-00000/00000-{modality}.nii.gz
# File mapping: source_file -> destination_file
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
echo "Testing PINNGBM Docker container"
echo "============================================================"
echo "Image: ${FULL_TAG}"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run Docker container
docker run --rm \
    -v "$INPUT_DIR:/mlcube_io0:ro" \
    -v "$OUTPUT_DIR:/mlcube_io1:rw" \
    -e PINNGBM_SUBJECT_ID="${SUBJECT_ID}" \
    -e PINNGBM_FORCE_CPU=1 \
    ${FULL_TAG} \
    flags small,local

echo ""
echo "============================================================"
echo "Docker test completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "============================================================"
echo ""
echo "To clean up temporary input directory:"
echo "  rm -rf $INPUT_DIR"
