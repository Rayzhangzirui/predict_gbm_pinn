# PINN-GBM

Physics-Informed Neural Networks for Glioblastoma Growth Modeling.

This repository contains the implementation of [PINN-GBM](https://github.com/Rayzhangzirui/pinngbm), designed to interface with the [PredictGBM](https://github.com/BrainLesion/PredictGBM) framework for evaluation of tumor growth models.

## Integration with PredictGBM

The primary interface for PredictGBM is located in `PINNGBMModel.py`.

*   **`PINNGBMModel`**: This class encapsulates the PINN-GBM pipeline. It handles configuration, data loading, execution of the full inference pipeline, and saving the final prediction as a NIfTI file.
*   **`predict_tumor_growth`**: A wrapper function that matches the signature expected by evaluation scripts. It accepts paths to required NIfTI files (tumor segmentation, tissue probability maps) and an output directory.

### Example usage
```python
from PINNGBMModel import predict_tumor_growth
from pathlib import Path

predict_tumor_growth(
    tumorseg_file=Path("path/to/tumor_seg.nii.gz"),
    gm_file=Path("path/to/gm_prob.nii.gz"),
    wm_file=Path("path/to/wm_prob.nii.gz"),
    csf_file=Path("path/to/csf_prob.nii.gz"),
    model_id="pinngbm",
    outdir=Path("output_directory"),
    cuda_device="0"
)
```

## Running Evaluations

An example script for running evaluations on a dataset is provided in `evaluate_pinngbm.py`.

To run the evaluation:

```bash
python evaluate_pinngbm.py
```

## Requirements

*   PredictGBM 
*   PyTorch
*   NumPy
*   SciPy
*   Mlflow (Optional)

## Tutorial
See [`tutorial.ipynb`](tutorial.ipynb) for a brief tutorial on using PINN-GBM

## Docker Integration for PredictGBM

PINNGBM provides Docker containers compatible with the PredictGBM framework. The containers follow the PredictGBM I/O specification and can be used directly with `predict_tumor_growth()`.

### Dockerfiles

- **`docker/Dockerfile.cpu`**: CPU-only image (for local testing, macOS)
- **`docker/Dockerfile.gpu`**: GPU image with CUDA 11.8 (for cluster use)

Both images use `container_inference.py` as the entrypoint, which reads from `/mlcube_io0/Patient-{subject_id}/` and writes to `/mlcube_io1/{subject_id}.nii.gz`.

### Building Images

Build scripts are located in `docker/`:

**CPU version:**
```bash
cd pinngbmtorch
bash docker/build_cpu.sh
```
Creates `pinngbm_cpu:latest` and saves to `../PredictGBM/predict_gbm/data/models/pinngbm_cpu.tar` (relative to project root).

**GPU version:**
```bash
cd pinngbmtorch
bash docker/build_gpu.sh
```
Creates `pinngbm:latest` and saves to `../PredictGBM/predict_gbm/data/models/pinngbm.tar` (relative to project root).

**Custom output directory:**
```bash
# Relative path (from project root)
PREDICTGBM_MODEL_DIR=custom/models bash docker/build_cpu.sh

# Absolute path
PREDICTGBM_MODEL_DIR=/absolute/path/to/models bash docker/build_cpu.sh
```

**Note:** The default path assumes `PredictGBM/` is a sibling directory to `pinngbmtorch/`. If your structure differs, set `PREDICTGBM_MODEL_DIR` accordingly.

### Using Pre-built Images from GHCR

Pre-built GPU images are available on GitHub Container Registry:

```bash
# Pull directly
docker pull ghcr.io/rayzhangzirui/pinngbm:gpu-latest

# Or for Singularity on clusters
singularity build pinngbm.sif docker://ghcr.io/rayzhangzirui/pinngbm:gpu-latest
```

### Using with PredictGBM

Once the `.tar` file is in `PredictGBM/predict_gbm/data/models/`, use it with:

```python
from predict_gbm.prediction.predict import predict_tumor_growth

predict_tumor_growth(
    tumorseg_file=Path("tumor_seg.nii.gz"),
    gm_file=Path("gm_pbmap.nii.gz"),
    wm_file=Path("wm_pbmap.nii.gz"),
    csf_file=Path("csf_pbmap.nii.gz"),
    model_id="pinngbm_cpu",  # or "pinngbm" for GPU version
    outdir=Path("output"),
)
```

### Testing

Test scripts are provided to verify PINNGBM integration:

- **`test_predictgbm_integration.py`**: Simplified integration test that runs prediction and evaluation on a single patient. Takes an input directory with preprocessed NIfTI files and tests both `predict_tumor_growth()` and `evaluate_tumor_model()`.
- **`scripts/test_singularity.sh`**: Test script for running PINNGBM with Singularity on clusters. Copies and renames data files to match the container interface, then runs the Singularity container.
- **`PredictGBM/tests/prediction/test_docker.py`**: PredictGBM's unit tests for Docker functionality (tests PredictGBM's Docker integration framework, not PINNGBM specifically).

Example usage:
```bash
# Test with local data directory
python test_predictgbm_integration.py --input_dir /path/to/data/directory

# Test with Singularity (on cluster)
bash scripts/test_singularity.sh /path/to/data/directory
```