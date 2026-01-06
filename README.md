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