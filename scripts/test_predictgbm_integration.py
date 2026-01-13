#!/usr/bin/env python3
"""Simplified test script for PredictGBM integration."""

import argparse
import sys
from pathlib import Path

# Add PredictGBM to path
sys.path.insert(0, str(Path(__file__).parent.parent / "PredictGBM"))

from predict_gbm.prediction import predict_tumor_growth
from predict_gbm.evaluation import evaluate_tumor_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test PINNGBM integration with PredictGBM pipeline"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input files (tumor_seg.nii.gz, gm_pbmap.nii.gz, wm_pbmap.nii.gz, csf_pbmap.nii.gz, etc.)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="predict_eval",
        help="Output directory"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="pinngbm_cpu",
        help="Model ID to use"
    )
    parser.add_argument(
        "--cuda_device",
        type=str,
        default="0",
        help="GPU id to run on"
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation (only run prediction)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build file paths from input_dir
    tumorseg_file = input_dir / "tumor_seg.nii.gz"
    gm_file = input_dir / "gm_pbmap.nii.gz"
    wm_file = input_dir / "wm_pbmap.nii.gz"
    csf_file = input_dir / "csf_pbmap.nii.gz"
    recurrence_file = input_dir / "recurrence_preop.nii.gz"
    brain_mask_file = input_dir / "t1c_bet_mask.nii.gz"

    # Check required files exist
    for f in [tumorseg_file, gm_file, wm_file, csf_file]:
        if not f.exists():
            print(f"Error: Required file not found: {f}")
            sys.exit(1)

    print(f"Running prediction with model: {args.model_id}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {outdir}")

    # Run prediction
    predict_tumor_growth(
        tumorseg_file=tumorseg_file,
        gm_file=gm_file,
        wm_file=wm_file,
        csf_file=csf_file,
        model_id=args.model_id,
        outdir=outdir,
        cuda_device="cpu",
    )

    pred_file = outdir / f"growth_models/{args.model_id}/{args.model_id}_pred.nii.gz"
    print(f"Prediction saved to: {pred_file}")

    # Run evaluation if not skipped and recurrence file exists
    if not args.skip_eval:
        if recurrence_file.exists():
            print(f"\nRunning evaluation...")
            results = evaluate_tumor_model(
                tumorseg_file=tumorseg_file,
                recurrence_file=recurrence_file,
                pred_file=pred_file,
                brain_mask_file=brain_mask_file if brain_mask_file.exists() else None,
                ctv_margin=15,
            )
            print(f"\nEvaluation results:")
            for key, value in results[0].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"\nSkipping evaluation: recurrence file not found ({recurrence_file})")

    print("\nTest completed successfully!")
