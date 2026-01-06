import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Add PredictGBM to path if not installed
sys.path.insert(0, str(Path(__file__).parent.parent / "PredictGBM"))


from predict_gbm.utils.parsing import PatientDataset
from predict_gbm.utils.constants import PREDICT_GBM_DIR
from predict_gbm.evaluation import evaluate_tumor_model

# Import the local predict_tumor_growth
from PINNGBMModel import predict_tumor_growth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    model_id = "pinngbm"
    outdir = Path("predict_eval")
    outdir.mkdir(parents=True, exist_ok=True)
    
    
    predict_gbm_rootdir = "../data/" 

    predict_gbm_dataset = PatientDataset()
    predict_gbm_dataset.load('test_subset.json')
    predict_gbm_dataset.set_root_dir(predict_gbm_rootdir)

    all_results = []

    for patient in predict_gbm_dataset:
        print(f"Processing patient: {patient['patient_id']}")

        patient_outdir = outdir / patient['patient_id']
        patient_outdir.mkdir(parents=True, exist_ok=True)

        derivatives = patient["derivatives"]

        predict_tumor_growth(
            tumorseg_file=derivatives["tumor_seg"],
            gm_file=derivatives["gm_pbmap"],
            wm_file=derivatives["wm_pbmap"],
            csf_file=derivatives["csf_pbmap"],
            model_id=model_id,
            outdir=patient_outdir,
            cuda_device=args.cuda_device
        )

        pred_file = patient_outdir / f"growth_models/{model_id}/{model_id}_pred.nii.gz"
        
        # Check if recurrence file exists
        recurrence_file = derivatives.get("recurrence_seg")
        if not recurrence_file or not Path(recurrence_file).exists():
             print(f"Skipping evaluation for {patient['patient_id']}: No recurrence file.")
             continue
        
        results, standard_plan_nii, model_plan_nii = evaluate_tumor_model(
            tumorseg_file=derivatives["tumor_seg"],
            recurrence_file=recurrence_file,
            pred_file=pred_file,
            brain_mask_file=derivatives["brain_mask"],
            ctv_margin=15,
        )
        print(f"Results for {patient['patient_id']}: {results}")
        all_results.append(results)

    
    if all_results:
        coverages_standard = [r["recurrence_coverage_standard"] for r in all_results]
        coverages_model = [r["recurrence_coverage_model"] for r in all_results]

        mean_coverage_standard = 100 * np.mean(coverages_standard)
        mean_coverage_model = 100 * np.mean(coverages_model)

        print("Finished evaluation.")
        print(f"Standard plan coverge: {mean_coverage_standard:.2f}")
        print(f"Model plan coverge: {mean_coverage_model:.2f}")
    else:
        print("No results to evaluate.")
