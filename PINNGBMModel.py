import shutil
import tempfile
import copy
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Optional
from loguru import logger
import os

from Engine import Engine
from Options import Options

class PINNGBMModel:
    def __init__(
        self, 
        algorithm: str = "pinngbm", 
        cuda_device: Optional[str] = "0", 
        force_cpu: bool = False
    ):
        self.algorithm = algorithm
        self.cuda_device = cuda_device
        self.force_cpu = force_cpu

    def predict_single(
        self,
        gm: Path,
        wm: Path,
        csf: Path,
        tumorseg: Path,
        outdir: Path,
        pet: Optional[Path] = None,
        log_file: Optional[Path] = None,
    ) -> None:
        logger.info(f"Starting PINN-GBM prediction for output: {outdir}")
        
        # 1. Configure Options
        gbmopts = Options()
        # Set device
        if self.force_cpu:
            gbmopts.opts['device'] = 'cpu'
        else:
            gbmopts.opts['device'] = f'cuda:{self.cuda_device}' if self.cuda_device else 'cuda'

        # Configure Output
        gbmopts.opts['logger_opts']['use_mlflow'] = False
        gbmopts.opts['logger_opts']['use_tmp'] = True
        gbmopts.opts['logger_opts']['use_stdout'] = True
        gbmopts.opts['skip_fig'] = True
        
        # temporary for debugging
        gbmopts.opts['flags'] = 'small'

        # for debugging with mlflow
        # gbmopts.opts['logger_opts']['use_mlflow'] = True
        # gbmopts.opts['logger_opts']['experiment_name'] = 'pipelinetest'
        # get the last part of outdir as run_name
        # gbmopts.opts['logger_opts']['run_name'] = outdir.name 


        # 2. Prepare Input Data
        gbmopts.opts['dataset_opts']['gm_file'] = str(gm)
        gbmopts.opts['dataset_opts']['wm_file'] = str(wm)
        gbmopts.opts['dataset_opts']['csf_file'] = str(csf)
        gbmopts.opts['dataset_opts']['segpre_file'] = str(tumorseg)
        
        # 3. Initialize and Run Engine
        try:
            gbmopts.process_options()
            engine = Engine(gbmopts.opts)
            
            # Execute pipeline steps explicitly
            engine.run()
            phiu_final = engine.vdict['upred_fdm'][..., -1]
            
            # 4. Convert Output to NIfTI
            self._save_nifti_output(phiu_final, outdir, wm, self.algorithm)

            engine.logger.close()
            
        except Exception as e:
            logger.error(f"PINN-GBM execution failed: {e}")
            raise e
                
        logger.info(f"Finished PINN-GBM prediction. Output saved to {outdir}")

    def _save_nifti_output(self, u_pred: np.ndarray, outdir: Path, ref_nifti_path: Path, algorithm: str):
        # Get geometry from reference NIfTI (e.g., WM)
        ref_img = nib.load(ref_nifti_path)
        affine = ref_img.affine
        header = ref_img.header

        # Create NIfTI image
        nifti_img = nib.Nifti1Image(u_pred, affine, header)
        
        # Save as expected by PredictGBM structure
        final_out_dir = outdir / "growth_models" / algorithm
        final_out_dir.mkdir(parents=True, exist_ok=True)
        out_name = final_out_dir / f"{algorithm}_pred.nii.gz"
        
        nib.save(nifti_img, out_name)
        logger.info(f"Saved NIfTI prediction to {out_name}")

def predict_tumor_growth(
    tumorseg_file: Path,
    gm_file: Path,
    wm_file: Path,
    csf_file: Path,
    model_id: str,
    outdir: Path,
    cuda_device: Optional[str] = "0",
) -> None:
    """
    Predict tumor cell concentration with PINNGBM model.
    """
    logger.info(f"Starting growth prediction with {model_id}.")

    model = PINNGBMModel(algorithm=model_id, cuda_device=cuda_device)
    model.predict_single(
        gm=gm_file,
        wm=wm_file,
        csf=csf_file,
        tumorseg=tumorseg_file,
        outdir=outdir
    )
    logger.info(f"Finished growth prediction, output saved to {outdir}.")
